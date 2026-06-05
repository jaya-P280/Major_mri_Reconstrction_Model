from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import shutil
import numpy as np
import cv2
import base64
import os
import h5py

from model import load_model
from utils import run_inference

app = FastAPI()

# Load model
model, device = load_model()

# Mount static folder
app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve HTML
@app.get("/", response_class=HTMLResponse)
async def serve_home():
    with open("static/home.html", "r", encoding="utf-8") as f:
        return f.read()

# Encode image
def encode_image(img):
    _, buffer = cv2.imencode('.png', img)
    return base64.b64encode(buffer).decode('utf-8')


# ─────────────────────────────────────────────────────────────────────────────
# Coil simulation
# ─────────────────────────────────────────────────────────────────────────────

def simulate_coil_sensitivities(H, W, num_coils=16):
    """
    Simulate num_coils sensitivity maps as Gaussians placed around a circle.
    num_coils=16 matches the real fastMRI brain dataset coil count.
    """
    coil_maps = np.zeros((num_coils, H, W), dtype=np.complex64)
    cx, cy = W / 2, H / 2
    radius_x, radius_y = W * 0.6, H * 0.6
    y_grid, x_grid = np.mgrid[0:H, 0:W]

    for i in range(num_coils):
        angle  = 2 * np.pi * i / num_coils
        coil_x = cx + radius_x * np.cos(angle)
        coil_y = cy + radius_y * np.sin(angle)
        sensitivity = np.exp(
            -((x_grid - coil_x) ** 2 / (2 * (W * 0.6) ** 2) +
              (y_grid - coil_y) ** 2 / (2 * (H * 0.6) ** 2))
        )
        phase = np.exp(1j * 0.3 * np.cos(angle) * (x_grid / W + y_grid / H))
        coil_maps[i] = sensitivity * phase

    # Normalize: RSS across coils = 1 at every pixel
    rss = np.sqrt(np.sum(np.abs(coil_maps) ** 2, axis=0, keepdims=True))
    coil_maps = coil_maps / (rss + 1e-8)
    return coil_maps  # [num_coils, H, W] complex64


def single_to_multicoil(single_coil_kspace, sensitivity_maps):
    """
    Convert single-coil centered k-space to multi-coil k-space.

    Args:
        single_coil_kspace : [H, W] complex64  — centered k-space
        sensitivity_maps   : [C, H, W] complex64
    Returns:
        multicoil_kspace   : [C, H, W] complex64
    """
    # k-space → image domain
    image_sc = np.fft.fftshift(
        np.fft.ifft2(np.fft.ifftshift(single_coil_kspace))
    )

    # Multiply image by each coil sensitivity map
    multi_coil_images = image_sc[np.newaxis, :, :] * sensitivity_maps  # [C, H, W]

    # FFT each coil image back to k-space
    multicoil_kspace = np.fft.fftshift(
        np.fft.fft2(
            np.fft.ifftshift(multi_coil_images, axes=(-2, -1))
        ),
        axes=(-2, -1)
    )
    return multicoil_kspace.astype(np.complex64)  # [C, H, W]


# ─────────────────────────────────────────────────────────────────────────────
# Image → h5 conversion
# ─────────────────────────────────────────────────────────────────────────────

def image_to_h5(image_path, output_h5="temp.h5", num_coils=16):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # ── Resize keeping aspect ratio, then center-crop to 640x320 ─────────
    # instead of stretching which creates black borders
    target_h, target_w = 640, 320
    h, w = img.shape[:2]

    # Scale so the image fills 640x320 with no black bars
    scale = max(target_h / h, target_w / w)
    new_h, new_w = int(h * scale), int(w * scale)
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Center crop to exactly 640x320
    y0 = (new_h - target_h) // 2
    x0 = (new_w - target_w) // 2
    img = img[y0 : y0 + target_h, x0 : x0 + target_w]

    # Flip to match fastMRI orientation
    img = cv2.flip(img, 0)

    img = img.astype(np.float32) / 255.0
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    img = np.power(img, 0.7)                        # gamma boost

    # rest stays the same ...
    img_complex      = img.astype(np.complex64)
    single_coil_kspace = np.fft.fftshift(
        np.fft.fft2(np.fft.ifftshift(img_complex))
    ).astype(np.complex64)

    sensitivity_maps = simulate_coil_sensitivities(640, 320, num_coils=num_coils)
    multicoil_kspace = single_to_multicoil(single_coil_kspace, sensitivity_maps)

    current_max      = np.abs(multicoil_kspace).max()
    multicoil_kspace = multicoil_kspace * (0.004 / (current_max + 1e-8))
    print(f"[scale] before={current_max:.6f}  after={np.abs(multicoil_kspace).max():.6f}")

    k_space = multicoil_kspace[np.newaxis, :, :, :]

    mask_np = np.zeros(320, dtype=np.float32)
    mask_np[160 - 20 : 160 + 20] = 1.0

    with h5py.File(output_h5, "w") as f:
        f.create_dataset("kspace", data=k_space)
        f.create_dataset("mask",   data=mask_np)

    return output_h5

def verify_h5(path):
    """Sanity check — printed to console, not returned to client."""
    with h5py.File(path, "r") as f:
        k = f["kspace"][()]
        m = f["mask"][()]
    print(f"[verify] shape      : {k.shape}")          # (1, 16, 640, 320)
    print(f"[verify] dtype      : {k.dtype}")           # complex64
    print(f"[verify] kspace max : {np.abs(k).max():.6f}")  # ~0.001-0.006
    print(f"[verify] mask sum   : {m.sum()}")           # 40


# ─────────────────────────────────────────────────────────────────────────────
# Predict endpoint
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):

    filename  = file.filename.lower()
    temp_file = "temp.h5"

    # ── Case 1: Real .h5 file ─────────────────────────────────────────────
    if filename.endswith(".h5"):
        with open(temp_file, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

    # ── Case 2: MRI image → synthesize multi-coil h5 ─────────────────────
    elif filename.endswith((".png", ".jpg", ".jpeg")):
        temp_img = "temp_image.png"
        with open(temp_img, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        image_to_h5(temp_img, temp_file)
        verify_h5(temp_file)   # prints to console — remove after confirmed working

    else:
        return {"error": "Unsupported file type. Upload .h5 or .png/.jpg/.jpeg"}

    # ── Inference ─────────────────────────────────────────────────────────
    recon, zf, heatmap, psnr, ssim = run_inference(model, device, temp_file)

    # Replace the existing normalize function with this:
    def normalize(img):
        # Clip top 1% of intensities to reduce bright outlier dominance
        p1,  p99 = np.percentile(img, 1), np.percentile(img, 99)
        img = np.clip(img, p1, p99)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        return (img * 255).astype(np.uint8)

    recon_img     = cv2.cvtColor(normalize(recon),    cv2.COLOR_GRAY2BGR)
    zf_img        = cv2.cvtColor(normalize(zf),       cv2.COLOR_GRAY2BGR)
    heatmap_color = cv2.applyColorMap(normalize(heatmap), cv2.COLORMAP_JET)

    return {
        "reconstruction": encode_image(recon_img),
        "zero_filled":    encode_image(zf_img),
        "heatmap":        encode_image(heatmap_color),
        "psnr":           psnr,
        "ssim":           ssim
    }