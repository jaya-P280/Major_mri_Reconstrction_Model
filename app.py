from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import shutil
import numpy as np
import cv2
from model import load_model
from utils import run_inference

app = FastAPI()
model, device = load_model()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):

    temp_file = "temp.h5"

    # Save uploaded file
    with open(temp_file, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Run model inference
    recon, zf, heatmap, psnr, ssim = run_inference(model, device, temp_file)

    # Normalize images
    def normalize(img):
        img = img - img.min()
        img = img / (img.max() + 1e-8)
        return (img * 255).astype(np.uint8)

    recon_img = normalize(recon)
    zf_img = normalize(zf)
    heatmap_img = normalize(heatmap)

    # Apply heatmap color
    heatmap_color = cv2.applyColorMap(heatmap_img, cv2.COLORMAP_JET)

    # Convert grayscale to 3 channel
    recon_img = cv2.cvtColor(recon_img, cv2.COLOR_GRAY2BGR)
    zf_img = cv2.cvtColor(zf_img, cv2.COLOR_GRAY2BGR)

    # Combine images
    combined = np.hstack((zf_img, recon_img, heatmap_color))

    output_path = "output.png"
    cv2.putText(combined, f"PSNR: {psnr:.2f}", (10,30),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    cv2.putText(combined, f"SSIM: {ssim:.3f}", (10,60),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    cv2.imwrite(output_path, combined)
    cv2.putText(zf_img, "Zero Filled", (10,20),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    cv2.putText(recon_img, "Reconstructed", (10,20),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    cv2.putText(heatmap_color, "Heatmap", (10,20),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    
    return FileResponse("output.png", media_type="image/png")