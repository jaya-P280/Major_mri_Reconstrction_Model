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

    with open(temp_file, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    recon, zf, change_map = run_inference(model, device, temp_file)

    # Convert tensors to numpy
    recon = recon.numpy()
    zf = zf.numpy()
    change_map = change_map.numpy()

    # Normalize images to 0-255
    def normalize(img):
        img = img - img.min()
        img = img / (img.max() + 1e-8)
        return (img * 255).astype(np.uint8)

    recon_img = normalize(recon)
    zf_img = normalize(zf)
    heatmap_img = normalize(change_map)

    # Apply heatmap color
    heatmap_color = cv2.applyColorMap(heatmap_img, cv2.COLORMAP_JET)

    # Convert grayscale to 3 channel
    recon_img = cv2.cvtColor(recon_img, cv2.COLOR_GRAY2BGR)
    zf_img = cv2.cvtColor(zf_img, cv2.COLOR_GRAY2BGR)

    # Stack horizontally
    combined = np.hstack((zf_img, recon_img, heatmap_color))

    output_path = "output.png"
    cv2.imwrite(output_path, combined)

    return FileResponse(output_path, media_type="image/png")