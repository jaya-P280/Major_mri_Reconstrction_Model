import torch
import h5py
import fastmri
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def run_inference(model, device, file_path):
    with h5py.File(file_path, "r") as f:
        kspace_np = f["kspace"][0]       # [16, 640, 320] complex64
        mask_np   = f["mask"][:]         # [320] float32

    kspace = torch.from_numpy(kspace_np)
    kspace = torch.view_as_real(kspace)              # [16, 640, 320, 2]
    kspace = kspace.unsqueeze(0).to(device)          # [1, 16, 640, 320, 2]

    mask = torch.from_numpy(mask_np)
    mask = mask.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(-1)  # [1,1,1,320,1]
    mask = mask.bool().to(device)                    # ← bool for torch.where()

    with torch.inference_mode():
        recon = model(kspace, mask)                  # [1, 640, 320]

    recon  = recon.squeeze().cpu().float().numpy()   # [640, 320]

    # Zero-filled: IFFT → RSS across coils
    zf     = fastmri.ifft2c(kspace.cpu().float())    # [1, 16, 640, 320, 2]
    zf_abs = fastmri.complex_abs(zf)                 # [1, 16, 640, 320]
    zf_rss = fastmri.rss(zf_abs, dim=1)              # [1, 640, 320]
    zf_np  = zf_rss.squeeze().numpy()                # [640, 320]

    # Normalize to [0,1]
    def norm01(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-8)

    recon_norm = norm01(recon)
    zf_norm    = norm01(zf_np)

    # Metrics
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    psnr = float(peak_signal_noise_ratio(recon_norm, zf_norm, data_range=1.0))
    ssim = float(structural_similarity(recon_norm,   zf_norm, data_range=1.0))

    # Heatmap
    heatmap = np.abs(recon_norm - zf_norm)
    heatmap = heatmap / (heatmap.max() + 1e-8)

    return recon_norm, zf_norm, heatmap, psnr, ssim