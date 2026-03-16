import torch
import h5py
import fastmri
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def run_inference(model, device, file_path):

    with h5py.File(file_path, "r") as f:
        kspace_np = f["kspace"][0]
        mask_np = f["mask"][:]

    kspace = torch.from_numpy(kspace_np)
    kspace = torch.view_as_real(kspace)
    kspace = kspace.unsqueeze(0).to(device)

    mask = torch.from_numpy(mask_np).bool()
    mask = mask.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(-1)
    mask = mask.to(device)

    with torch.inference_mode():
        recon = model(kspace, mask)

    recon = recon.squeeze().cpu()

    # Zero-filled reconstruction
    zf = fastmri.ifft2c(kspace)
    zf = fastmri.complex_abs(zf)
    zf = zf.sum(dim=1).squeeze().cpu()

    recon_np = recon.numpy()
    zf_np = zf.numpy()

    
    recon_norm = (recon_np - recon_np.min()) / (recon_np.max() - recon_np.min() + 1e-8)
    zf_norm = (zf_np - zf_np.min()) / (zf_np.max() - zf_np.min() + 1e-8)

    
    psnr = float(peak_signal_noise_ratio(zf_norm, recon_norm, data_range=1.0))
    ssim = float(structural_similarity(zf_norm, recon_norm, data_range=1.0))


    change_map = np.abs(recon_norm - zf_norm)
    change_map = change_map / (change_map.max() + 1e-8)

    return recon_np, zf_np, change_map, psnr, ssim