import torch
import h5py
import fastmri

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

    zf = fastmri.ifft2c(kspace)
    zf = fastmri.complex_abs(zf)
    zf = zf.sum(dim=1).squeeze().cpu()

    change_map = torch.abs(recon - zf)
    change_map = change_map / change_map.max()

    return recon, zf, change_map