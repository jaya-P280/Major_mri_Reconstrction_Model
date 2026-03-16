import torch
from fastmri.models import VarNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():

    model = VarNet(
        num_cascades=12,
        sens_chans=8,
        sens_pools=4,
        chans=18,
        pools=4
    )

    checkpoint = torch.load(
        "trained_model.pt",
        map_location=device
    )

    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    # If GPU available → use half precision
    if device.type == "cuda":
        model = model.half()

    # Disable gradient tracking globally
    for param in model.parameters():
        param.requires_grad = False

    return model, device