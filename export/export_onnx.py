import torch
from models.vision_mamba import VisionMambaDenoiser

device="cuda" if torch.cuda.is_available() else "cpu"

model=VisionMambaDenoiser().to(device)

model.load_state_dict(torch.load("mamba_denoiser.pth"))

model.eval()

dummy=torch.randn(1,3,256,256).to(device)

torch.onnx.export(
    model,
    dummy,
    "mamba_denoiser.onnx",
    opset_version=17
)

print("ONNX export complete")