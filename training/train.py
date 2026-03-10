import torch
from models.vision_mamba import VisionMambaDenoiser

device = "cuda" if torch.cuda.is_available() else "cpu"

model = VisionMambaDenoiser().to(device)

optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)

loss_fn = torch.nn.MSELoss()

for step in range(500):

    clean = torch.rand(1,3,256,256).to(device)

    noisy = clean + torch.randn_like(clean)*0.05

    output = model(noisy)

    loss = loss_fn(output,clean)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    print("step:",step,"loss:",loss.item())

torch.save(model.state_dict(),"mamba_denoiser.pth")