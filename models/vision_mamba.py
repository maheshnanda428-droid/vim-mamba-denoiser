import torch
import torch.nn as nn
from mamba_ssm import Mamba

class VisionMambaDenoiser(nn.Module):

    def __init__(self, d_model=192, patch_size=16, channels=3):
        super().__init__()

        self.patch_embed = nn.Conv2d(
            channels,
            d_model,
            kernel_size=patch_size,
            stride=patch_size
        )

        self.mamba = Mamba(
            d_model=d_model,
            d_state=16,
            d_conv=4,
            expand=2
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                d_model,
                64,
                kernel_size=patch_size,
                stride=patch_size
            ),
            nn.ReLU(),
            nn.Conv2d(64, channels, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):

        B,C,H,W = x.shape

        x = self.patch_embed(x)

        h,w = x.shape[2],x.shape[3]

        x = x.flatten(2).transpose(1,2)

        x = self.mamba(x)

        x = x.transpose(1,2).view(B,-1,h,w)

        x = self.decoder(x)

        return x