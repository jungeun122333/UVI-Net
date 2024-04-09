import torch.nn as nn
import torch
import numpy as np
from scipy.signal import gaussian
from scipy.ndimage import gaussian_filter


class FeatureExtract(nn.Module):
    def __init__(self):
        super(FeatureExtract, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv3d(
                in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(inplace=False),
            nn.Conv3d(
                in_channels=4, out_channels=4, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(inplace=False),
        )

        self.layer2 = nn.Sequential(
            nn.Conv3d(
                in_channels=4, out_channels=8, kernel_size=3, stride=2, padding=1
            ),
            nn.ReLU(inplace=False),
            nn.Conv3d(
                in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(inplace=False),
        )

        self.layer3 = nn.Sequential(
            nn.Conv3d(
                in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=1
            ),
            nn.ReLU(inplace=False),
            nn.Conv3d(
                in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        layer1 = self.layer1(x)
        layer2 = self.layer2(layer1)
        out = self.layer3(layer2)
        return [layer1, layer2, out]
