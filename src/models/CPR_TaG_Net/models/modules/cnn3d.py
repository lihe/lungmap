import torch.nn as nn
import torch.nn.functional as F

class CNN3D(nn.Module):
    def __init__(self, in_channels=1, hidden_channels=64):
        super(CNN3D, self).__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),

            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),

            nn.Conv3d(64, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(hidden_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)