import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_Bottleneck(nn.Module):
    def __init__(self, bottleneck_dim=64):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU()
        )

        self.pool = nn.AdaptiveAvgPool2d((8, 8))
        self.flatten = nn.Flatten()
        self.bottleneck = nn.Linear(128 * 8 * 8, bottleneck_dim)
        self.classifier = nn.Linear(bottleneck_dim, 2)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.flatten(x)
        z = self.bottleneck(x)
        out = self.classifier(F.relu(z))
        return out
