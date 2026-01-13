import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_Bottleneck(nn.Module):
    def __init__(self, bottleneck_dim=64):
        super().__init__()

        # Extraction des features
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),  # 224 -> 112
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), # 112 -> 56
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),# 56 -> 28
            nn.ReLU()
        )

        # Pooling adaptatif pour avoir taille fixe
        self.pool = nn.AdaptiveAvgPool2d((8, 8))
        self.flatten = nn.Flatten()

        # Bottleneck layer (compression)
        self.bottleneck = nn.Linear(128 * 8 * 8, bottleneck_dim)

        # Classification
        self.classifier = nn.Linear(bottleneck_dim, 2)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.flatten(x)

        z = self.bottleneck(x)
        z_relu = F.relu(z)  # activé pour classifier

        out = self.classifier(z_relu)

        # Si tu veux visualiser les embeddings, tu peux faire :
        # return out, z
        return out
