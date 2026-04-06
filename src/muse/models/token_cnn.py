import torch
import torch.nn as nn


class TokenCNN(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.fc = nn.Linear(64, embedding_dim)

    def forward(self, x):
        # x: [B*N, 1, 128, 45]
        x = self.net(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)  # [B*N, D]