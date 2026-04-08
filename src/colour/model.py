import torch
import torch.nn as nn


class VADToColor(nn.Module):
    """
    Maps VAD (valence, arousal, dominance) → RGB colour.
    Input:  [B, 3]
    Output: [B, 3] in range [0, 1]
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        depth: int = 2,
        dropout: float = 0.0
    ) -> None:
        super().__init__()

        layers = []
        in_dim = 3

        for _ in range(depth):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            in_dim = hidden_dim

        layers.append(nn.Linear(hidden_dim, 3))
        layers.append(nn.Sigmoid())  # constrain to [0,1]

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    