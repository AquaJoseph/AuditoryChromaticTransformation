import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(
            self, 
            d_model : 
            int, max_len: int = 5000) -> None:
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1).float()

        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        # Positional Encoding
        pe = torch.zeros(max_len, d_model)

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self , x : torch.Tensor) -> torch.Tensor:

        seq_len = x.size(1)

        return x + self.pe[:, :seq_len]
    

class TokenEncoder(nn.Module):
    def __init__(self, d_model : int = 128, channels=None) -> None:
        super().__init__()

        if channels is None:
            channels = [1,16,32,64]
        
        self.conv_net = self._build_stack = (channels)

        dummy = torch.zeros(1, 1, 128, 45)

        with torch.no_grad():
            out = self.conv(dummy)
            self.flatten_dim = out.view(1, -1).shape[1]

        self.projection = nn.Linear(self.flatten_dim, d_model) 

    @staticmethod
    def _conv_block(in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
    

    def _build_stack(self, channels):
        layers = [
            self._conv_block(channels[i], channels[i + 1]) for i in range(len(channels) - 1)
        ]
        return nn.Sequential(*layers)
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, F, W = x.shape
        x = x.view(B * S, 1, F, W)
        x = self.conv(x)
        x = x.flatten(start_dim=1)
        x = self.projection(x)
        x = x.view(B, S, -1)

        return x


