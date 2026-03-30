import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(
            self, 
            d_model : 
            int, max_len: int = 5000) -> None:
        super.__init__()

        position = torch.arange(max_len).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        # Positional Encoding
        pe = torch.zero(max_len, d_model)

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self , x : torch.Tensor) -> torch.Tensor:

        seq_len = x.size(1)

        return x + self.pe[:, :seq_len]

