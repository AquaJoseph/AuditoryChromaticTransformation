import torch
import torch.nn.functional as F

from typing import List, Tuple

def build_mel_sequence(
        mel: torch.Tensor,
        token_width : int = 45,
        stride : int = 15
) -> torch.Tensor:
    """
    Doc String to do
    """

    if mel.ndim != 2: raise ValueError(f"Expected Mel shape [128, T], got {mel.shape}")

    _, time_steps = mel.shape

    tokens = []


    for start in range(0, time_steps, stride):
        end = start + token_width
        chunk = mel[:, start:end]

        if chunk.shape[1] == 0:
            break

        if chunk.shape[1] < token_width:
            pad = token_width - chunk.shape[1]
            chunk = F.pad(chunk, (0, pad))

        tokens.append(chunk)

    return torch.stack(tokens, dim=0)


def collate_mel_sequence(
        batch : List[Tuple[torch.Tensor, torch.Tensor]],
        token_width : int = 45,
        stride : int = 15
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Do Doc String Later 
    """

    token_sequences = []
    labels = []

    for x, t in batch:
        tokens = build_mel_sequence(
            x,
            token_width=token_width,
            stride=stride
        )
        token_sequences.append(tokens)
        labels.append(t)

    
