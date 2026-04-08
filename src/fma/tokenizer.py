import torch
import torch.nn.functional as F
import random as R

from typing import List, Tuple

def mel_tokenizer(
        mel: torch.Tensor,
        token_width : int = 45,
        stride : int = 15,
        max_tokens : int | None = 128,
        random_crop : bool = True
) -> torch.Tensor:
    """
    Doc String to do
    """

    if mel.ndim != 2: raise ValueError(f"Expected Mel shape [128, T], got {mel.shape}")

    _, time_steps = mel.shape

    tokens = []

    if time_steps < token_width: 
        pad = token_width - time_steps
        chunk = F.pad (mel, (0,pad))
        return chunk.unsqueeze(0)


    for start in range(0, time_steps, stride):
        end = start + token_width
        chunk = mel[:, start:end]

        if chunk.shape[1] == 0:
            break

        if chunk.shape[1] < token_width:
            pad = token_width - chunk.shape[1]
            chunk = F.pad(chunk, (0, pad))

        tokens.append(chunk)

    tokens = torch.stack(tokens, dim=0)

    if max_tokens != None and tokens.shape[0] > max_tokens:
        if random_crop:
            start = R.randint(0, tokens.shape[0] - max_tokens)
            tokens = tokens[start:start + max_tokens]
        else:
            tokens = tokens[:max_tokens]

    return tokens