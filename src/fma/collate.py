import torch
from src.fma.tokenizer import build_mel_sequence
from typing import List, Tuple

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

    max_seq_len = max(s.shape[0] for s in token_sequences)
    n_mels = token_sequences[0].shape[1]

    padded_batch = []
    masks = []


    for s in token_sequences:
        seq_len = s.shape[0]
        pad_len  =  max_seq_len - seq_len

        if pad_len > 0: 
            pad = torch.zeros(
                pad_len,
                n_mels,
                token_width,
                dtype= s.dtype
            )

            s = torch.cat([s, pad], dim=0)

        mask = torch.zeros(max_seq_len, dtype=torch.bool)

        mask[:seq_len] = True

        masks.append(mask)

    padded_tokens = torch.stack(padded_batch, dim=0)
    attention_mask = torch.stack(masks, dim=0)