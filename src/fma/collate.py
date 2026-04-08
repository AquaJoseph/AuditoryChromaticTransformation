import torch
from typing import List

def collate_token_seqs(batch, pad_value: float = -80.0):
    has_labels = isinstance(batch[0], tuple)

    if has_labels:
        sequences, labels = zip(*batch)
    else:
        sequences = batch
        labels = None

    max_len = max(seq.shape[0] for seq in sequences)

    padded = []
    masks = []

    for seq in sequences:
        seq_len, n_mels, token_width = seq.shape
        pad_len = max_len - seq_len

        if pad_len > 0:
            pad = torch.full(
                (pad_len, n_mels, token_width),
                pad_value,
                dtype=seq.dtype
            )
            seq = torch.cat([seq, pad], dim=0)

        mask = torch.zeros(max_len, dtype=torch.bool)
        mask[:seq_len] = True

        padded.append(seq)
        masks.append(mask)

    padded_tokens = torch.stack(padded, dim=0)
    attention_mask = torch.stack(masks, dim=0)

    if labels is None:
        return padded_tokens, attention_mask

    labels = torch.stack(labels, dim=0)
    return padded_tokens, attention_mask, labels






