import torch
from typing import List

def collate_token_sequence(batch: List[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Do Doc String
    """

    max_len = max(seq.shape[0] for seq in batch)

    padded, masks = [], []

    for seq in batch:
        token, bins, width = seq.shape
        pad_len = max_len - token

        if pad_len > 0:
            pad = torch.full((pad_len, bins, width), -80.0 , dtype=seq.dtype)
            seq = torch.cat([seq,pad], dim=0)
        
        mask = torch.cat([
            torch.ones(token, dtype=torch.bool),
            torch.zeros(pad_len, dtype=torch.bool)
        ])

        padded.append(seq)
        masks.append(mask)
    
    padded_tokens = torch.stack(padded, dim=0)
    attention_mask = torch.stack(masks, dim=0)

    return padded_tokens, attention_mask






