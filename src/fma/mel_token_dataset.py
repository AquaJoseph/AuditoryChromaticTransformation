import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class TokenMelDataset(Dataset):
    def __init__(self, token_seqs: list[torch.Tensor | np.ndarray]):
        self.token_seqs = [self._to_tensor(seq) for seq in token_seqs]

    @staticmethod
    def _to_tensor(seq: torch.Tensor | np.ndarray) -> torch.Tensor:
        if isinstance(seq, np.ndarray):
            return torch.from_numpy(seq).float()
        if isinstance(seq, torch.Tensor):
            return seq.float()
        raise TypeError(f"Expected np.ndarray or torch.Tensor, got {type(seq)}")

    def __len__(self): len(self.token_seqs)

    def __getitem__(self, index): return self.token_seqs[index]

  