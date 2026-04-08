import numpy as np
import torch
from torch.utils.data import Dataset

from src.fma.tokenizer import mel_tokenizer


class TokenMelDataset(Dataset):
    def __init__(
        self,
        spectrograms,
        labels=None,
        tokenize: bool = False,
        token_width: int = 45,
        stride: int = 15,
        max_tokens: int | None = 128,
        random_crop: bool = True,
    ) -> None:
        self.spectrograms = spectrograms
        self.labels = labels
        self.tokenize = tokenize

        self.token_width = token_width
        self.stride = stride
        self.max_tokens = max_tokens
        self.random_crop = random_crop

    @staticmethod
    def _to_tensor(x):
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).float()
        if isinstance(x, torch.Tensor):
            return x.float()
        raise TypeError(f"Expected np.ndarray or torch.Tensor, got {type(x)}")

    def __len__(self) -> int:
        return len(self.spectrograms)

    def __getitem__(self, index: int):
        mel = self._to_tensor(self.spectrograms[index])

        if self.tokenize:
            mel = mel_tokenizer(
                mel=mel,
                token_width=self.token_width,
                stride=self.stride,
                max_tokens=self.max_tokens,
                random_crop=self.random_crop
            )

        if self.labels is None:
            return mel

        label = self._to_tensor(self.labels[index])
        return mel, label