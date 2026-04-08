import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

from src.fma.tokenizer import mel_tokenizer


class TokenMelDataset(Dataset):
    def __init__(
        self,
        data: list[torch.Tensor | np.ndarray],
        tokenize: bool = False,
        token_width: int = 45,
        stride: int = 15,
        max_tokens: int | None = 128,
        random_crop: bool = True,
    ) -> None:
        """
        Args:
            data:
                Either:
                - list of token sequences [T, 128, 45]
                - OR list of Mel spectrograms [128, T]

            tokenize:
                If True → apply mel_tokenizer to each item
                If False → assume already tokenized

            tokenizer params:
                Used only if tokenize=True
        """
        self.data = data
        self.tokenize = tokenize

        self.token_width = token_width
        self.stride = stride
        self.max_tokens = max_tokens
        self.random_crop = random_crop

    def __len__(self) -> int:
        return len(self.data)

    def _to_tensor(self, x: torch.Tensor | np.ndarray) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).float()
        if isinstance(x, torch.Tensor):
            return x.float()
        raise TypeError(f"Expected np.ndarray or torch.Tensor, got {type(x)}")

    def __getitem__(self, index: int) -> torch.Tensor:
        x = self.data[index]
        x = self._to_tensor(x)

        if self.tokenize:
            x = mel_tokenizer(
                mel=x,
                token_width=self.token_width,
                stride=self.stride,
                max_tokens=self.max_tokens,
                random_crop=self.random_crop
            )

        return x

  