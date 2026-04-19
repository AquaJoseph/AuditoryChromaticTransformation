## Make Doc Strings Later on

import pandas as pd
import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast


class MuSeDataset(Dataset):
    def __init__(
            self,
            df : pd.DataFrame,
            tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
            prompt_col: str = "prompt",
            label_col: str = "vad",
            max_len: int = 128) -> None:
        
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer

        self.prompt_col = prompt_col
        self.label_col = label_col

        if self.prompt_col not in self.df.columns: raise ValueError(f"Missing prompt column: {self.prompt_col}")
        
        if self.label_col not in self.df.columns: raise ValueError(f"Missing label column: {self.label_col}")

        self.max_len = max_len

    def __len__(self) -> int : return len(self.df)

    def __getitem__(self, index : int) -> dict[str, torch.Tensor]:
        prompt = self.df.loc[index, self.prompt_col]
        label = self.df.loc[index, self.label_col]

        if not isinstance(label, (list, tuple, np.ndarray)): raise TypeError(f"Expected label to be list, tuple or nd array, got {type(label)}")
        if len(label) != 3: raise ValueError(f"Expected label of length 3, got length of {len(label)}")

        encoding = self.tokenizer(
            prompt,
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        item = {key : value.squeeze(0) for key, value in encoding.items()}
        item["labels"] = torch.tensor(label, dtype=torch.float32)

        return item

def load_dataframe(file_path: str) -> pd.DataFrame: return pd.read_parquet(path=file_path)

def build_dataset(
        df : pd.DataFrame, 
        tokenizer : PreTrainedTokenizer | PreTrainedTokenizerFast,
        prompt_col: str = "prompt",
        label_col: str = "vad",
        max_len: int = 128) -> MuSeDataset: 
     return MuSeDataset(
            df=df, 
            tokenizer=tokenizer,
            prompt_col=prompt_col,
            label_col=label_col,
            max_len=max_len)

def build_loader(
        dataset : MuSeDataset, 
        batch_size : int, 
        shuffle : bool) -> DataLoader : 
     return DataLoader(
            dataset=dataset, 
            batch_size=batch_size, 
            shuffle=shuffle)
