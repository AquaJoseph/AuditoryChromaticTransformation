import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast



class FMAInferenceDataset(Dataset):
    def __init__(
            self,
            df : pd.DataFrame,
            tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
            prompt_col: str = "prompt",
            max_len: int = 128) -> None:
        
        self.df = df
        self.tokenizer = tokenizer
        self.prompt_col = prompt_col
        self.max_len = max_len

        if self.prompt_col not in self.df.columns: raise ValueError(f"Missing prompt column: {self.prompt_col}")

    def __len__(self): return len(self.df)

    
    def __getitem__(self, index : int) -> dict[str, torch.Tensor]:
        prompt = self.df.loc[index, self.prompt_col]

        encoding = self.tokenizer(
            prompt,
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        item = {key : value.squeeze(0) for key, value in encoding.items()}

        return item
    

def build_dataset(
    df: pd.DataFrame,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast) -> FMAInferenceDataset: 
    return FMAInferenceDataset(df=df, tokenizer=tokenizer)

def build_loader(
        dataset: FMAInferenceDataset,
        batch_size: int,
        shuffle: bool = False) -> DataLoader:
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )
