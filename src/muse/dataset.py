import pandas as pd
import torch

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

class MuSeDataset(Dataset):
    def __init__(
            self,
            df : pd.DataFrame,
            tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
            prompt_col: str = "prompt",
            label_col: str = "vad_01",
            max_len: int = 128) -> None:
        
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.prompt_col = prompt_col
        self.label_col = label_col
        self.max_len = max_len
        

    @staticmethod
    def load_dataframe(file_path: str) -> pd.DataFrame: pd.read_parquet(file_path)