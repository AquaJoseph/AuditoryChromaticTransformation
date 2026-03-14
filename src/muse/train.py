import torch
import torch.nn as nn
import configparser
import pandas as pd

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from torch.optim import AdamW

from dataset import load_dataframe, MuSeDataset
from model import BertVADRegressor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = configparser.ConfigParser()
config.read("config.cfg")

df = load_dataframe("datasets/muse/modified/bert_df.parquet")

train_df, temp_df = train_test_split(
    df,
    test_size=config.getfloat("data", "test_split"),
    random_state=42
)

val_df, test_df = train_test_split(
    df,
    test_size=0.5,
    random_state=42
)