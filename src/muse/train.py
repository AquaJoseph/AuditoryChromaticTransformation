import torch
import torch.nn as nn
import configparser
import pandas as pd

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from torch.optim import AdamW

from dataset import load_dataframe, build_dataset, build_loader
from model import BertVADRegressor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = configparser.ConfigParser()
config.read("src\\muse\\config.cfg")

df = load_dataframe("datasets/muse/modified/bert_df.parquet")

# Split
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

tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")
# Datasets
datasets = {
    "train" : build_dataset(train_df, tokenizer),
    "val": build_dataset(val_df, tokenizer),
    "test" : build_dataset(test_df, tokenizer)
}

# Loaders
dataloaders = {
    "train" : build_loader(datasets["train"], config.getint("train", "batch_size"), True),
    "val": build_loader(datasets["val"], config.getint("train", "batch_size"), False),
    "test" : build_loader(datasets["test"], config.getint("train", "batch_size"), False)
}

# Model Loading

model = BertVADRegressor().to(device=device)

for param in model.bert.parameters():
    param.requires_grad = False

criterion = nn.MSELoss()

optimiser = AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr= config.getfloat("train", "learning_rate")
)


def forward_batch(
        model: BertVADRegressor,
        batch: dict[str, torch.Tensor],
        criterion: nn.Module,
        device: torch.device) -> torch.Tensor:
    in_ids = batch["input_ids"].to(device)
    atten_mask = batch["attention_mask"].to(device)
    target = batch["labels"].to(device)

    pred = model(
    input_ids=in_ids,
    attention_mask=atten_mask
    )

    return criterion(pred, target)


def train_epoch(
        model : BertVADRegressor,
        dataloader : DataLoader,
        criterion : nn.Module,
        optimiser : AdamW,
        device: str) -> float:
    
    model.train()

    total_loss = 0.0

    for batch in dataloader:
        optimiser.zero_grad()

        loss = forward_batch(
            model=model,
            batch=batch,
            criterion=criterion,
            device=device
        )

        loss.backward()
        optimiser.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def validate_epoch(
        model : BertVADRegressor,
        dataloader : DataLoader,
        criterion : nn.Module,
        device: str) -> float:
    
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            loss = forward_batch(
            model=model,
            batch=batch,
            criterion=criterion,
            device=device
        )
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)