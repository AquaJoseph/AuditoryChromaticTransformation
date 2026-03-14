import torch
import torch.nn as nn
import configparser
import pandas as pd
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from torch.optim import AdamW
from tqdm import tqdm

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
    temp_df,
    test_size=0.5,
    random_state=42
)

tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])
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

model = BertVADRegressor(model_name=config["model"]["name"]).to(device=device)

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
        device: torch.device) -> float:
    
    model.train()

    total_loss = 0.0

    progress_bar = tqdm(dataloader, desc="Training", leave=False)


    for batch in progress_bar:
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
        device: torch.device) -> float:
    
    model.eval()
    total_loss = 0.0

    progress_bar = tqdm(dataloader, desc="Validation", leave=False)


    with torch.no_grad():
        for batch in progress_bar:
            loss = forward_batch(
                model=model,
                batch=batch,
                criterion=criterion,
                device=device
            )

            total_loss += loss.item()
    
    return total_loss / len(dataloader)


epochs = config.getint("train", "epochs")

best_val_loss = float("-inf")

train_losses = []
val_losses = []

for epoch in range(epochs):
    train_loss, val_loss = train_epoch(
        model=model,
        dataloader=dataloaders["train"],
        criterion=criterion,
        optimiser=optimiser,
        device=device
    ), validate_epoch(
        model=model,
        dataloader=dataloaders["val"],
        criterion=criterion,
        device=device
    )

    print(
        f"Epoch {epoch + 1}/{epochs} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f}"
    )

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "checkpoints\\muse\\bert-large-uncased-saves\\best_bert_vad.pt")


plt.figure(figsize=(10, 6))
plt.plot(range(1, epochs + 1), train_losses, label="Training Loss")
plt.plot(range(1, epochs + 1), val_losses, label="Validation Loss")

plt.title("Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()

plt.savefig("plots/muse/train/training_validation_loss.png", dpi=300, bbox_inches="tight")
plt.savefig("plots/muse/train/bert_vad_loss_curve.png", dpi=300, bbox_inches="tight")