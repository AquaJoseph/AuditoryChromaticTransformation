"""Training entrypoint and loop utilities for MuSe VAD regression."""

import configparser
import json
import warnings
from datetime import datetime, timezone
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers.utils import logging as hf_logging

from .dataset import build_dataset, build_loader, load_dataframe
from .model import BertVADRegressor

warnings.filterwarnings("ignore")
hf_logging.set_verbosity_error()


def forward_batch(
    model: BertVADRegressor,
    batch: dict[str, torch.Tensor],
    criterion: nn.Module,
    device: torch.device,
) -> torch.Tensor:
    """Template docstring.

    Args:
        model: TODO.
        batch: TODO.
        criterion: TODO.
        device: TODO.

    Returns:
        torch.Tensor: TODO.
    """
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    target = batch["labels"].to(device)

    prediction = model(input_ids=input_ids, attention_mask=attention_mask)
    return criterion(prediction, target)


def train_epoch(
    model: BertVADRegressor,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: AdamW,
    device: torch.device,
) -> float:
    """Template docstring.

    Args:
        model: TODO.
        dataloader: TODO.
        criterion: TODO.
        optimizer: TODO.
        device: TODO.

    Returns:
        float: TODO.
    """
    model.train()
    total_loss = 0.0

    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    for batch in progress_bar:
        optimizer.zero_grad()
        loss = forward_batch(
            model=model,
            batch=batch,
            criterion=criterion,
            device=device,
        )
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


def validate_epoch(
    model: BertVADRegressor,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Template docstring.

    Args:
        model: TODO.
        dataloader: TODO.
        criterion: TODO.
        device: TODO.

    Returns:
        float: TODO.
    """
    model.eval()
    total_loss = 0.0

    progress_bar = tqdm(dataloader, desc="Validation", leave=False)
    with torch.no_grad():
        for batch in progress_bar:
            loss = forward_batch(
                model=model,
                batch=batch,
                criterion=criterion,
                device=device,
            )
            total_loss += loss.item()

    return total_loss / len(dataloader)

    return total_loss / len(dataloader)


def load_config(config_path: Path) -> configparser.ConfigParser:
    """Template docstring.

    Args:
        config_path: TODO.

    Returns:
        configparser.ConfigParser: TODO.
    """
    config = configparser.ConfigParser()
    config.read(config_path)
    return config


def main() -> None:
    """Template docstring.

    Summary:
        TODO: Describe end-to-end training workflow and side effects.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = load_config(Path("src/muse/config.cfg"))

    prompt_col = config.get("data", "prompt_col", fallback="prompt")
    label_col = config.get("data", "label_col", fallback="vad")
    max_len = config.getint("data", "max_len", fallback=128)

    batch_size = config.getint("train", "batch_size")
    learning_rate = config.getfloat("train", "learning_rate")
    epochs = config.getint("train", "epochs")
    unfreeze_epoch = config.getint("train", "unfreeze_epoch", fallback=-1)

    test_split = config.getfloat("data", "test_split")

    df = load_dataframe("datasets/muse/modified/bert_df.parquet")

    train_df, temp_df = train_test_split(df, test_size=test_split, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])

    datasets = {
        "train": build_dataset(
            train_df,
            tokenizer,
            prompt_col=prompt_col,
            label_col=label_col,
            max_len=max_len,
        ),
        "val": build_dataset(
            val_df,
            tokenizer,
            prompt_col=prompt_col,
            label_col=label_col,
            max_len=max_len,
        ),
        "test": build_dataset(
            test_df,
            tokenizer,
            prompt_col=prompt_col,
            label_col=label_col,
            max_len=max_len,
        ),
    }

    dataloaders = {
        "train": build_loader(datasets["train"], batch_size, True),
        "val": build_loader(datasets["val"], batch_size, False),
        "test": build_loader(datasets["test"], batch_size, False),
    }

    model = BertVADRegressor(model_name=config["model"]["name"]).to(device=device)

    for param in model.bert.parameters():
        param.requires_grad = False

    criterion = nn.MSELoss()

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
    )

    best_val_loss = float("inf")
    train_losses: list[float] = []
    val_losses: list[float] = []

    checkpoint_dir = Path("checkpoints/muse/bert-large-uncased-saves")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        if epoch == unfreeze_epoch:
            for param in model.bert.parameters():
                param.requires_grad = True

            optimizer = AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=learning_rate,
            )

        train_loss = train_epoch(
            model=model,
            dataloader=dataloaders["train"],
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        val_loss = validate_epoch(
            model=model,
            dataloader=dataloaders["val"],
            criterion=criterion,
            device=device,
        )

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f}"
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)

def load_config(config_path: Path) -> configparser.ConfigParser:
    """Template docstring.

    Args:
        config_path: TODO.

    Returns:
        configparser.ConfigParser: TODO.
    """
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_dir / "best_bert_vad.pt")

    torch.save(model.state_dict(), checkpoint_dir / "final_bert_vad.pt")

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), train_losses, label="Training Loss")
    plt.plot(range(1, epochs + 1), val_losses, label="Validation Loss")

    plt.title("Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()


    return total_loss / len(dataloader)


def load_config(config_path: Path) -> configparser.ConfigParser:
    config = configparser.ConfigParser()
    config.read(config_path)
    return config


def main() -> None:
    """Template docstring.

    Summary:
        TODO: Describe end-to-end training workflow and side effects.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = load_config(Path("src/muse/config.cfg"))

    prompt_col = config.get("data", "prompt_col", fallback="prompt")
    label_col = config.get("data", "label_col", fallback="vad")
    max_len = config.getint("data", "max_len", fallback=128)

    batch_size = config.getint("train", "batch_size")
    learning_rate = config.getfloat("train", "learning_rate")
    epochs = config.getint("train", "epochs")
    unfreeze_epoch = config.getint("train", "unfreeze_epoch", fallback=-1)

    test_split = config.getfloat("data", "test_split")

    df = load_dataframe("datasets/muse/modified/bert_df.parquet")

    train_df, temp_df = train_test_split(df, test_size=test_split, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])

    datasets = {
        "train": build_dataset(
            train_df,
            tokenizer,
            prompt_col=prompt_col,
            label_col=label_col,
            max_len=max_len,
        ),
        "val": build_dataset(
            val_df,
            tokenizer,
            prompt_col=prompt_col,
            label_col=label_col,
            max_len=max_len,
        ),
        "test": build_dataset(
            test_df,
            tokenizer,
            prompt_col=prompt_col,
            label_col=label_col,
            max_len=max_len,
        ),
    }

    dataloaders = {
        "train": build_loader(datasets["train"], batch_size, True),
        "val": build_loader(datasets["val"], batch_size, False),
        "test": build_loader(datasets["test"], batch_size, False),
    }

    model = BertVADRegressor(model_name=config["model"]["name"]).to(device=device)

    for param in model.bert.parameters():
        param.requires_grad = False

    criterion = nn.MSELoss()

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
    )

    best_val_loss = float("inf")
    train_losses: list[float] = []
    val_losses: list[float] = []
    test_losses: list[float] = []

    run_tag = (
        f"epochs_{epochs}_batch_{batch_size}_"
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )

    checkpoint_dir = Path("checkpoints/muse/bert-large-uncased-saves")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)


    test_split = config.getfloat("data", "test_split")

    df = load_dataframe("datasets/muse/modified/bert_df.parquet")

    train_df, temp_df = train_test_split(df, test_size=test_split, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])

    datasets = {
        "train": build_dataset(
            train_df,
            tokenizer,
            prompt_col=prompt_col,
            label_col=label_col,
            max_len=max_len,
        ),
        "val": build_dataset(
            val_df,
            tokenizer,
            prompt_col=prompt_col,
            label_col=label_col,
            max_len=max_len,
        ),
        "test": build_dataset(
            test_df,
            tokenizer,
            prompt_col=prompt_col,
            label_col=label_col,
            max_len=max_len,
        ),
    }

    dataloaders = {
        "train": build_loader(datasets["train"], batch_size, True),
        "val": build_loader(datasets["val"], batch_size, False),
        "test": build_loader(datasets["test"], batch_size, False),
    }

    model = BertVADRegressor(model_name=config["model"]["name"]).to(device=device)

    for param in model.bert.parameters():
        param.requires_grad = False

    criterion = nn.MSELoss()

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
    )

    best_val_loss = float("inf")
    train_losses: list[float] = []
    val_losses: list[float] = []

    checkpoint_dir = Path("checkpoints/muse/bert-large-uncased-saves")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        if epoch == unfreeze_epoch:
            for param in model.bert.parameters():
                param.requires_grad = True

            optimizer = AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=learning_rate,
            )

        train_loss = train_epoch(
            model=model,
            dataloader=dataloaders["train"],
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        val_loss = validate_epoch(
            model=model,
            dataloader=dataloaders["val"],
            criterion=criterion,
            device=device,
        )
        test_loss = validate_epoch(
            model=model,
            dataloader=dataloaders["test"],
            criterion=criterion,
            device=device,
        )

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Test Loss: {test_loss:.4f}"
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        test_losses.append(test_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_dir / "best_bert_vad.pt")

    torch.save(model.state_dict(), checkpoint_dir / "final_bert_vad.pt")

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), train_losses, label="Training Loss")
    plt.plot(range(1, epochs + 1), val_losses, label="Validation Loss")
    plt.plot(range(1, epochs + 1), test_losses, label="Test Loss")

    plt.title("Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()

    plot_dir = Path("plots/muse/train")
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_name = f"loss_curves_{run_tag}.png"
    plt.savefig(plot_dir / plot_name, dpi=300, bbox_inches="tight")

    metrics_dir = Path("metrics/muse/train")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / f"losses_{run_tag}.json"
    metrics = {
        "run_tag": run_tag,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "best_val_loss": best_val_loss,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "test_losses": test_losses,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")



            optimizer = AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=learning_rate,
            )

        train_loss = train_epoch(
            model=model,
            dataloader=dataloaders["train"],
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        val_loss = validate_epoch(
            model=model,
            dataloader=dataloaders["val"],
            criterion=criterion,
            device=device,
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
            torch.save(model.state_dict(), checkpoint_dir / "best_bert_vad.pt")

    torch.save(model.state_dict(), checkpoint_dir / "final_bert_vad.pt")

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), train_losses, label="Training Loss")
    plt.plot(range(1, epochs + 1), val_losses, label="Validation Loss")

    plt.title("Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()

    plot_dir = Path("plots/muse/train")
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_name = (
        f"training_validation_loss_over_{epochs}_epochs"
        f"_with_lr_{learning_rate}_and_batch_size_{batch_size}.png"
    )
    plt.savefig(plot_dir / plot_name, dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
