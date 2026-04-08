import configparser
import pandas as pd
import torch
import numpy as np
import torch.nn as nn
import random as R
import json


from torch.utils.data import DataLoader, TensorDataset, random_split
from src.colour.model import VADToColor
from pathlib import Path
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = ROOT / "src" / "colour" / "config.cfg"
DATA_PATH = ROOT / "datasets" / "colour" / "vad_rgb_dataset.csv"
CHECKPOINT_DIR = ROOT / "checkpoints" / "colour"
LOG_DIR = ROOT / "logs" / "vad_color"

def set_seed(seed: int) -> None:
    R.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_config(path: Path) -> configparser.ConfigParser:
    cfg = configparser.ConfigParser()
    read_files = cfg.read(path)

    if not read_files:
        raise FileNotFoundError(f"Could not read config file: {path}")

    return cfg

def build_settings(cfg):
    return {
        "seed": cfg.getint("project", "seed", fallback=42),

        "csv_path": ROOT / cfg.get("data", "csv_path"),
        "num_workers": cfg.getint("data", "num_workers", fallback=0),

        "hidden_dim": cfg.getint("model", "hidden_dim"),
        "depth": cfg.getint("model", "depth"),
        "dropout": cfg.getfloat("model", "dropout"),

        "batch_size": cfg.getint("train", "batch_size"),
        "epochs": cfg.getint("train", "epochs"),
        "learning_rate": cfg.getfloat("train", "learning_rate"),

        "train_split": cfg.getfloat("split", "train_split"),
        "val_split": cfg.getfloat("split", "val_split"),
        "test_split": cfg.getfloat("split", "test_split"),
    }

def build_dataloaders(settings):
    df = pd.read_csv(settings["csv_path"])

    X = torch.tensor(
        df[["valence", "arousal", "dominance"]].values,
        dtype=torch.float32
    )
    y = torch.tensor(
        df[["r", "g", "b"]].values,
        dtype=torch.float32
    )

    dataset = TensorDataset(X, y)

    total = len(dataset)
    train_size = int(settings["train_split"] * total)
    val_size = int(settings["val_split"] * total)
    test_size = total - train_size - val_size

    train_ds, val_ds, test_ds = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(settings["seed"])
    )

    train_loader = DataLoader(train_ds, batch_size=settings["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=settings["batch_size"], shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=settings["batch_size"], shuffle=False)

    return train_loader, val_loader, test_loader

def run_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    running_loss = 0.0

    with torch.set_grad_enabled(is_train):
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            pred = model(x)
            loss = criterion(pred, y)

            if is_train:
                loss.backward()
                optimizer.step()

            running_loss += loss.item()

    return running_loss / len(loader)

def rgb_to_hex(rgb: torch.Tensor) -> str:
    rgb = rgb.detach().cpu().clamp(0, 1)
    r, g, b = (rgb * 255).int().tolist()
    return f"#{r:02X}{g:02X}{b:02X}"


def main() -> None:
    cfg = load_config(CONFIG_PATH)
    settings = build_settings(cfg)

    set_seed(settings["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader = build_dataloaders(settings)

    model = VADToColor(
        hidden_dim=settings["hidden_dim"],
        depth=settings["depth"],
        dropout=settings["dropout"]
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings["learning_rate"]
    )

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    train_losses = []
    val_losses = []

    epochs = settings["epochs"]

    epoch_bar = tqdm(range(epochs), desc="Epochs")

    for epoch in epoch_bar:
        train_loss = run_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
        )

        val_loss = run_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        epoch_bar.set_postfix(
            train_loss=f"{train_loss:.4f}",
            val_loss=f"{val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = CHECKPOINT_DIR / "vad_to_color_best.pt"
            torch.save(model.state_dict(), best_path)

    test_loss = run_one_epoch(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        optimizer=None,
    )

    final_path = CHECKPOINT_DIR / "vad_to_color_final.pt"
    torch.save(model.state_dict(), final_path)

    log_data = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val_loss": best_val_loss,
        "test_loss": test_loss,
        "epochs": epochs,
        "batch_size": settings["batch_size"],
        "learning_rate": settings["learning_rate"],
        "hidden_dim": settings["hidden_dim"],
        "depth": settings["depth"],
        "dropout": settings["dropout"],
    }

    log_path = LOG_DIR / "vad_to_color_train_log.json"
    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=4)



