import json
import configparser
import random as R
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import random_split,DataLoader
from tqdm import tqdm

from src.fma.mel_token_dataset import TokenMelDataset
from src.fma.collate import collate_token_seqs
from src.fma.model import MusicTransformerEncoder
from src.fma.loss_fn import HybridLoss


ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = Path(__file__).resolve().parent / "config.cfg"
CHECKPOINT_DIR = ROOT / "checkpoints" / "transformer"
EPOCH_SAVE_DIR = CHECKPOINT_DIR / "epoch_saves"
LOG_DIR = ROOT / "logs"


def load_config(config_path: Path) -> configparser.ConfigParser:
    cfg = configparser.ConfigParser()
    read_files = cfg.read(config_path)

    if not read_files:
        raise FileNotFoundError(f"Could not read config file: {config_path}")

    return cfg


def parse_int_list(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",")]


def parse_float_tuple(value: str) -> tuple[float, float]:
    values = [float(x.strip()) for x in value.split(",")]
    if len(values) != 2:
        raise ValueError(f"Expected two floats, got: {value}")
    return values[0], values[1]


def set_seed(seed: int) -> None:
    R.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_settings(cfg: configparser.ConfigParser) -> dict:
    settings = {
        "seed": cfg.getint("project", "seed", fallback=42),
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "npz_path": ROOT / cfg.get(
            "data", "npz_path", fallback=r"datasets\fma\mel_vad_dataset.npz"
        ),
        "num_workers": cfg.getint("data", "num_workers", fallback=0),
        "test_split": cfg.getfloat("data", "test_split", fallback = 0.2),
        "pad_value": cfg.getfloat("tokenizer", "pad_value", fallback=-80.0),
        "token_width": cfg.getint("tokenizer", "token_width", fallback=45),
        "stride": cfg.getint("tokenizer", "stride", fallback=15),
        "max_tokens": cfg.getint("tokenizer", "max_tokens", fallback=128),
        "random_crop": cfg.getboolean("tokenizer", "random_crop", fallback=True),
        "channels": parse_int_list(cfg.get("cnn_encoder", "channels")),
        "d_model": cfg.getint("cnn_encoder", "d_model", fallback=256),
        "nhead": cfg.getint("transformer", "nhead", fallback=8),
        "num_layers": cfg.getint("transformer", "num_layers", fallback=4),
        "dim_feedforward": cfg.getint("transformer", "dim_feedforward", fallback=512),
        "dropout": cfg.getfloat("transformer", "dropout", fallback=0.1),
        "max_len": cfg.getint("transformer", "max_len", fallback=5000),
        "batch_size": cfg.getint("train", "batch_size", fallback=8),
        "epochs": cfg.getint("train", "epochs", fallback=20),
        "shuffle": cfg.getboolean("train", "shuffle", fallback=True),
        "gradient_clip": cfg.getfloat("train", "gradient_clip", fallback=1.0),
        "log_every": cfg.getint("train", "log_every", fallback=10),
        "save_every": cfg.getint("train", "save_every", fallback=1),
        "learning_rate": cfg.getfloat("train", "learning_rate", fallback=1e-4),
        "weight_decay": cfg.getfloat("train", "weight_decay", fallback=1e-4),
        "optimizer_name": cfg.get("optimizer", "name", fallback="adamw").lower(),
        "betas": parse_float_tuple(cfg.get("optimizer", "betas", fallback="0.9,0.999")),
        "eps": cfg.getfloat("optimizer", "eps", fallback=1e-8),
        "loss_name": cfg.get("loss", "name", fallback="hybrid").lower(),
        "use_diversity": cfg.getboolean("loss", "use_diversity", fallback=True),
        "use_affinity": cfg.getboolean("loss", "use_affinity", fallback=False),
        "diversity_weight": cfg.getfloat("loss.diversity", "weight", fallback=1.0),
        "affinity_weight": cfg.getfloat("loss.affinity", "weight", fallback=1.0),
        "vad_weight": cfg.getfloat("loss.vad", "weight", fallback=1.0),
    }
    return settings


def build_optimizer(
    model: torch.nn.Module,
    settings: dict
) -> torch.optim.Optimizer:
    if settings["optimizer_name"] == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=settings["learning_rate"],
            weight_decay=settings["weight_decay"],
            betas=settings["betas"],
            eps=settings["eps"],
        )

    raise ValueError(f"Unsupported optimizer: {settings['optimizer_name']}")


def build_criterion(settings: dict) -> HybridLoss:
    if settings["loss_name"] != "hybrid":
        raise ValueError(f"Unsupported loss name: {settings['loss_name']}")

    diversity_weight = settings["diversity_weight"] if settings["use_diversity"] else 0.0
    affinity_weight = settings["affinity_weight"] if settings["use_affinity"] else 0.0
    vad_weight = settings["vad_weight"] if settings["vad_weight"] else 0.0

    return HybridLoss(
        diversity_weight=diversity_weight,
        affinity_weight=affinity_weight,
        vad_weight = vad_weight
    )


def build_dataloaders(settings: dict):
    npz_path = settings["npz_path"]

    if not npz_path.exists():
        raise FileNotFoundError(f"Could not find dataset at: {npz_path}")

    print(f"Loading dataset from: {npz_path}")
    npz_data = np.load(npz_path, allow_pickle=True)

    spectrograms = npz_data["spectrograms"]
    labels = npz_data["labels"] if "labels" in npz_data.files else None
    labels = (labels - labels.min()) / (labels.max() - labels.min())
    
    dataset = TokenMelDataset(
        spectrograms=spectrograms,
        labels=labels,
        tokenize=True,
        token_width=settings["token_width"],
        stride=settings["stride"],
        max_tokens=settings["max_tokens"],
        random_crop=settings["random_crop"],
    )

    total_size = len(dataset)

    train_size = int((1 - settings["test_split"]) * total_size)
    remaining = total_size - train_size

    val_size = remaining // 2
    test_size = remaining - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(settings["seed"])
    )

    collate_fn = lambda batch: collate_token_seqs(
        batch, pad_value=settings["pad_value"]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=settings["batch_size"],
        shuffle=True,
        num_workers=settings["num_workers"],
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=settings["batch_size"],
        shuffle=False,
        num_workers=settings["num_workers"],
        collate_fn=collate_fn,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=settings["batch_size"],
        shuffle=False,
        num_workers=settings["num_workers"],
        collate_fn=collate_fn,
    )

    return train_loader, val_loader, test_loader


def build_model(settings: dict) -> MusicTransformerEncoder:
    return MusicTransformerEncoder(
        d_model=settings["d_model"],
        nhead=settings["nhead"],
        num_layers=settings["num_layers"],
        dim_feedforward=settings["dim_feedforward"],
        dropout=settings["dropout"],
        channels=settings["channels"]
    )

def get_loss_type(settings: dict) -> str:
    if settings["loss_name"] != "hybrid":
        return settings["loss_name"]

    parts = []
    if settings["use_diversity"]:
        parts.append("div")
    if settings["use_affinity"]:
        parts.append("aff")

    return "_".join(parts) if parts else "none"


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
    epoch: int,
    epochs: int,
    gradient_clip: float,
    log_every: int,
) -> float:
    model.train()
    running_loss = 0.0

    batch_bar = tqdm(
        enumerate(loader),
        total=len(loader),
        desc=f"Epoch {epoch + 1}/{epochs}",
        leave=False,
        position=1,
    )

    for batch_idx, batch in batch_bar:
        if len(batch) == 3:
            x, mask, labels = batch
            affinity_targets = labels.to(device)
            vad_targets = labels.to(device)
        else:
            x, mask = batch
            affinity_targets = None
            vad_targets = None

        x = x.to(device)
        mask = mask.to(device)

        optimizer.zero_grad(set_to_none=True)

        seq_embeddings, vad_pred = model(x, mask)

        loss = criterion(
            sequence_embeddings=seq_embeddings,
            attention_mask=mask,
            vad_pred=vad_pred,
            vad_targets=vad_targets,
            affinity_targets=affinity_targets if criterion.affinity_weight > 0 else None,
        )

        loss.backward()

        if gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

        optimizer.step()

        running_loss += loss.item()
        avg_loss = running_loss / (batch_idx + 1)

        if batch_idx % max(log_every, 1) == 0 or batch_idx == len(loader) - 1:
            batch_bar.set_postfix(
                batch_loss=f"{loss.item():.4f}",
                avg_loss=f"{avg_loss:.4f}",
            )

    return running_loss / len(loader)

def validate_one_epoch(
    model,
    loader,
    criterion,
    device,
):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                x, mask, labels = batch
                affinity_targets = labels.to(device)
                vad_targets = labels.to(device)
            else:
                x, mask = batch
                affinity_targets = None
                vad_targets = labels

            x = x.to(device)
            mask = mask.to(device)

            seq_embeddings, vad_pred = model(x, mask)

            loss = criterion(
                sequence_embeddings=seq_embeddings,
                attention_mask=mask,
                vad_pred=vad_pred,
                vad_targets=vad_targets,
                affinity_targets=affinity_targets if criterion.affinity_weight > 0 else None,
            )

            running_loss += loss.item()

    return running_loss / len(loader)

def save_epoch_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train_loss: float,
) -> None:
    EPOCH_SAVE_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint_path = EPOCH_SAVE_DIR / f"epoch_{epoch + 1}.pt"
    torch.save(
        {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
        },
        checkpoint_path,
    )


def save_final_model(model: torch.nn.Module) -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    final_path = CHECKPOINT_DIR / "music_transformer_encoder_final.pt"
    torch.save(model.state_dict(), final_path)
    print(f"Saved final model to: {final_path}")


def save_training_log(
    epoch_losses: list[float],
    val_losses: list[float],
    settings: dict,
) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    lr_str = f"{settings['learning_rate']:.0e}"
    loss_type = get_loss_type(settings)

    log_name = (
        f"[e{settings['epochs']}]"
        f"[b{settings['batch_size']}]"
        f"[{loss_type}]"
        f"[lr{lr_str}].json"
    )
    log_path = LOG_DIR / log_name

    log_data = {
        "train_losses": epoch_losses,
        "val_losses": val_losses,
        "final_train_loss": epoch_losses[-1] if epoch_losses else None,
        "final_val_loss": val_losses[-1] if val_losses else None,
        "epochs": settings["epochs"],
        "batch_size": settings["batch_size"],
        "learning_rate": settings["learning_rate"],
        "loss_type": loss_type,
    }

    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=4)

    print(f"Saved training log to: {log_path}")


def main() -> None:
    cfg = load_config(CONFIG_PATH)
    settings = build_settings(cfg)
    print(f"device={settings['device']}")

    set_seed(settings["seed"])

    train_loader, val_loader, _ = build_dataloaders(settings)
    model = build_model(settings).to(settings["device"])
    criterion = build_criterion(settings)
    optimizer = build_optimizer(model, settings)

    epoch_losses = []
    val_losses = []

    epoch_bar = tqdm(range(settings["epochs"]), desc="Epochs", position=0)

    for epoch in epoch_bar:
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=settings["device"],
            epoch=epoch,
            epochs=settings["epochs"],
            gradient_clip=settings["gradient_clip"],
            log_every=settings["log_every"],
        )

        val_loss = validate_one_epoch(
        model=model,
        loader=val_loader,
        criterion=criterion,
        device=settings["device"],
        )
        

        epoch_losses.append(train_loss)
        val_losses.append(val_loss)
        epoch_bar.set_postfix(train_loss=f"{train_loss:.4f}", val= f"{val_loss:.4f}")

        if (epoch + 1) % settings["save_every"] == 0:
            save_epoch_checkpoint(model, optimizer, epoch, train_loss)

    save_final_model(model)
    save_training_log(epoch_losses, val_losses, settings)


if __name__ == "__main__":
    main()