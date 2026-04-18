from __future__ import annotations

import configparser
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.fma.collate import collate_token_seqs
from src.fma.mel_token_dataset import TokenMelDataset
from src.fma.model import MusicTransformerEncoder
from src.colour.calibrate import compute_percentile_stats, save_calibration_stats


ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = ROOT / "config.cfg"
DATA_PATH = ROOT / "datasets" / "fma" / "mel_vad_dataset.npz"
MODEL_PATH = ROOT / "checkpoints" / "transformer" / "music_transformer_encoder_final.pt"
SAVE_PATH = ROOT / "checkpoints" / "colour" / "0_1_vad_calibration_stats.json"





def build_fma_settings(cfg: configparser.ConfigParser) -> dict:
    return {
        "d_model": cfg.getint("cnn_encoder", "d_model"),
        "nhead": cfg.getint("transformer", "nhead"),
        "num_layers": cfg.getint("transformer", "num_layers"),
        "dim_feedforward": cfg.getint("transformer", "dim_feedforward"),
        "dropout": cfg.getfloat("transformer", "dropout"),
        "channels": [
            int(x.strip())
            for x in cfg.get("cnn_encoder", "channels").split(",")
        ],
        "token_width": cfg.getint("tokenizer", "token_width"),
        "stride": cfg.getint("tokenizer", "stride"),
        "max_tokens": cfg.getint("tokenizer", "max_tokens"),
        "random_crop": cfg.getboolean("tokenizer", "random_crop"),
        "pad_value": cfg.getfloat("tokenizer", "pad_value"),
        "batch_size": cfg.getint("train", "batch_size", fallback=16),
        "num_workers": cfg.getint("data", "num_workers", fallback=0),
    }

def load_config(config_path: Path) -> configparser.ConfigParser:
    cfg = configparser.ConfigParser()
    cfg.read(config_path)

    if not cfg.sections():
        raise FileNotFoundError(f"Could not load config file or config is empty: {config_path}")

    return cfg

def load_model(device: torch.device, settings: dict) -> MusicTransformerEncoder:
    model = MusicTransformerEncoder(
        d_model=settings["d_model"],
        nhead=settings["nhead"],
        num_layers=settings["num_layers"],
        dim_feedforward=settings["dim_feedforward"],
        dropout=settings["dropout"],
        channels=settings["channels"],
    ).to(device)

    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    return model


def build_loader(settings: dict) -> DataLoader:
    npz = np.load(DATA_PATH, allow_pickle=True)
    spectrograms = npz["spectrograms"]

    dataset = TokenMelDataset(
        spectrograms=spectrograms,
        labels=None,
        tokenize=True,
        token_width=settings["token_width"],
        stride=settings["stride"],
        max_tokens=settings["max_tokens"],
        random_crop=settings["random_crop"],
    )

    loader = DataLoader(
        dataset,
        batch_size=settings["batch_size"],
        shuffle=False,
        num_workers=settings["num_workers"],
        collate_fn=lambda b: collate_token_seqs(b, pad_value=settings["pad_value"]),
    )

    return loader


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = load_config(CONFIG_PATH)
    settings = build_fma_settings(cfg)

    model = load_model(device, settings)
    loader = build_loader(settings)

    all_vads = []

    with torch.no_grad():
        for batch in loader:
            x, mask = batch

            x = x.to(device)
            mask = mask.to(device)

            _, vad_pred = model(x, mask)
            all_vads.append(vad_pred.cpu())

    all_vads = torch.cat(all_vads, dim=0).numpy()

    print("Collected VAD shape:", all_vads.shape)

    stats = compute_percentile_stats(all_vads, lower=5, upper=95)

    print("Calibration stats:")
    print(stats)

    save_calibration_stats(stats, SAVE_PATH)

    print(f"Saved to: {SAVE_PATH}")


if __name__ == "__main__":
    main()