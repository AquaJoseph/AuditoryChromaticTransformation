import configparser
import math
import torch

from pathlib import Path
from __future__ import annotations


VAD_TO_RGB_INDEX = {
    "valence": 1,
    "arousal": 0,
    "dominance": 2,
}

def build_colour_bias_settings(cfg: configparser.ConfigParser) -> dict:
    """Build colour bias settings from the config file."""
    return {
        "base_low": cfg.getfloat("colour_bias", "base_low"),
        "base_high": cfg.getfloat("colour_bias", "base_high"),
        "rank_bonuses": [
            float(x.strip())
            for x in cfg.get("colour_bias", "rank_bonuses").split(",")
        ],
        "variance_alpha": cfg.getfloat("colour_bias", "variance_alpha"),
        "spread_gain": cfg.getfloat("colour_bias", "spread_gain"),
    }


def load_colour_bias_settings(config_path: str | Path) -> dict:
    """Load the config file and return only the colour bias settings."""
    cfg = configparser.ConfigParser()
    cfg.read(config_path)
    return build_colour_bias_settings(cfg)


def clamp_tensor(x: torch.Tensor, low: float = 0.0, high: float = 255.0) -> torch.Tensor:
    """Clamp tensor into valid RGB byte range."""
    return torch.clamp(x, min=low, max=high)


def rank_vad_tensor(vad_row: torch.Tensor) -> list[tuple[str, float]]:
    """Return VAD values sorted from highest to lowest."""
    valence, arousal, dominance = vad_row.tolist()
    named = [
        ("valence", valence),
        ("arousal", arousal),
        ("dominance", dominance),
    ]
    return sorted(named, key=lambda x: x[1], reverse=True)


def minmax_scale_tensor(
    value: torch.Tensor,
    low: float,
    high: float,
    out_low: float,
    out_high: float,
) -> torch.Tensor:
    """Min-max scale a tensor value into a target range."""
    if math.isclose(high, low):
        midpoint = (out_low + out_high) / 2.0
        return torch.full_like(value, midpoint)

    t = (value - low) / (high - low)
    return out_low + t * (out_high - out_low)


def rgb_offsets_from_mean_tensor(rgb_255: torch.Tensor) -> torch.Tensor:
    """Return per-row RGB offsets from each row mean."""
    mu = rgb_255.mean(dim=1, keepdim=True)
    return rgb_255 - mu


def soft_base_rgb_from_vad_row(
    vad_row: torch.Tensor,
    colour_bias_settings: dict,
) -> torch.Tensor:
    """Convert one VAD row into a base RGB colour using colour bias settings."""
    base_low = colour_bias_settings["base_low"]
    base_high = colour_bias_settings["base_high"]
    rank_bonuses = colour_bias_settings["rank_bonuses"]
    spread_gain = colour_bias_settings["spread_gain"]

    if len(rank_bonuses) != 3:
        raise ValueError(
            f"Expected 3 rank bonuses, got {len(rank_bonuses)}: {rank_bonuses}"
        )

    rgb = torch.zeros(3, dtype=torch.float32, device=vad_row.device)

    named_scores = {
        "valence": vad_row[0],
        "arousal": vad_row[1],
        "dominance": vad_row[2],
    }

    for name, score in named_scores.items():
        idx = VAD_TO_RGB_INDEX[name]
        rgb[idx] = minmax_scale_tensor(
            score,
            low=0.0,
            high=1.0,
            out_low=base_low,
            out_high=base_high,
        )

    ranked = rank_vad_tensor(vad_row)

    for rank_idx, (name, _) in enumerate(ranked):
        idx = VAD_TO_RGB_INDEX[name]
        rgb[idx] += rank_bonuses[rank_idx]

    spread = torch.max(vad_row) - torch.min(vad_row)
    spread_boost = spread * spread_gain

    winner_name = ranked[0][0]
    loser_name = ranked[-1][0]

    rgb[VAD_TO_RGB_INDEX[winner_name]] += spread_boost
    rgb[VAD_TO_RGB_INDEX[loser_name]] -= spread_boost * 0.45

    return clamp_tensor(rgb)


def apply_soft_vad_colour_bias(
    rgb_pred: torch.Tensor,
    vad_calibrated: torch.Tensor,
    colour_bias_settings: dict,
) -> torch.Tensor:
    """Apply VAD-based colour bias while preserving some variance from the original RGB."""
    variance_alpha = colour_bias_settings["variance_alpha"]

    if rgb_pred.ndim != 2 or rgb_pred.shape[-1] != 3:
        raise ValueError(f"Expected rgb_pred shape [B, 3], got {tuple(rgb_pred.shape)}")

    if vad_calibrated.ndim != 2 or vad_calibrated.shape[-1] != 3:
        raise ValueError(
            f"Expected vad_calibrated shape [B, 3], got {tuple(vad_calibrated.shape)}"
        )

    if rgb_pred.shape[0] != vad_calibrated.shape[0]:
        raise ValueError(
            "rgb_pred and vad_calibrated must have the same batch size, "
            f"got {rgb_pred.shape[0]} and {vad_calibrated.shape[0]}"
        )

    if rgb_pred.max() <= 1.0:
        original_rgb_255 = rgb_pred * 255.0
    else:
        original_rgb_255 = rgb_pred.clone()

    base_rows = []
    for i in range(vad_calibrated.shape[0]):
        base_rows.append(
            soft_base_rgb_from_vad_row(vad_calibrated[i], colour_bias_settings)
        )

    base_rgb_255 = torch.stack(base_rows, dim=0)

    offsets = rgb_offsets_from_mean_tensor(original_rgb_255)
    transformed_rgb_255 = clamp_tensor(base_rgb_255 + variance_alpha * offsets)

    return transformed_rgb_255 / 255.0