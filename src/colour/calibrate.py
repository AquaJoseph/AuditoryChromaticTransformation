import json
from pathlib import Path

import numpy as np
import torch


def compute_percentile_stats(
    vad_preds: np.ndarray,
    lower: float = 5.0,
    upper: float = 95.0,
) -> dict:
    if vad_preds.ndim != 2 or vad_preds.shape[1] != 3:
        raise ValueError(
            f"Expected vad_preds shape [N, 3], got {vad_preds.shape}"
        )

    vad_min = np.percentile(vad_preds, lower, axis=0)
    vad_max = np.percentile(vad_preds, upper, axis=0)

    return {
        "lower_percentile": lower,
        "upper_percentile": upper,
        "vad_min": vad_min.tolist(),
        "vad_max": vad_max.tolist(),
    }


def save_calibration_stats(stats: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(stats, f, indent=4)


def load_calibration_stats(path: Path) -> dict:

    if not path.exists():
        raise FileNotFoundError(f"Calibration stats not found: {path}")

    with open(path, "r") as f:
        return json.load(f)


def stats_to_tensors(
    stats: dict,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    vad_min = torch.tensor(stats["vad_min"], dtype=dtype, device=device)
    vad_max = torch.tensor(stats["vad_max"], dtype=dtype, device=device)

    return vad_min, vad_max


def calibrate_vad(
    vad_pred: torch.Tensor,
    vad_min: torch.Tensor,
    vad_max: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    vad_01 = (vad_pred - vad_min) / (vad_max - vad_min + eps)
    return vad_01.clamp(0.0, 1.0)