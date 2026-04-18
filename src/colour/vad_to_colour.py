from pathlib import Path

import torch

from src.colour.calibrate import (
    load_calibration_stats,
    stats_to_tensors,
    calibrate_vad,
)
from src.colour.model import VADToColor


ROOT = Path(__file__).resolve().parent.parent.parent
COLOUR_MODEL_PATH = ROOT / "checkpoints" / "colour" / "vad_to_color_best.pt"
CALIBRATION_PATH = ROOT / "checkpoints" / "colour" / "1_10_vad_calibration_stats.json"


def load_colour_model(
    device: torch.device,
    hidden_dim: int = 64,
    depth: int = 2,
    dropout: float = 0.0,
) -> VADToColor:
    if not COLOUR_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Colour model checkpoint not found: {COLOUR_MODEL_PATH}"
        )

    model = VADToColor(
        hidden_dim=hidden_dim,
        depth=depth,
        dropout=dropout,
    ).to(device)

    state_dict = torch.load(COLOUR_MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    return model


def load_calibration_tensors(
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    stats = load_calibration_stats(CALIBRATION_PATH)
    vad_min, vad_max = stats_to_tensors(stats, device=device)
    return vad_min, vad_max


def rgb_to_hex(rgb: torch.Tensor) -> str:
    rgb = rgb.detach().cpu().clamp(0.0, 1.0)
    r, g, b = (rgb * 255).round().int().tolist()
    return f"#{r:02X}{g:02X}{b:02X}"


def rgb_batch_to_hex(rgb_batch: torch.Tensor) -> list[str]:
    return [rgb_to_hex(rgb) for rgb in rgb_batch]


@torch.no_grad()
def vad_to_colour(
    vad_pred: torch.Tensor,
    colour_model: VADToColor,
    vad_min: torch.Tensor,
    vad_max: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    
    single_input = vad_pred.ndim == 1
    if single_input:
        vad_pred = vad_pred.unsqueeze(0)

    vad_calibrated = calibrate_vad(vad_pred, vad_min, vad_max)
    rgb_pred = colour_model(vad_calibrated)
    hex_list = rgb_batch_to_hex(rgb_pred)

    if single_input:
        return vad_calibrated.squeeze(0), rgb_pred.squeeze(0), [hex_list[0]]

    return vad_calibrated, rgb_pred, hex_list


@torch.no_grad()
def audio_vad_to_colour(
    vad_pred: torch.Tensor,
    device: torch.device,
    hidden_dim: int = 64,
    depth: int = 2,
    dropout: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    
    colour_model = load_colour_model(
        device=device,
        hidden_dim=hidden_dim,
        depth=depth,
        dropout=dropout,
    )
    vad_min, vad_max = load_calibration_tensors(device=device)

    return vad_to_colour(
        vad_pred=vad_pred.to(device),
        colour_model=colour_model,
        vad_min=vad_min,
        vad_max=vad_max,
    )