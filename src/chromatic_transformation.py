from pathlib import Path
from typing import Any
import os
import subprocess
import tempfile
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
import matplotlib.pyplot as plt
import imageio.v2 as imageio

from src.fma.extractor import extract_mel_spectrogram
from src.fma.tokenizer import mel_tokenizer
from src.fma.collate import collate_token_seqs
from src.fma.model import MusicTransformerEncoder
from src.colour.vad_to_colour import audio_vad_to_colour, rgb_batch_to_hex
from src.colour.colour_bias import apply_soft_vad_colour_bias

ROOT = Path(__file__).resolve().parent.parent
AUDIO_MODEL_PATH = ROOT / "checkpoints" / "transformer" / "run_three_music_transformer_encoder_final.pt"


def load_audio_model(
    device: torch.device,
    d_model: int = 256,
    nhead: int = 8,
    num_layers: int = 4,
    dim_feedforward: int = 512,
    dropout: float = 0.1,
    channels: list[int] | None = None,
) -> MusicTransformerEncoder:

    if channels is None:
        channels = [1, 16, 32, 64, 128]

    if not AUDIO_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Audio model checkpoint not found: {AUDIO_MODEL_PATH}"
        )

    model = MusicTransformerEncoder(
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        channels=channels,
    ).to(device)

    state_dict = torch.load(AUDIO_MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    return model


def split_mel_into_chunks(
    mel: torch.Tensor,
    chunk_frames: int,
    hop_frames: int | None = None,
) -> list[torch.Tensor]:
    if mel.ndim != 2:
        raise ValueError(f"Expected mel shape [128, T], got {mel.shape}")

    _, time_steps = mel.shape

    if hop_frames is None:
        hop_frames = chunk_frames

    chunks = []
    for start in range(0, time_steps, hop_frames):
        end = start + chunk_frames
        chunk = mel[:, start:end]

        if chunk.shape[1] == 0:
            break

        chunks.append(chunk)

        if end >= time_steps:
            break

    return chunks


def mel_chunks_to_token_sequences(
    mel_chunks: list[torch.Tensor],
    token_width: int = 45,
    stride: int = 15,
    max_tokens: int | None = 128,
) -> list[torch.Tensor]:

    token_seqs = []

    for chunk in mel_chunks:
        tokens = mel_tokenizer(
            mel=chunk,
            token_width=token_width,
            stride=stride,
            max_tokens=max_tokens,
            random_crop=False,  # deterministic at inference
        )
        token_seqs.append(tokens)

    return token_seqs


@torch.no_grad()
def infer_colour_sequence_from_tokens(
    token_seqs: list[torch.Tensor],
    device: torch.device,
    audio_model_kwargs: dict[str, Any] | None = None,
    colour_model_kwargs: dict[str, Any] | None = None,
    pad_value: float = -80.0,
) -> dict[str, Any]:

    if audio_model_kwargs is None:
        audio_model_kwargs = {}
    if colour_model_kwargs is None:
        colour_model_kwargs = {}

    if len(token_seqs) == 0:
        raise ValueError("token_seqs is empty")

    x, attention_mask = collate_token_seqs(token_seqs, pad_value=pad_value)
    x = x.to(device)
    attention_mask = attention_mask.to(device)

    audio_model = load_audio_model(device=device, **audio_model_kwargs)
    sequence_embeddings, vad_pred = audio_model(x, attention_mask)

    vad_calibrated, rgb_pred, hex_list = audio_vad_to_colour(
        vad_pred=vad_pred,
        device=device,
        **colour_model_kwargs,
    )

    rgb_pred = apply_soft_vad_colour_bias(
        rgb_pred=rgb_pred,
        vad_calibrated=vad_calibrated,
        variance_alpha=0.4,
    )

    hex_list = rgb_batch_to_hex(rgb_pred)

    return {
        "sequence_embeddings": sequence_embeddings,
        "vad_pred": vad_pred,
        "vad_calibrated": vad_calibrated,
        "rgb_pred": rgb_pred,
        "hex_list": hex_list,
    }


def music_file_to_colour_sequence(
    audio_path: str | Path,
    device: torch.device | None = None,
    *,
    sample_rate: int = 22050,
    hop_length: int = 512,
    chunk_seconds: float = 8.0,
    overlap: float = 0.5,
    token_width: int = 45,
    stride: int = 15,
    max_tokens: int | None = 128,
    audio_model_kwargs: dict[str, Any] | None = None,
    colour_model_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mel_np = extract_mel_spectrogram(
        audio_path=str(audio_path),
        sample_rate=sample_rate,
        hop_length=hop_length,
    )

    if mel_np is None:
        raise RuntimeError(f"Failed to extract Mel spectrogram from: {audio_path}")

    mel = torch.from_numpy(mel_np).float()

    # Convert seconds -> spectrogram frames
    frames_per_second = sample_rate / hop_length
    chunk_frames = max(1, int(chunk_seconds * frames_per_second))

    # overlap in [0, 1)
    hop_seconds = chunk_seconds * (1.0 - overlap)
    if hop_seconds <= 0:
        raise ValueError("overlap must be < 1.0")

    hop_frames = max(1, int(hop_seconds * frames_per_second))

    mel_chunks = split_mel_into_chunks(
        mel=mel,
        chunk_frames=chunk_frames,
        hop_frames=hop_frames,
    )

    token_seqs = mel_chunks_to_token_sequences(
        mel_chunks=mel_chunks,
        token_width=token_width,
        stride=stride,
        max_tokens=max_tokens,
    )

    outputs = infer_colour_sequence_from_tokens(
        token_seqs=token_seqs,
        device=device,
        audio_model_kwargs=audio_model_kwargs,
        colour_model_kwargs=colour_model_kwargs,
    )

    time_steps = [i * hop_seconds for i in range(len(token_seqs))]

    return {
        "audio_path": str(audio_path),
        "mel": mel,
        "time_steps": time_steps,
        "vad_pred": outputs["vad_pred"],
        "vad_calibrated": outputs["vad_calibrated"],
        "rgb_pred": outputs["rgb_pred"],
        "hex_list": outputs["hex_list"],
    }


def plot_colour_sequence(
    rgb_pred: torch.Tensor,
    title: str = "Colour Sequence",
) -> None:
    """
    Static strip view of the colour sequence.
    """
    colors = rgb_pred.detach().cpu().clamp(0.0, 1.0).numpy()

    plt.figure(figsize=(10, 2))
    plt.imshow([colors], aspect="auto")
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()
    plt.show()

def save_colour_sequence_mp4(
    rgb_pred: torch.Tensor,
    audio_path: str | Path,
    output_path: str | Path,
    time_steps: list[float] | None = None,
    fps: int = 30,
    width: int = 1280,
    height: int = 720,
    title: str = "Chromatic Transformation",
) -> None:
    """
    Save an MP4 where the screen colour follows rgb_pred and the original
    audio file is attached as the soundtrack.

    Parameters
    ----------
    rgb_pred:
        Tensor of shape [T, 3] in range [0, 1].
    audio_path:
        Path to the source audio file.
    output_path:
        Final MP4 path.
    time_steps:
        Times for each colour state. If None, colours are spaced evenly at 1 sec.
    fps:
        Output video fps.
    width, height:
        Frame size.
    title:
        Currently unused in frame drawing, but kept for future expansion.
    """
    audio_path = Path(audio_path)
    output_path = Path(output_path)

    colors = rgb_pred.detach().cpu().clamp(0.0, 1.0).numpy()

    if colors.ndim != 2 or colors.shape[1] != 3:
        raise ValueError(f"Expected rgb_pred shape [T, 3], got {colors.shape}")

    if time_steps is None or len(time_steps) != len(colors):
        time_steps = [float(i) for i in range(len(colors))]

    if len(colors) == 0:
        raise ValueError("rgb_pred is empty")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        silent_video_path = tmpdir / "silent_colour_video.mp4"

        writer = imageio.get_writer(
            silent_video_path,
            fps=fps,
            codec="libx264",
            format="FFMPEG",
        )

        for i, color in enumerate(colors):
            start_t = time_steps[i]
            if i < len(colors) - 1:
                end_t = time_steps[i + 1]
            else:
                # hold the final colour for half a second
                end_t = start_t + 0.5

            duration = max(1.0 / fps, end_t - start_t)
            num_frames = max(1, int(round(duration * fps)))

            rgb_255 = (color * 255.0).astype(np.uint8)
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[:, :] = rgb_255

            for _ in range(num_frames):
                writer.append_data(frame)

        writer.close()

        cmd = [
            "ffmpeg",
            "-y",
            "-i", str(silent_video_path),
            "-i", str(audio_path),
            "-c:v", "copy",
            "-c:a", "aac",
            "-shortest",
            str(output_path),
        ]

        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except FileNotFoundError as exc:
            raise RuntimeError(
                "ffmpeg was not found on your system. Install ffmpeg and try again."
            ) from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"ffmpeg failed while creating the MP4:\n{exc.stderr.decode(errors='ignore')}"
            ) from exc


def play_colour_sequence(
    rgb_pred: torch.Tensor,
    time_steps: list[float] | None = None,
    interval: float = 0.5,
    title: str = "Chromatic Transformation",
) -> None:
    colors = rgb_pred.detach().cpu().clamp(0.0, 1.0).numpy()

    plt.ion()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xticks([])
    ax.set_yticks([])

    img = ax.imshow([[colors[0]]], aspect="auto")
    title_obj = ax.set_title(title)

    for i, color in enumerate(colors):
        img.set_data([[color]])

        if time_steps is not None and i < len(time_steps):
            title_obj.set_text(f"{title} | t={time_steps[i]:.2f}s")
        else:
            title_obj.set_text(f"{title} | step={i}")

        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(interval)

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    audio_path = ROOT / "example" / "031125.mp3"

    outputs = music_file_to_colour_sequence(
        audio_path=audio_path,
        device=device,
        chunk_seconds=1,
        overlap=0.5,
        token_width=45,
        stride=15,
        max_tokens=128,
        audio_model_kwargs={
            "d_model": 256,
            "nhead": 8,
            "num_layers": 4,
            "dim_feedforward": 512,
            "dropout": 0.1,
            "channels": [1, 16, 32, 64, 128],
        },
        colour_model_kwargs={
            "hidden_dim": 64,
            "depth": 2,
            "dropout": 0.0,
        },
    )

    print("Raw VAD:")
    print(outputs["vad_pred"][:5])

    print("\nCalibrated VAD:")
    print(outputs["vad_calibrated"][:5])

    print("\nHEX:")
    print(outputs["hex_list"][:10])

    plot_colour_sequence(outputs["rgb_pred"], title="Predicted Colour Sequence")
    play_colour_sequence(
        outputs["rgb_pred"],
        time_steps=outputs["time_steps"],
        interval=1,
        title="Chromatic Transformation"
    )

    output_path = "..\\mp4_out"
    save_colour_sequence_mp4(
        rgb_pred=outputs["rgb_pred"],
        audio_path=audio_path,
        output_path= output_path,
        time_steps=outputs["time_steps"],
        fps=30,
        width=1280,
        height=720,
        title="Chromatic Transformation",
    )