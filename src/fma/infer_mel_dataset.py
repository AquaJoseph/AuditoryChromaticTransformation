import numpy as np
import pandas as pd

from pathlib import Path
from tqdm import tqdm

from src.fma.extractor import extract_mel_spectrogram


def extract_track_id(audio_path : str | Path) -> int: return int(Path(audio_path).stem)

def get_audio_files(root_dir: str | Path) -> list[Path]: return sorted(Path(root_dir).rglob("*.mp3"))
def load_vad(csv_path: str | Path) -> dict[int, np.ndarray]:
    df = pd.read_csv(csv_path)

    df["track_id"] = pd.to_numeric(df["track_id"], errors="coerce")

    bad_track_id_rows = df["track_id"].isna().sum()
    print(f"Bad track_id rows dropped: {bad_track_id_rows}")

    df = df.dropna(subset=["track_id"]).copy()
    df["track_id"] = df["track_id"].astype(int)

    pred_cols = ["valence_pred", "arousal_pred", "dominance_pred"]
    for col in pred_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    bad_pred_rows = df[pred_cols].isna().any(axis=1).sum()
    print(f"Bad prediction rows dropped: {bad_pred_rows}")

    df = df.dropna(subset=pred_cols).copy()

    vad_rows = {
        row["track_id"]: np.array(
            [
                row["valence_pred"],
                row["arousal_pred"],
                row["dominance_pred"],
            ],
            dtype=np.float32,
        )
        for _, row in df.iterrows()
    }

    return vad_rows


def build_dataset(audio_root: Path, csv_path: Path):
    audio_files = get_audio_files(audio_root)


    vad_infer = load_vad(csv_path)


    spectrograms, labels, track_ids, failed = [], [], [], []
    print("CSV rows:", len(vad_infer))
    exit
    for audio_path in tqdm(audio_files, desc="Processing Audio File: "):
        try:
            track_id = extract_track_id(audio_path)

            if track_id not in vad_infer:
                failed.append(str(audio_path))

                continue

            mel = extract_mel_spectrogram(audio_path)

            spectrograms.append(mel)
            labels.append(vad_infer[track_id])
            track_ids.append(track_id)

        except Exception as e:
            print(f"Failed to process {audio_path} due to {e}")
            failed.append(str(audio_path))

    return spectrograms, labels, track_ids, failed

def to_object_array(items):
    arr = np.empty(len(items), dtype=object)
    arr[:] = items

    return arr


def save_dataset(path, spectrograms, labels, track_ids, failed):
    np.savez_compressed(
        path,
        spectrograms=to_object_array(spectrograms),
        labels=np.array(labels, dtype=np.float32),
        track_ids=np.array(track_ids, dtype=np.int32),
        failed=np.array(failed, dtype=object),
    )

def main():
    root = Path.cwd()
    while not (root / "datasets").exists():
        root = root.parent

    audio_root = root / "datasets" / "fma" / "fma_large"
    csv_path = root / "datasets" / "fma" / "fma_metadata" / "modified" / "full_fma_vad_predictions_group.csv"
    output_path = root / "datasets" / "fma" / "mel_vad_dataset.npz"

    spectrograms, labels, track_ids, failed = build_dataset(audio_root, csv_path)

    save_dataset(output_path, spectrograms, labels, track_ids, failed)


if __name__ == "__main__":
    main()