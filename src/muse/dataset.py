import torch
import numpy as np
from src.muse.extractor import extract_global_features
from src.fma.extractor import extract_mel_spectrogram
from src.muse.utlis import normalise_text, make_track_artist_stem
from torch.utils.data import Dataset
from pathlib import Path




import torch
import numpy as np
from src.muse.extractor import extract_global_features
from src.fma.extractor import extract_mel_spectrogram
from src.muse.utlis import normalise_text, make_track_artist_stem
from torch.utils.data import Dataset
from pathlib import Path


class MuSeDataset(Dataset):
    def __init__(
        self,
        df,
        audio_root: str | Path,
        chunk_size: int = 45,
        stride: int = 22,
        allowed_suffixes: tuple[str, ...] = (".wav", ".mp3", ".flac", ".ogg", ".m4a"),
    ):
        self.audio_root = Path(audio_root)
        self.chunk_size = chunk_size
        self.stride = stride
        self.allowed_suffixes = tuple(s.lower() for s in allowed_suffixes)

        self.file_index = self._build_file_index()
        print(f"Indexed audio files: {len(self.file_index)}")

        self.df = self._filter_existing_rows(df).reset_index(drop=True)
        print(f"Matched dataset rows: {len(self.df)}")

    def __len__(self) -> int:
        return len(self.df)

    def _build_file_index(self) -> dict[str, Path]:
        index = {}

        for path in self.audio_root.rglob("*"):
            if path.is_file() and path.suffix.lower() in self.allowed_suffixes:
                stem = normalise_text(path.stem)
                if stem not in index:
                    index[stem] = path

        return index

    def _find_audio_path(self, track: str, artist: str) -> Path | None:
        stem = make_track_artist_stem(track, artist)
        return self.file_index.get(stem)

    def _filter_existing_rows(self, df):
        keep_rows = []

        for idx in range(len(df)):
            row = df.iloc[idx]
            track = row["track"]
            artist = row["artist"]

            audio_path = self._find_audio_path(track, artist)
            if audio_path is not None:
                keep_rows.append(idx)

        return df.iloc[keep_rows].copy()

    def _chunk_mel(self, mel: np.ndarray) -> np.ndarray:
        _, total_frames = mel.shape
        chunks = []

        start = 0
        while start < total_frames:
            end = start + self.chunk_size
            chunk = mel[:, start:end]

            if chunk.shape[1] < self.chunk_size:
                pad_width = self.chunk_size - chunk.shape[1]
                chunk = np.pad(chunk, ((0, 0), (0, pad_width)), mode="constant")

            chunks.append(chunk)
            start += self.stride

        if not chunks:
            chunks.append(np.zeros((mel.shape[0], self.chunk_size), dtype=np.float32))

        chunks = np.stack(chunks, axis=0)
        chunks = np.expand_dims(chunks, axis=1)

        max_num_tokens = 512
        if chunks.shape[0] > max_num_tokens:
            chunks = chunks[:max_num_tokens]

        return chunks.astype(np.float32)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        track = row["track"]
        artist = row["artist"]

        audio_path = self._find_audio_path(track, artist)
        if audio_path is None:
            return None

        mel = extract_mel_spectrogram(str(audio_path))
        if mel is None:
            return None

        global_features = extract_global_features(str(audio_path))
        if global_features is None:
            return None

        tokens = self._chunk_mel(mel)

        vad = torch.tensor(
            [
                float(row["valence_tags"]),
                float(row["arousal_tags"]),
                float(row["dominance_tags"]),
            ],
            dtype=torch.float32,
        )

        return {
            "tokens": torch.tensor(tokens, dtype=torch.float32),
            "vad": vad,
            "global_features": torch.tensor(global_features, dtype=torch.float32),
            "num_tokens": tokens.shape[0],
        }