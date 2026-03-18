from pathlib import Path

import librosa
import numpy as np


def extract_mel_spectrogram(
        audio_path : str,
        n_mels : int,
        hop_length: int = 512, 
        fmin : float = 0,
        fmax : float = 0.0,
        power: float = 2.0,
        n_fft: int = 2048,
        sample_rate: int = 22050) -> np.ndarray:
