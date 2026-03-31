from pathlib import Path

import librosa as rosa
import numpy as np


def extract_mel_spectrogram(
        audio_path : str, 
        n_mels : int= 128, 
        hop_length: int = 512, 
        fmin : float = 0.0, 
        fmax : float | None = None, 
        power: float = 2.0, 
        n_fft: int = 2048, 
        sample_rate: int = 22050) -> np.ndarray | None:
        '''
        Do Doc String
        '''

        try: 
                waveform, sr = rosa.load(audio_path, sr=sample_rate, mono= True)

                mel_spec = rosa.feature.melspectrogram(
                        y = waveform,
                        sr = sr,
                        n_fft=n_fft,
                        hop_length=hop_length,
                        n_mels=n_mels,
                        fmin=fmin,
                        fmax=fmax,
                        power=power
                )

                log_mel_spec = rosa.power_to_db(mel_spec, ref=1.0).astype(np.float32)
                return log_mel_spec
        except Exception as e:
                return None

