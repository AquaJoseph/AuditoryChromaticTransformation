import librosa as rosa
import numpy as np


def extract_global_features(
    audio_path: str,
    sample_rate: int = 22050,
    hop_length: int = 512,
    n_fft: int = 2048,
) -> np.ndarray | None:
    try:
        waveform, sr = rosa.load(audio_path, sr=sample_rate, mono=True)

        duration = len(waveform) / sr

        tempo, beat_frames = rosa.beat.beat_track(y=waveform, sr=sr, hop_length=hop_length)
        beat_count = len(beat_frames)

        rms = rosa.feature.rms(y=waveform, frame_length=n_fft, hop_length=hop_length)[0]
        centroid = rosa.feature.spectral_centroid(
            y=waveform, sr=sr, n_fft=n_fft, hop_length=hop_length
        )[0]
        bandwidth = rosa.feature.spectral_bandwidth(
            y=waveform, sr=sr, n_fft=n_fft, hop_length=hop_length
        )[0]
        zcr = rosa.feature.zero_crossing_rate(
            y=waveform, hop_length=hop_length
        )[0]
        rolloff = rosa.feature.spectral_rolloff(
            y=waveform, sr=sr, n_fft=n_fft, hop_length=hop_length
        )[0]
        onset_env = rosa.onset.onset_strength(y=waveform, sr=sr, hop_length=hop_length)

        features = np.array(
            [
                float(tempo),
                float(duration),
                float(beat_count),
                float(np.mean(rms)),
                float(np.std(rms)),
                float(np.mean(centroid)),
                float(np.std(centroid)),
                float(np.mean(bandwidth)),
                float(np.std(bandwidth)),
                float(np.mean(zcr)),
                float(np.std(zcr)),
                float(np.mean(rolloff)),
                float(np.std(rolloff)),
                float(np.mean(onset_env)),
                float(np.std(onset_env)),
            ],
            dtype=np.float32,
        )

        return features

    except Exception:
        return None