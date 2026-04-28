# file: feature_extraction.py
# Acoustic feature extraction using librosa — MFCC, Pitch, Energy

import numpy as np
import librosa


def extract_features(audio, sr):
    """
    Extract acoustic features for stress detection.

    Features (must match training format exactly):
      - MFCC (13 coefficients): mean, std           → 2 values
      - MFCC delta: mean, std                       → 2 values
      - MFCC delta2: mean, std                      → 2 values
      - Chroma STFT: mean, std                      → 2 values
      - Spectral contrast: mean, std                → 2 values
      - Tonnetz: mean, std                          → 2 values
      - RMS energy: mean, std                       → 2 values
      - Zero-crossing rate: mean, std               → 2 values
      - Pitch (F0 via pyin): mean, std              → 2 values
                                              Total: 18 values

    Returns a (1, 18) numpy array ready for model input.
    """
    feats = []

    def stats(x):
        a = np.array(x)
        return [float(np.mean(a)), float(np.std(a))]

    # 1. MFCC
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    feats.extend(stats(mfcc))

    # 2. MFCC delta + delta2
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    feats.extend(stats(delta))
    feats.extend(stats(delta2))

    # 3. Chroma
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    feats.extend(stats(chroma))

    # 4. Spectral contrast
    contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
    feats.extend(stats(contrast))

    # 5. Tonnetz
    try:
        tonnetz = librosa.feature.tonnetz(
            y=librosa.effects.harmonic(audio), sr=sr
        )
        feats.extend(stats(tonnetz))
    except Exception:
        feats.extend([0.0, 0.0])

    # 6. RMS energy
    rms = librosa.feature.rms(y=audio)
    feats.extend(stats(rms))

    # 7. Zero-crossing rate
    zcr = librosa.feature.zero_crossing_rate(audio)
    feats.extend(stats(zcr))

    # 8. Pitch (fundamental frequency via pyin)
    try:
        f0, _, _ = librosa.pyin(audio, fmin=50, fmax=500, sr=sr)
        f0 = np.nan_to_num(f0)
        feats.extend([float(np.mean(f0)), float(np.std(f0))])
    except Exception:
        feats.extend([0.0, 0.0])

    return np.array(feats, dtype=np.float32).reshape(1, -1)


def get_debug_features(audio, sr):
    """
    Return a dict of key acoustic features for debugging / display.
    Separate from model features — used for frontend cards.
    """
    # pitch
    try:
        f0, _, _ = librosa.pyin(audio, fmin=50, fmax=500, sr=sr)
        f0 = np.nan_to_num(f0)
        pitch_mean = float(np.mean(f0))
        pitch_max = float(np.max(f0)) if len(f0) > 0 else 0.0
    except Exception:
        pitch_mean = 0.0
        pitch_max = 0.0

    # energy
    rms = librosa.feature.rms(y=audio)[0]
    energy_mean = float(np.mean(rms))
    energy_max = float(np.max(rms))

    # zcr
    zcr = librosa.feature.zero_crossing_rate(audio)[0]

    return {
        "pitch_mean": round(pitch_mean, 2),
        "pitch_max": round(pitch_max, 2),
        "energy_mean": round(energy_mean, 4),
        "energy_max": round(energy_max, 4),
        "zcr_mean": round(float(np.mean(zcr)), 4),
    }
