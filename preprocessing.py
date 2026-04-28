# file: preprocessing.py
# Audio preprocessing — noise reduction + normalization

import numpy as np
import librosa

TARGET_SR = 16000


def normalize_audio(audio):
    """Peak-normalize waveform to [-1, 1]."""
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak
    return audio.astype(np.float32)


def reduce_noise_simple(audio, sr):
    """
    Basic noise reduction using spectral gating.
    Uses the first 0.25s as a noise profile estimate and subtracts
    its mean spectrum from the rest of the signal.
    """
    n_fft = 2048
    hop = 512

    # estimate noise from the first 0.25s
    noise_samples = int(0.25 * sr)
    noise_clip = audio[:noise_samples] if len(audio) > noise_samples else audio

    noise_stft = np.abs(librosa.stft(noise_clip, n_fft=n_fft, hop_length=hop))
    noise_profile = np.mean(noise_stft, axis=1, keepdims=True)

    # full signal STFT
    S = librosa.stft(audio, n_fft=n_fft, hop_length=hop)
    mag = np.abs(S)
    phase = np.angle(S)

    # spectral subtraction
    cleaned_mag = np.maximum(mag - noise_profile * 0.8, 0.0)
    cleaned = cleaned_mag * np.exp(1j * phase)

    return librosa.istft(cleaned, hop_length=hop).astype(np.float32)


def preprocess_audio(audio, sr):
    """
    Full preprocessing pipeline:
      1. Resample to TARGET_SR if needed
      2. Mono conversion
      3. DC offset removal
      4. Noise reduction
      5. Peak normalization
    """
    # resample
    if sr != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
        sr = TARGET_SR

    # stereo -> mono
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    audio = audio.astype(np.float32)

    # remove DC offset
    audio = audio - np.mean(audio)

    # noise reduction
    if len(audio) > sr * 0.3:  # only if we have enough audio
        audio = reduce_noise_simple(audio, sr)

    # normalize
    audio = normalize_audio(audio)

    return audio, sr
