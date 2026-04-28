"""
audio_capture.py
─────────────────────────────────────────────────────────────────────────────
Audio input handlers for two modes:
  1. File mode   — Accepts uploaded WAV bytes or a file path
  2. Stream mode — Accumulates raw PCM chunks from WebSocket
                   and yields analysis-ready numpy arrays

WebSocket audio format (from browser MediaRecorder):
  - PCM Float32, mono, 16 000 Hz (downsample target: 22 050 Hz handled in preprocess)
  - Chunk size: ~512 ms of audio per WebSocket message
─────────────────────────────────────────────────────────────────────────────
"""

import io
import numpy as np
import soundfile as sf
from pathlib import Path

# Minimum audio length (seconds) required before analysis is triggered
MIN_ANALYSIS_SECONDS = 1.5
STREAM_SAMPLE_RATE   = 16_000   # Browser sends at 16 kHz


def load_wav_bytes(wav_bytes: bytes) -> tuple[np.ndarray, int]:
    """
    Load raw WAV bytes (from HTTP file upload) into a numpy waveform.
    Resamples to TARGET_SR (22050 Hz) immediately — same as librosa.load()
    with sr=22050 used in the file-path code path.

    Parameters
    ----------
    wav_bytes : bytes   Raw bytes of a WAV file

    Returns
    -------
    (y, sr) : (np.ndarray, int)   Waveform at 22050 Hz
    """
    import librosa
    TARGET_SR = 22_050

    buffer = io.BytesIO(wav_bytes)
    y, sr  = sf.read(buffer, dtype="float32", always_2d=False)

    # Convert stereo to mono
    if y.ndim > 1:
        y = np.mean(y, axis=1)

    # Resample to standard rate if needed (RAVDESS is 48kHz, TESS is 24kHz)
    if sr != TARGET_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
        sr = TARGET_SR

    return y.astype(np.float32), sr


def load_wav_file(path: str | Path) -> tuple[np.ndarray, int]:
    """
    Load a WAV file from a local path.

    Parameters
    ----------
    path : str | Path   Absolute or relative path to a .wav file

    Returns
    -------
    (y, sr) : (np.ndarray, int)   Waveform array + sample rate
    """
    y, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    return y.astype(np.float32), sr


class AudioStreamBuffer:
    """
    Accumulates raw PCM Float32 chunks received over a WebSocket connection.

    The browser MediaRecorder sends audio as ArrayBuffer (Float32Array).
    Each message is a chunk of ~512 ms. This buffer accumulates chunks
    and yields a full waveform once MIN_ANALYSIS_SECONDS is reached.

    Usage
    -----
        buf = AudioStreamBuffer()
        while True:
            chunk_bytes = await ws.receive_bytes()
            result = buf.push(chunk_bytes)
            if result is not None:
                y, sr = result
                # run analysis
    """

    def __init__(self,
                 sample_rate: int = STREAM_SAMPLE_RATE,
                 min_seconds: float = MIN_ANALYSIS_SECONDS):
        self._sr          = sample_rate
        self._min_samples = int(min_seconds * sample_rate)
        self._buffer: list[np.ndarray] = []
        self._total_samples = 0

    def push(self, chunk_bytes: bytes) -> tuple[np.ndarray, int] | None:
        """
        Push a raw PCM Float32 audio chunk into the buffer.

        Parameters
        ----------
        chunk_bytes : bytes
            Raw bytes representing a Float32Array from the browser.

        Returns
        -------
        (y, sr) if enough audio has accumulated; None otherwise.
        """
        chunk = np.frombuffer(chunk_bytes, dtype=np.float32)
        self._buffer.append(chunk)
        self._total_samples += len(chunk)

        if self._total_samples >= self._min_samples:
            y = np.concatenate(self._buffer)
            self.reset()
            return y, self._sr

        return None

    def reset(self) -> None:
        """Clear the internal buffer (called after yielding a full segment)."""
        self._buffer        = []
        self._total_samples = 0

    @property
    def buffered_seconds(self) -> float:
        """How much audio (in seconds) is currently buffered."""
        return self._total_samples / self._sr
