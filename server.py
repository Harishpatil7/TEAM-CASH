# file: server.py

import io
import os
import json
import traceback
import numpy as np
import soundfile as sf
import librosa

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from preprocessing import preprocess_audio, TARGET_SR
from feature_extraction import extract_features, get_debug_features
from model_handler import get_model, predict_stress
from triage import classify

# ==============================
# INIT
# ==============================

app = FastAPI(title="Pulse-Voice Triage System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# preload model at startup so it's not loaded per-request
print("[server] Loading model at startup...")
get_model()
print("[server] Ready.")

# ==============================
# WEBSOCKET AUDIO BUFFER
# ==============================

BUFFER_SECONDS = 0.5  # accumulate ~0.5s of audio before analysis
SILENCE_THRESHOLD = 0.005


class AudioStreamBuffer:
    """
    Buffers raw Float32 PCM chunks from the browser WebSocket.
    Yields a numpy array once enough audio has accumulated (~2.5 seconds).
    """

    def __init__(self, sample_rate=TARGET_SR, min_seconds=BUFFER_SECONDS):
        self._sr = sample_rate
        self._min_samples = int(min_seconds * sample_rate)
        self._chunks = []
        self._total = 0

    def push(self, chunk_bytes):
        """
        Push a raw PCM Float32 chunk.
        Returns (audio_array, sr) when buffer is full, None otherwise.
        """
        chunk = np.frombuffer(chunk_bytes, dtype=np.float32)
        self._chunks.append(chunk)
        self._total += len(chunk)

        if self._total >= self._min_samples:
            audio = np.concatenate(self._chunks)
            self._reset()
            return audio, self._sr

        return None

    def _reset(self):
        self._chunks = []
        self._total = 0

    @property
    def buffered_seconds(self):
        return self._total / self._sr


# ==============================
# ANALYSIS PIPELINE
# ==============================

def run_pipeline(audio, sr):
    """
    Full analysis pipeline:
      1. Preprocess (noise reduction + normalization)
      2. Feature extraction (MFCC, pitch, energy via librosa)
      3. Model inference (XGBoost → stress score 0–100)
      4. Triage (score → Low/Medium/High)
      5. Return structured JSON
    """
    # step 1: preprocess
    audio, sr = preprocess_audio(audio, sr)

    # step 2: extract features
    features = extract_features(audio, sr)

    # step 3: model inference
    stress_score = predict_stress(features)

    # step 4: triage classification
    result = classify(stress_score)

    # step 5: estimated heart rate (resting 60 + stress-driven increase)
    result["heart_rate"] = int(60 + (stress_score / 100) * 80)

    return result




# ==============================
# ROUTES
# ==============================

@app.get("/")
def index():
    html_path = os.path.join(os.path.dirname(__file__), "main.html")
    if not os.path.exists(html_path):
        return JSONResponse({"error": "dashboard file not found"}, status_code=404)
    return FileResponse(html_path)


@app.get("/health")
def health():
    return {"status": "healthy"}




# ==============================
# WEBSOCKET — REAL-TIME AUDIO
# ==============================

@app.websocket("/ws/audio")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    print("[ws] Client connected")

    buffer = AudioStreamBuffer()

    try:
        while True:
            chunk = await ws.receive_bytes()
            result_audio = buffer.push(chunk)

            if result_audio is not None:
                audio, sr = result_audio

                # check for silence
                audio_level = float(np.mean(np.abs(audio)))
                if audio_level < SILENCE_THRESHOLD:
                    await ws.send_json({
                        "status": "silence",
                        "audio_level": audio_level
                    })
                    continue

                try:
                    result = run_pipeline(audio, sr)
                    result["status"] = "result"
                    result["audio_level"] = round(audio_level, 4)
                    await ws.send_json(result)

                except Exception as e:
                    await ws.send_json({
                        "status": "error",
                        "message": str(e)
                    })
                    traceback.print_exc()

    except WebSocketDisconnect:
        print("[ws] Client disconnected")
    except Exception as e:
        print(f"[ws] Unexpected error: {e}")
        traceback.print_exc()


# ==============================
# ENTRY POINT
# ==============================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)