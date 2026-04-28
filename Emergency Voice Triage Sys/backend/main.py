"""
main.py  — v3 (clean)
─────────────────────────────────────────────────────────────────────────────
FastAPI application — Emergency Voice Triage System backend.

Endpoints:
  GET  /              -> Dispatcher dashboard (index.html)
  GET  /mobile        -> Mobile mic input page
  GET  /health        -> Health check
  GET  /qr            -> Local network URL + WebSocket URL for mobile pairing
  POST /analyze       -> Upload WAV/MP3/FLAC for analysis
  POST /reload-model  -> Hot-reload model after retraining
  WS   /ws            -> Real-time WebSocket stream (desktop)
  WS   /ws/mobile     -> Real-time WebSocket stream (mobile)

Key fixes (v3):
  - Single pipeline call per WS segment (no double-preprocess)
  - ConfirmationBuffer updated from result, alert upgraded in-place
  - Removed amplitude silence filter (was dropping real quiet voices)
  - preprocess() now correctly passes sr so 16kHz audio is resampled
─────────────────────────────────────────────────────────────────────────────
"""

import json
import socket
import traceback
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from audio_capture import load_wav_bytes, AudioStreamBuffer
from preprocess    import preprocess
from features      import extract_features, estimate_heart_rate, compute_jitter_shimmer
from model         import get_model
from triage        import classify, ConfirmationBuffer, Priority

# ── App ────────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "Emergency Voice Triage System",
    description = "Real-time acoustic stress analysis for emergency call prioritisation",
    version     = "3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ── Static files ───────────────────────────────────────────────────────────
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


# ── Helpers ────────────────────────────────────────────────────────────────
def get_local_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def run_pipeline(y_raw, sr_raw: int,
                 denoise: bool = True,
                 confirmed_high: bool = False) -> dict:
    """
    Full pipeline: preprocess -> features -> model -> HR -> jitter -> triage.

    y_raw   : np.ndarray  Raw waveform (any sample rate; will be resampled)
    sr_raw  : int         Sample rate of y_raw (e.g. 16000 from browser mic)
    """
    model = get_model()
    print(f"\n[pipeline] Starting analysis | Input: {len(y_raw)} samples @ {sr_raw}Hz")

    # 1. Preprocess — resamples 16kHz -> 22050Hz, NR, trim, normalize
    y_clean, sr = preprocess(y_raw, sr=sr_raw, denoise=denoise)
    print(f"[pipeline] After preprocess: {len(y_clean)} samples @ {sr}Hz")

    # 2. Extract 54-feature acoustic vector
    features = extract_features(y_clean, sr)
    print(f"[pipeline] Features extracted: 54-dim vector ready")

    # 3. Model inference
    stress_score, confidence = model.predict(features)
    print(f"[pipeline] Model inference: score={stress_score:.4f}, confidence={confidence:.4f}, mode={model.mode}")

    # 4. Heart rate estimation
    try:
        hr_data = estimate_heart_rate(y_clean, sr)
    except Exception as e:
        print(f"[pipeline] ⚠️  Heart rate estimation failed: {e}")
        hr_data = None

    # 5. Jitter / shimmer / HNR
    try:
        vocal_data = compute_jitter_shimmer(y_clean, sr)
    except Exception as e:
        print(f"[pipeline] ⚠️  Vocal quality estimation failed: {e}")
        vocal_data = None

    # 6. Triage classification
    result = classify(
        stress_score   = stress_score,
        confidence     = confidence,
        heart_rate     = hr_data,
        vocal_quality  = vocal_data,
        confirmed_high = confirmed_high,
        features_debug = {
            "pitch_mean":  round(features.pitch_mean, 2),
            "pitch_max":   round(features.pitch_max, 2),
            "pitch_std":   round(features.pitch_std, 2),
            "energy_mean": round(features.energy_mean, 4),
            "energy_max":  round(features.energy_max, 4),
            "pause_ratio": round(features.pause_ratio, 4),
            "zcr_mean":    round(features.zcr_mean, 4),
            "model_mode":  model.mode,
        }
    )

    return result.to_dict()


# ── WebSocket helper ────────────────────────────────────────────────────────
async def process_ws_segment(y_raw, sr_raw, confirm: ConfirmationBuffer,
                              websocket: WebSocket, source: str = "desktop"):
    """
    Run pipeline on one accumulated segment, update confirmation buffer,
    and send result over WebSocket.
    """
    try:
        # Single pipeline call — no double processing
        result = run_pipeline(y_raw, sr_raw, denoise=True, confirmed_high=False)

        # Update confirmation buffer
        prio      = Priority(result["priority"])
        confirmed = confirm.update(prio)
        if prio != Priority.HIGH:
            confirm.reset()

        # Upgrade alert in-place if confirmed by buffer
        if confirmed and prio == Priority.HIGH:
            result["alert"] = True

        if source == "mobile":
            result["source"] = "mobile"

        await websocket.send_text(json.dumps(result))

    except Exception as e:
        await websocket.send_text(
            json.dumps({"status": "error", "message": str(e)})
        )
        traceback.print_exc()


# ══════════════════════════════════════════════════════════════════════════
#  Routes
# ══════════════════════════════════════════════════════════════════════════

@app.get("/")
async def root():
    index = FRONTEND_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return {"status": "running", "version": "3.0.0"}


@app.get("/mobile")
async def mobile():
    page = FRONTEND_DIR / "mobile.html"
    if page.exists():
        return FileResponse(str(page))
    raise HTTPException(status_code=404, detail="mobile.html not found")


@app.get("/health")
async def health():
    return {"status": "healthy", "model_mode": get_model().mode, "version": "3.0.0"}


@app.get("/qr")
async def qr_info():
    local_ip = get_local_ip()
    return {
        "mobile_url": f"http://{local_ip}:8000/mobile",
        "ws_url":     f"ws://{local_ip}:8000/ws/mobile",
        "local_ip":   local_ip,
    }


@app.post("/analyze")
async def analyze_file(file: UploadFile = File(...)):
    """Analyze an uploaded audio file (WAV, MP3, FLAC)."""
    if not file.filename.lower().endswith((".wav", ".mp3", ".flac")):
        raise HTTPException(status_code=400,
                            detail="Unsupported file type. Upload WAV, MP3, or FLAC.")
    try:
        print(f"\n[/analyze] Received file: {file.filename} (size: {file.size or '?'} bytes)")
        raw_bytes = await file.read()
        print(f"[/analyze] Read {len(raw_bytes)} bytes")
        y, sr     = load_wav_bytes(raw_bytes)          # already at 22050Hz
        print(f"[/analyze] Loaded audio: {len(y)} samples @ {sr}Hz")
        result    = run_pipeline(y, sr, denoise=True, confirmed_high=True)
        print(f"[/analyze] Analysis complete: {result['priority']}")
        return {"status": "ok", "result": result}
    except Exception as e:
        print(f"[/analyze] ❌  Error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/reload-model")
async def reload_model():
    get_model().reload()
    return {"status": "ok", "model_mode": get_model().mode}


# ── WebSocket: desktop ──────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_stream(websocket: WebSocket):
    """
    Real-time WebSocket stream — desktop dashboard.
    Receives Float32 PCM at 16kHz, accumulates 1.5s, runs full pipeline.
    """
    await websocket.accept()
    buffer  = AudioStreamBuffer()
    confirm = ConfirmationBuffer()
    stream_sr = 16_000
    print("[ws/desktop] Client connected")

    try:
        while True:
            message = await websocket.receive()

            # Optional metadata frame from frontend:
            # {"type":"meta","sample_rate":48000}
            if message.get("text"):
                try:
                    meta = json.loads(message["text"])
                    if meta.get("type") == "meta" and isinstance(meta.get("sample_rate"), (int, float)):
                        stream_sr = int(meta["sample_rate"])
                        buffer = AudioStreamBuffer(sample_rate=stream_sr)
                        print(f"[ws/desktop] Stream sample rate set to {stream_sr} Hz")
                except Exception:
                    pass
                continue

            data = message.get("bytes")
            if not data:
                continue
            segment = buffer.push(data)
            if segment is not None:
                y_raw, sr_raw = segment
                await process_ws_segment(y_raw, sr_raw, confirm, websocket, "desktop")

    except WebSocketDisconnect:
        print("[ws/desktop] Client disconnected")
    except Exception as e:
        print(f"[ws/desktop] Error: {e}")
        traceback.print_exc()


# ── WebSocket: mobile ───────────────────────────────────────────────────────

@app.websocket("/ws/mobile")
async def websocket_mobile(websocket: WebSocket):
    """
    Real-time WebSocket stream — mobile browser mic.
    Same protocol as /ws but result tagged with source=mobile.
    """
    await websocket.accept()
    buffer  = AudioStreamBuffer()
    confirm = ConfirmationBuffer()
    stream_sr = 16_000
    print("[ws/mobile] Mobile connected")

    try:
        while True:
            message = await websocket.receive()

            if message.get("text"):
                try:
                    meta = json.loads(message["text"])
                    if meta.get("type") == "meta" and isinstance(meta.get("sample_rate"), (int, float)):
                        stream_sr = int(meta["sample_rate"])
                        buffer = AudioStreamBuffer(sample_rate=stream_sr)
                        print(f"[ws/mobile] Stream sample rate set to {stream_sr} Hz")
                except Exception:
                    pass
                continue

            data = message.get("bytes")
            if not data:
                continue
            segment = buffer.push(data)
            if segment is not None:
                y_raw, sr_raw = segment
                await process_ws_segment(y_raw, sr_raw, confirm, websocket, "mobile")

    except WebSocketDisconnect:
        print("[ws/mobile] Mobile disconnected")
    except Exception as e:
        print(f"[ws/mobile] Error: {e}")
        traceback.print_exc()


# ── Entry point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    local_ip = get_local_ip()
    print(f"\n[server] Dashboard  : http://localhost:8000")
    print(f"[server] Mobile URL : http://{local_ip}:8000/mobile")
    print(f"[server] QR info    : http://{local_ip}:8000/qr\n")

    uvicorn.run(
        "main:app",
        host    = "0.0.0.0",
        port    = 8000,
        reload  = True,
        workers = 1,
    )
