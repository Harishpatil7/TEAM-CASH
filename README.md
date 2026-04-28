# Pulse-Voice Triage System

Pulse-Voice is a real-time, AI-driven emergency voice triage system built for 24-hour hackathons and rapid emergency response scenarios. It analyzes live audio streams from a browser microphone to estimate stress levels, approximate heart rate, and classify emergency priority (Low, Medium, High).

## Features

- **Real-Time Audio Streaming**: Process live audio chunks via WebSockets.
- **Machine Learning Pipeline**: 
  - **Preprocessing**: Noise reduction and audio normalization.
  - **Feature Extraction**: Uses `librosa` to extract MFCCs, zero-crossing rate, RMSE, pitch (jitter/shimmer).
  - **Inference**: Uses an XGBoost Regressor to predict real-time stress scores.
- **Live Triage Dashboard**:
  - Live EKG-style monitor synchronized with the speaker's predicted heart rate and voice activity.
  - Real-time stress scoring and priority categorization (Low, Medium, High).
  - Web-based interface built with HTML5, Chart.js, and Vanilla CSS.

## Architecture

- `server.py`: The main FastAPI backend application handling REST endpoints and WebSocket microphone streams.
- `main.html`: The frontend dashboard UI, featuring microphone capture and real-time plotting.
- `preprocessing.py`: Audio format conversion, silence trimming, and noise reduction.
- `feature_extraction.py`: Advanced audio feature extraction using `librosa`.
- `model_handler.py`: Model loading and inference execution for predicting stress scores.
- `triage.py`: Logic to convert numeric stress scores into actionable priority levels.

## Prerequisites

Ensure you have Python 3.10+ installed.

```bash
# Install required dependencies
pip install fastapi uvicorn websockets numpy soundfile librosa xgboost python-multipart
```

*(Note: Depending on your exact environment, you may also need `scipy` or specific system-level audio libraries).*

## Running the Application Locally

1. **Start the FastAPI Server**:
   ```bash
   uvicorn server:app --host 0.0.0.0 --port 8000 --reload
   ```

2. **Access the Dashboard**:
   Open your web browser and navigate to: [http://localhost:8000](http://localhost:8000)

3. **Test with Browser Mic**:
   Click **"🎙 Start Mic"** on the dashboard. Grant microphone permissions, and speak. You will see the EKG monitor and metrics respond in real-time.

## Disclaimer
This project was developed as a hackathon prototype. The machine learning models and vitals estimations (like heart rate derived from voice) are experimental approximations and are **not intended for real medical diagnosis or production emergency triage**.
