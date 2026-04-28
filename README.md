# Pulse-Voice Triage System

Pulse-Voice is a real-time, AI-driven emergency voice triage system built for 24-hour hackathons and rapid emergency response scenarios. It analyzes live audio streams (from a browser microphone or a real Twilio phone call) to estimate stress levels, approximate heart rate, and classify emergency priority (Low, Medium, High).

## Features

- **Real-Time Audio Streaming**: Process live audio chunks via WebSockets.
- **Twilio Media Streams Integration**: Hook into real live phone calls. The backend intercepts Twilio's base64 encoded u-law audio, decodes it, and runs live inference.
- **Machine Learning Pipeline**: 
  - **Preprocessing**: Noise reduction and audio normalization.
  - **Feature Extraction**: Uses `librosa` to extract MFCCs, zero-crossing rate, RMSE, pitch (jitter/shimmer).
  - **Inference**: Uses an XGBoost Regressor to predict real-time stress scores.
- **Live Triage Dashboard**:
  - Live EKG-style monitor synchronized with the speaker's predicted heart rate and voice activity.
  - Real-time stress scoring and priority categorization (Low, Medium, High).
  - Web-based interface built with HTML5, Chart.js, and Vanilla CSS.

## Architecture

- `server.py`: The main FastAPI backend application handling REST endpoints, WebSocket microphone streams, and the Twilio Media Streams webhook.
- `main.html`: The frontend dashboard UI, featuring microphone capture, real-time plotting, and a standalone "Monitor Call" mode.
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

## Twilio Phone Call Integration Setup

To test the system with a real telephone call, you must expose your local server to the internet using a tunneling tool like **ngrok**.

### 1. Start Ngrok
Leave your `uvicorn` server running, open a new terminal, and run:
```bash
ngrok http 8000
```
*Copy the forwarding URL (e.g., `https://abc-123.ngrok.app`).*

### 2. Configure Twilio
1. Create a free account at [Twilio](https://www.twilio.com/) and provision a trial phone number.
2. In the Twilio Console, go to **Phone Numbers > Manage > Active numbers** and select your number.
3. Under the **"Voice & Fax"** section, find **"A CALL COMES IN"**.
4. Select **Webhook**, and paste your ngrok URL with the `/twilio/incoming` endpoint:
   `https://abc-123.ngrok.app/twilio/incoming`
5. Ensure the HTTP method is set to **POST**. Save the configuration.

### 3. Monitor a Live Call
1. Open your local dashboard (`http://localhost:8000`).
2. Click the yellow **"📞 Monitor Call"** button. The dashboard will enter a listening mode waiting for the Twilio stream.
3. Call your Twilio phone number from your cell phone. 
4. The dashboard will instantly visualize the call metrics as you speak!

## Disclaimer
This project was developed as a hackathon prototype. The machine learning models and vitals estimations (like heart rate derived from voice) are experimental approximations and are **not intended for real medical diagnosis or production emergency triage**.
