# Pulse-Voice: AI-Driven Emergency Triage

## 1. Overview
**Pulse-Voice** is an AI-based system designed to assist emergency dispatchers by analyzing a caller's voice to estimate stress and urgency levels in real-time [cite: 18, 48]. By extracting key speech patterns, the system provides an objective 'Stress Profile' to help prioritize incoming calls [cite: 18, 48, 54].

---

## 2. System Architecture
Our system is designed as a lightweight, modular pipeline for real-time processing [cite: 29, 61].

### Architecture Diagram


### Module Breakdown [cite: 61]
* **Audio Input Module**: Captures mic input or recorded call files using WebSockets for real-time flow [cite: 62, 85, 87].
* **Preprocessing Module**: Performs noise reduction and normalization to prepare the signal [cite: 63].
* **Feature Extraction**: Uses **Librosa** to extract MFCCs, pitch, energy, and pauses [cite: 64, 81].
* **Analysis Module**: Utilizes an **XGBoost** model to process extracted features [cite: 65, 82].
* **Triage Engine**: Applies priority decision logic to categorize the situation [cite: 66].
* **Dashboard**: Displays results with simple visual indicators and triggers alerts for critical cases [cite: 39, 40, 67].

---

## 3. Code & Data Flow [cite: 69]
The execution plan follows a strict linear data flow to minimize latency [cite: 29, 50, 69]:

1.  **Ingestion**: `FastAPI` receives binary audio chunks through a `WebSocket` connection.
2.  **Processing**: The `audio_processor.py` script uses `Librosa` to convert audio into a numerical feature vector.
3.  **Inference**: The `model_handler.py` feeds the vector into the `XGBoost` regressor to generate a **Stress Score**.
4.  **Classification**: The Triage Engine maps the score to **Low, Medium, or High** priority levels [cite: 37, 54].
5.  **Visualization**: The frontend `index.html` updates the live waveform and priority badges via a JSON response from the backend.

---

## 4. Technical Stack [cite: 78]
* **Backend**: Python / FastAPI [cite: 79, 80]
* **Signal Processing**: Librosa [cite: 81]
* **Machine Learning**: XGBoost [cite: 82]
* **Real-time Data**: WebSockets [cite: 87]
* **Frontend**: HTML, CSS, JavaScript (Chart.js) [cite: 83]

---

## 5. Setup & Usage
1.  **Clone**: `git clone https://github.com/team-cash/pulse-voice.git`
2.  **Install**: `pip install fastapi uvicorn librosa xgboost`
3.  **Run**: `uvicorn main:app --reload`
4.  **Access**: Navigate to `http://localhost:8000` to view the Triage Dashboard.

---

## 6. Project Constraints [cite: 24]
* This is not a medical diagnosis system [cite: 25].
* Performance depends on audio quality [cite: 26].
* System is designed to be fast and lightweight for real-time use [cite: 29, 30].

**Developed by Team CASH** [cite: 5]
