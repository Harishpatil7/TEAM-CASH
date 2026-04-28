# Pulse-Voice: AI-Driven Emergency Triage

## 1. Overview
**Pulse-Voice** is an AI-based system designed to assist emergency dispatchers by analyzing a caller's voice to estimate stress and urgency levels in real-time . By extracting key speech patterns, the system provides an objective 'Stress Profile' to help prioritize incoming calls.

---

## 2. System Architecture
Our system is designed as a lightweight, modular pipeline for real-time processing.

| Stage | Process | Technology |
| :--- | :--- | :--- |
| **1. Input** | Real-time Audio Capture (Mic/Call) | **WebSockets / FastAPI** |
| **2. Preprocess** | Noise Reduction & Normalization | **Python / NumPy** |
| **3. Extract** | Feature Vectorization (MFCC, Pitch, Energy) | **Librosa** |
| **4. Analyze** | Stress & Urgency Regression | **XGBoost** |
| **5. Output** | Live Dashboard & Emergency Alerts | **HTML/JS / Chart.js** |

---

### **Logical Execution Flow**
```text
+------------------------------------------+
|               Audio Stream               |
|        (Live Capture / WebSocket)        |
+------------------------------------------+
                     |
                     v
+------------------------------------------+
|               Preprocessing              |
|     (Noise Reduction & Normalization)    |
+------------------------------------------+
                     |
                     v
+------------------------------------------+
|       Feature Extraction (Librosa)       |
|          (MFCCs, Pitch, Energy)          |
+------------------------------------------+
                     |
                     v
+------------------------------------------+
|          ML Inference (XGBoost)          |
|         (Calculate Stress Score)         |
+------------------------------------------+
                     |
                     v
+------------------------------------------+
|            Frontend Dashboard            |
|        (Live Vitals & Alerts UI)         |
+------------------------------------------+
```

### Module Breakdown
* **Audio Input Module**: Captures mic input or recorded call files using WebSockets for real-time flow.
* **Preprocessing Module**: Performs noise reduction and normalization to prepare the signal.
* **Feature Extraction**: Uses **Librosa** to extract MFCCs, pitch, energy, and pauses.
* **Analysis Module**: Utilizes an **XGBoost** model to process extracted features.
* **Triage Engine**: Applies priority decision logic to categorize the situation.
* **Dashboard**: Displays results with simple visual indicators and triggers alerts for critical cases.

---

## 3. Code & Data Flow
The execution plan follows a strict linear data flow to minimize latency:

1.  **Ingestion**: `FastAPI` receives binary audio chunks through a `WebSocket` connection.
2.  **Processing**: The `audio_processor.py` script uses `Librosa` to convert audio into a numerical feature vector.
3.  **Inference**: The `model_handler.py` feeds the vector into the `XGBoost` regressor to generate a **Stress Score**.
4.  **Classification**: The Triage Engine maps the score to **Low, Medium, or High** priority levels.
5.  **Visualization**: The frontend `index.html` updates the live waveform and priority badges via a JSON response from the backend.

---

## 4. Technical Stack
* **Backend**: Python / FastAPI
* **Signal Processing**: Librosa
* **Machine Learning**: XGBoost
* **Real-time Data**: WebSockets
* **Frontend**: HTML, CSS, JavaScript (Chart.js)

---

## 5. Setup & Usage
1.  **Clone**: `git clone https://github.com/team-cash/pulse-voice.git`
2.  **Install**: `pip install fastapi uvicorn librosa xgboost`
3.  **Run**: `uvicorn main:app --reload`
4.  **Access**: Navigate to `http://localhost:8000` to view the Triage Dashboard.

---

## 6. Project Constraints
* This is not a medical diagnosis system.
* Performance depends on audio quality.
* System is designed to be fast and lightweight for real-time use.

**Developed by Team CASH**
