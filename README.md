# Pulse-Voice: AI-Driven Emergency Triage

## 1. Overview
**Pulse-Voice** is an AI-based system designed to assist emergency dispatchers by analyzing a caller's voice to estimate stress and urgency levels in real-time . By extracting key speech patterns, the system provides an objective 'Stress Profile' to help prioritize incoming calls.

---

## 2. System Architecture
Our system is designed as a lightweight, modular pipeline for real-time processing.

### Architecture Diagram
graph TD
    %% Input Stage
    Start([Start Call]) --> In[Capture Audio via Mic/WebSocket] 
    
    %% Preprocessing
    In --> Pre[Preprocessing Module: Noise Reduction & Normalization] 
    
    %% Feature Engineering
    Pre --> Features{Feature Extraction} 
    Features -->|Librosa| F1[Pitch & Energy]
    Features -->|Librosa| F2[MFCCs & Spectral Features]
    Features -->|Librosa| F3[Pause Frequency & Duration]
    
    %% Analysis & ML
    F1 & F2 & F3 --> ML[Analysis Module: XGBoost Regressor] 
    ML --> Score[Calculate Stress & Urgency Score]
    
    %% Triage Logic
    Score --> Triage{Triage Engine}
    Triage -->|Score < 0.4| P1[Low Priority]
    Triage -->|Score 0.4 - 0.7| P2[Medium Priority]
    Triage -->|Score > 0.7| P3[High Priority]
    
    %% Output
    P1 & P2 & P3 --> Dash[Update Visual Dashboard]
    P3 --> Alert[Trigger Critical Emergency Alert]
    
    %% End
    Dash --> End([End Session])

    %% Styling
    style Start fill:#f9f,stroke:#333,stroke-width:2px
    style Triage fill:#ff9,stroke:#333,stroke-width:2px
    style Alert fill:#f66,stroke:#333,stroke-width:4px
    style End fill:#f9f,stroke:#333,stroke-width:2px

### Module Breakdown [cite: 61]
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
