# file: model_handler.py
# XGBoost model inference — loaded once, reused across requests

import os
import numpy as np
import joblib

MODEL_PATH = os.path.join(os.path.dirname(__file__), "ml model", "fast_pulse_voice_model.pkl")

# load model once at import time
_model = None


def _load_model():
    global _model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")
    _model = joblib.load(MODEL_PATH)
    print(f"[model] Loaded XGBoost model from {MODEL_PATH}")


def get_model():
    """Return the cached model, loading it on first call."""
    if _model is None:
        _load_model()
    return _model


def predict_stress(feature_vector):
    """
    Run model inference on the feature vector.

    Parameters
    ----------
    feature_vector : np.ndarray  shape (1, N)

    Returns
    -------
    int   stress_score in range 0–100
    """
    model = get_model()

    # binary model: predict_proba gives [p_normal, p_stress]
    prob = model.predict_proba(feature_vector)[0]

    # stress probability is class 1
    stress_prob = float(prob[1]) if len(prob) > 1 else float(prob[0])

    # convert to 0–100 score
    stress_score = int(round(stress_prob * 100))
    stress_score = max(0, min(100, stress_score))

    return stress_score
