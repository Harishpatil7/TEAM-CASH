import os
import time
import random
import numpy as np
import pandas as pd
import librosa
import joblib

from multiprocessing import Pool, cpu_count
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# ==============================
# CONFIG
# ==============================
SAMPLE_RATE = 16000
MAX_DURATION = 3  # seconds
MAX_SAMPLES = 2000
SAMPLE_FRACTION = 0.3
# cache includes some params so we can change extraction without clobbering
CACHE_FILE = f"features_cache_sr{SAMPLE_RATE}_dur{MAX_DURATION}_v2.pkl"

# resolve data dirs relative to this script
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
DATASET_PATHS = {
    "CREMA-D": os.path.join(BASE_DIR, "crema-d"),
    "RAVDESS": os.path.join(BASE_DIR, "ravdess"),
    "TESS": os.path.join(BASE_DIR, "tess")
}

PANIC_LABELS = {"angry", "fear", "fearful", "disgust"}
NORMAL_LABELS = {"neutral", "calm", "happy"}

# ==============================
# DATA LOADING (SAMPLED)
# ==============================
def infer_label(filename, folder):
    name = filename.lower()

    if "crema" in folder.lower():
        if "ang" in name: return "angry"
        if "dis" in name: return "disgust"
        if "fea" in name: return "fearful"
        if "hap" in name: return "happy"
        if "neu" in name: return "neutral"

    if "ravdess" in folder.lower():
        code = filename.split("-")[2]
        return {
            "01": "neutral", "02": "calm", "03": "happy",
            "05": "angry", "06": "fearful", "07": "disgust"
        }.get(code, None)

    if "tess" in folder.lower():
        if "angry" in name: return "angry"
        if "fear" in name: return "fearful"
        if "disgust" in name: return "disgust"
        if "happy" in name: return "happy"
        if "neutral" in name: return "neutral"

    return None


def load_sampled_data():
    files = []

    print("Scanning dataset folders:")
    for dataset, path in DATASET_PATHS.items():
        count = 0
        all_files = []
        if not os.path.exists(path):
            print(f" - {dataset}: {path} (missing)")
            continue

        for root, _, f in os.walk(path):
            for file in f:
                if file.lower().endswith(".wav"):
                    all_files.append((os.path.join(root, file), root))
        count = len(all_files)
        print(f" - {dataset}: {path} -> {count} .wav files")

        if count == 0:
            continue

        k = max(1, int(count * SAMPLE_FRACTION))
        sampled = all_files if k >= count else random.sample(all_files, k)
        files.extend(sampled)

    random.shuffle(files)
    files = files[:MAX_SAMPLES]

    data = []
    skipped = 0
    for filepath, folder in files:
        label = infer_label(os.path.basename(filepath), folder)
        if label is None:
            skipped += 1
            continue

        binary = 1 if label in PANIC_LABELS else 0
        data.append((filepath, binary))

    print(f"Collected {len(data)} samples (skipped {skipped} files due to unknown labels)")
    return pd.DataFrame(data, columns=["path", "label"])


# ==============================
# FAST FEATURE EXTRACTION
# ==============================
def extract_features_fast(row):
    path, label = row

    try:
        y, sr = librosa.load(path, sr=SAMPLE_RATE, duration=MAX_DURATION)
        if y is None or len(y) == 0:
            return None
        y = librosa.util.normalize(y)

        feats = []

        def stats_vec(x):
            # accept 2D arrays (n_features, T) or 1D
            a = np.array(x)
            if a.ndim == 2:
                return [np.mean(a), np.std(a)]
            return [np.mean(a), np.std(a)]

        # MFCC (13)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        feats.extend(stats_vec(mfcc))

        # Delta and delta2
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        feats.extend(stats_vec(delta))
        feats.extend(stats_vec(delta2))

        # Chroma (12)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        feats.extend(stats_vec(chroma))

        # Spectral Contrast
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        feats.extend(stats_vec(contrast))

        # Tonnetz (may fail for short signals)
        try:
            tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
            feats.extend(stats_vec(tonnetz))
        except Exception:
            feats.extend([0.0, 0.0])

        # RMS
        rms = librosa.feature.rms(y=y)
        feats.extend(stats_vec(rms))

        # ZCR
        zcr = librosa.feature.zero_crossing_rate(y)
        feats.extend(stats_vec(zcr))

        # Pitch (f0) using pyin
        try:
            f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=50, fmax=500, sr=sr)
            f0 = np.nan_to_num(f0)
            feats.extend([np.mean(f0), np.std(f0)])
        except Exception:
            feats.extend([0.0, 0.0])

        return feats, label

    except Exception:
        return None


# ==============================
# BUILD DATASET (PARALLEL)
# ==============================
def build_dataset(df):
    if os.path.exists(CACHE_FILE):
        print(f"Loading cached features from {CACHE_FILE}")
        return joblib.load(CACHE_FILE)

    rows = list(df.itertuples(index=False, name=None))

    print(f"Extracting features for {len(rows)} files using {cpu_count()} workers...")
    with Pool(cpu_count()) as p:
        results = p.map(extract_features_fast, rows)

    X, y = [], []
    skipped = 0
    for r in results:
        if r is None:
            skipped += 1
            continue
        feats, label = r
        X.append(feats)
        y.append(label)

    print(f"Feature extraction complete: {len(X)} extracted, {skipped} skipped")

    X = np.array(X)
    y = np.array(y)

    joblib.dump((X, y), CACHE_FILE)
    print(f"Saved feature cache to {CACHE_FILE}")
    return X, y


# ==============================
# MODEL TRAINING
# ==============================
def train_fast_model(X, y):
    model = XGBClassifier(
        n_estimators=80,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss"
    )

    model.fit(X, y)
    return model


# ==============================
# INFERENCE
# ==============================
def predict_sample(file_path, model):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=MAX_DURATION)
    y = librosa.util.normalize(y)

    def stats(x):
        return [np.mean(x), np.std(x)]

    feats = []

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    feats.extend(stats(mfcc))

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    feats.extend(stats(chroma))

    rms = librosa.feature.rms(y=y)
    feats.extend(stats(rms))

    zcr = librosa.feature.zero_crossing_rate(y)
    feats.extend(stats(zcr))

    feats = np.array(feats).reshape(1, -1)

    prob = model.predict_proba(feats)[0][1]
    cls = 1 if prob > 0.5 else 0

    stress = prob * 100
    hr = 60 + (stress / 100) * 80

    label = "Panic/Stress" if cls == 1 else "Normal"

    return {
        "class": label,
        "stress_score": float(stress),
        "estimated_hr": float(hr)
    }


# ==============================
# MAIN
# ==============================
def main():
    start = time.time()

    df = load_sampled_data()
    X, y = build_dataset(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = train_fast_model(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print("Accuracy:", acc)
    print("Training time (s):", time.time() - start)

    joblib.dump(model, "fast_pulse_voice_model.pkl")


if __name__ == "__main__":
    main()