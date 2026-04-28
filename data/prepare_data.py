"""
prepare_data.py
─────────────────────────────────────────────────────────────────────────────
Dataset preparation pipeline.
Walks RAVDESS, CREMA-D, and TESS dataset directories, applies
emotion → triage label mapping, outputs a unified dataset.csv.

Expected folder structure (place datasets here before running):
  data/
  ├── ravdess/      ← extracted RAVDESS audio-speech-actor-XX folders
  ├── crema-d/      ← extracted CREMA-D AudioWAV folder contents
  └── tess/         ← extracted TESS folder (OAF_*/YAF_* subfolders)

Output:
  data/dataset.csv  ← columns: filepath, label, source
─────────────────────────────────────────────────────────────────────────────
"""

import csv
import sys
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_DIR      = Path(__file__).parent
RAVDESS_DIR   = DATA_DIR / "ravdess"
CREMAD_DIR    = DATA_DIR / "crema-d"
TESS_DIR      = DATA_DIR / "tess"
OUTPUT_CSV    = DATA_DIR / "dataset.csv"

# ── Label constants ────────────────────────────────────────────────────────
LOW    = "LOW"
MEDIUM = "MEDIUM"
HIGH   = "HIGH"
SKIP   = None   # Exclude this file


# ══════════════════════════════════════════════════════════════════════════
# RAVDESS Parser
# Filename format: 03-01-[emotion]-[intensity]-[statement]-[rep]-[actor].wav
# Emotion: 01=neutral,02=calm,03=happy,04=sad,05=angry,06=fearful,07=disgust,08=surprised
# Intensity: 01=normal, 02=strong
# ══════════════════════════════════════════════════════════════════════════
RAVDESS_MAP = {
    # (emotion_code, intensity_code) → label
    ("06", "02"): HIGH,    # fearful + strong
    ("05", "02"): HIGH,    # angry + strong
    ("06", "01"): MEDIUM,  # fearful + normal
    ("05", "01"): MEDIUM,  # angry + normal
    ("04", "02"): MEDIUM,  # sad + strong
    ("07", "02"): MEDIUM,  # disgust + strong
    ("08", "02"): MEDIUM,  # surprised + strong
    ("04", "01"): LOW,     # sad + normal
    ("07", "01"): LOW,     # disgust + normal
    ("08", "01"): LOW,     # surprised + normal
    ("01", "01"): LOW,     # neutral
    ("01", "02"): LOW,     # neutral strong
    ("02", "01"): LOW,     # calm
    ("02", "02"): LOW,     # calm strong
    ("03", "01"): LOW,     # happy
    ("03", "02"): LOW,     # happy strong
}


def parse_ravdess(directory: Path) -> list[tuple[str, str]]:
    """Parse all RAVDESS WAV files and return [(filepath, label), ...]."""
    records = []
    wav_files = list(directory.rglob("*.wav"))

    if not wav_files:
        print(f"[RAVDESS] ⚠️  No WAV files found in {directory}")
        return records

    for wav in wav_files:
        parts = wav.stem.split("-")
        if len(parts) < 4:
            continue  # Skip malformed filenames

        emotion   = parts[2]
        intensity = parts[3]
        label     = RAVDESS_MAP.get((emotion, intensity), SKIP)

        if label is not None:
            records.append((str(wav), label))

    print(f"[RAVDESS] ✅  {len(records)} files parsed")
    return records


# ══════════════════════════════════════════════════════════════════════════
# CREMA-D Parser
# Filename format: [ActorID]_[Sentence]_[Emotion]_[Level].wav
# Emotion: ANG, DIS, FEA, HAP, NEU, SAD
# Level:   LO, MD, HI, XX  (XX = excluded)
# ══════════════════════════════════════════════════════════════════════════
CREMAD_MAP = {
    ("FEA", "HI"): HIGH,    ("ANG", "HI"): HIGH,
    ("FEA", "MD"): MEDIUM,  ("ANG", "MD"): MEDIUM,
    ("FEA", "LO"): MEDIUM,  ("ANG", "LO"): MEDIUM,
    ("SAD", "HI"): MEDIUM,  ("DIS", "HI"): MEDIUM,
    ("SAD", "MD"): LOW,     ("DIS", "MD"): LOW,
    ("SAD", "LO"): LOW,     ("DIS", "LO"): LOW,
    ("NEU", "LO"): LOW,     ("NEU", "MD"): LOW,
    ("NEU", "HI"): LOW,     ("NEU", "XX"): SKIP,
    ("HAP", "LO"): LOW,     ("HAP", "MD"): LOW,
    ("HAP", "HI"): LOW,     ("HAP", "XX"): SKIP,
    # XX intensity = excluded for all emotions
    ("FEA", "XX"): SKIP,    ("ANG", "XX"): SKIP,
    ("SAD", "XX"): SKIP,    ("DIS", "XX"): SKIP,
}


def parse_crema_d(directory: Path) -> list[tuple[str, str]]:
    """Parse all CREMA-D WAV files and return [(filepath, label), ...]."""
    records = []
    wav_files = list(directory.rglob("*.wav"))

    if not wav_files:
        print(f"[CREMA-D] ⚠️  No WAV files found in {directory}")
        return records

    for wav in wav_files:
        parts = wav.stem.split("_")
        if len(parts) < 4:
            continue

        emotion = parts[2].upper()
        level   = parts[3].upper()
        label   = CREMAD_MAP.get((emotion, level), SKIP)

        if label is not None:
            records.append((str(wav), label))

    print(f"[CREMA-D] ✅  {len(records)} files parsed")
    return records


# ══════════════════════════════════════════════════════════════════════════
# TESS Parser
# Filename format: [word]_[emotion].wav  (inside OAF_*/YAF_* subfolders)
# Emotions: angry, disgust, fear, happy, neutral, ps (pleasant surprise), sad
# ══════════════════════════════════════════════════════════════════════════
TESS_MAP = {
    "fear":    HIGH,
    "angry":   HIGH,
    "sad":     MEDIUM,
    "disgust": MEDIUM,
    "neutral": LOW,
    "happy":   LOW,
    "ps":      LOW,
}


def parse_tess(directory: Path) -> list[tuple[str, str]]:
    """Parse all TESS WAV files and return [(filepath, label), ...]."""
    records = []
    wav_files = list(directory.rglob("*.wav"))

    if not wav_files:
        print(f"[TESS]    ⚠️  No WAV files found in {directory}")
        return records

    for wav in wav_files:
        # Emotion is the last part of the filename after the final underscore
        stem_parts = wav.stem.lower().rsplit("_", 1)
        emotion    = stem_parts[-1] if len(stem_parts) > 1 else ""
        label      = TESS_MAP.get(emotion, SKIP)

        if label is not None:
            records.append((str(wav), label))

    print(f"[TESS]    ✅  {len(records)} files parsed")
    return records


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════
def prepare_dataset() -> None:
    """
    Walk all three dataset directories, apply label mappings,
    write unified dataset.csv.
    """
    print("\n🔍 Preparing dataset...\n")
    all_records: list[tuple[str, str, str]] = []   # (filepath, label, source)

    # Parse each dataset
    for parser, directory, source_name in [
        (parse_ravdess, RAVDESS_DIR, "ravdess"),
        (parse_crema_d, CREMAD_DIR,  "crema-d"),
        (parse_tess,    TESS_DIR,    "tess"),
    ]:
        if not directory.exists():
            print(f"⚠️  Directory not found: {directory}  — skipping {source_name}")
            continue

        records = parser(directory)
        for filepath, label in records:
            all_records.append((filepath, label, source_name))

    # Write CSV
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filepath", "label", "source"])
        writer.writerows(all_records)

    # Summary
    total  = len(all_records)
    counts = {LOW: 0, MEDIUM: 0, HIGH: 0}
    for _, label, _ in all_records:
        counts[label] += 1

    print(f"\n{'─'*50}")
    print(f"✅  Dataset CSV written → {OUTPUT_CSV}")
    print(f"{'─'*50}")
    print(f"  Total files  : {total}")
    print(f"  🟢 LOW        : {counts[LOW]}  ({counts[LOW]/total*100:.1f}%)")
    print(f"  🟡 MEDIUM     : {counts[MEDIUM]}  ({counts[MEDIUM]/total*100:.1f}%)")
    print(f"  🔴 HIGH       : {counts[HIGH]}  ({counts[HIGH]/total*100:.1f}%)")
    print(f"{'─'*50}\n")


if __name__ == "__main__":
    prepare_dataset()
