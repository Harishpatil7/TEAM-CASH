"""
download_datasets.py
─────────────────────────────────────────────────────────────────────────────
Automated dataset acquisition for the Emergency Voice Triage System.

Downloads:
  1. RAVDESS  — via kagglehub → data/ravdess/
  2. TESS     — via kagglehub → data/tess/
  3. CREMA-D  — via git sparse-checkout (AudioWAV only) → data/crema-d/

Then validates file counts and prints a summary.

Usage:
  python data/download_datasets.py
─────────────────────────────────────────────────────────────────────────────
"""

import sys
import shutil
import subprocess
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_DIR    = Path(__file__).parent
RAVDESS_DIR = DATA_DIR / "ravdess"
TESS_DIR    = DATA_DIR / "tess"
CREMAD_DIR  = DATA_DIR / "crema-d"
CREMAD_REPO = DATA_DIR / "crema-d-repo"


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def count_wavs(directory: Path) -> int:
    return len(list(directory.rglob("*.wav")))


# ══════════════════════════════════════════════════════════════════════════
#  Step 0 — install kagglehub if missing
# ══════════════════════════════════════════════════════════════════════════
def install_kagglehub():
    try:
        import kagglehub  # noqa
        print("[setup] ✅  kagglehub already installed")
    except ImportError:
        print("[setup] 📦  Installing kagglehub...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kagglehub", "-q"])
        print("[setup] ✅  kagglehub installed")


# ══════════════════════════════════════════════════════════════════════════
#  Step 1 — RAVDESS
# ══════════════════════════════════════════════════════════════════════════
def download_ravdess():
    print("\n[RAVDESS] ⬇️   Downloading RAVDESS...")
    import kagglehub  # noqa

    cache_path = Path(kagglehub.dataset_download("uwrfkaggler/ravdess-emotional-speech-audio"))
    print(f"[RAVDESS] 📂  Cache path: {cache_path}")

    ensure_dir(RAVDESS_DIR)

    # Copy all WAV files preserving subfolder structure
    wav_files = list(cache_path.rglob("*.wav"))
    print(f"[RAVDESS] 🔄  Copying {len(wav_files)} WAV files → {RAVDESS_DIR}")

    copied = 0
    for wav in wav_files:
        rel     = wav.relative_to(cache_path)
        dest    = RAVDESS_DIR / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(wav, dest)
        copied += 1

    print(f"[RAVDESS] ✅  {copied} files copied → {RAVDESS_DIR}")


# ══════════════════════════════════════════════════════════════════════════
#  Step 2 — TESS
# ══════════════════════════════════════════════════════════════════════════
def download_tess():
    print("\n[TESS]    ⬇️   Downloading TESS...")
    import kagglehub  # noqa

    cache_path = Path(kagglehub.dataset_download("ejlok1/toronto-emotional-speech-set-tess"))
    print(f"[TESS]    📂  Cache path: {cache_path}")

    ensure_dir(TESS_DIR)

    wav_files = list(cache_path.rglob("*.wav"))
    print(f"[TESS]    🔄  Copying {len(wav_files)} WAV files → {TESS_DIR}")

    copied = 0
    for wav in wav_files:
        rel  = wav.relative_to(cache_path)
        dest = TESS_DIR / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(wav, dest)
        copied += 1

    print(f"[TESS]    ✅  {copied} files copied → {TESS_DIR}")


# ══════════════════════════════════════════════════════════════════════════
#  Step 3 — CREMA-D via Kaggle mirror (GitHub LFS budget exceeded)
# ══════════════════════════════════════════════════════════════════════════
def download_crema_d():
    print("\n[CREMA-D] Downloading CREMA-D via Kaggle mirror...")
    import kagglehub

    # If already populated, skip
    if count_wavs(CREMAD_DIR) > 100:
        print(f"[CREMA-D] Already present ({count_wavs(CREMAD_DIR)} files) -- skipping")
        return

    ensure_dir(CREMAD_DIR)

    # Try primary Kaggle mirror
    kaggle_slugs = [
        "ejlok1/cremad",
        "dmitrybobkov/crema-d",
    ]
    cache_path = None
    for slug in kaggle_slugs:
        try:
            print(f"[CREMA-D]   Trying kaggle slug: {slug}")
            cache_path = Path(kagglehub.dataset_download(slug))
            print(f"[CREMA-D]   Cache path: {cache_path}")
            break
        except Exception as e:
            print(f"[CREMA-D]   Slug {slug} failed: {e}")

    if cache_path is None:
        raise RuntimeError("All Kaggle mirrors for CREMA-D failed.")

    # Find all .wav files inside downloaded path
    wav_files = list(cache_path.rglob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"No WAV files found in {cache_path}")

    print(f"[CREMA-D] Copying {len(wav_files)} WAV files -> {CREMAD_DIR}")
    for wav in wav_files:
        dest = CREMAD_DIR / wav.name
        shutil.copy2(wav, dest)

    print(f"[CREMA-D] {len(wav_files)} files copied -> {CREMAD_DIR}")


# ══════════════════════════════════════════════════════════════════════════
#  Summary
# ══════════════════════════════════════════════════════════════════════════
def print_summary():
    print("\n" + "═" * 55)
    print("  📊  Dataset Download Summary")
    print("═" * 55)

    for name, directory in [("RAVDESS", RAVDESS_DIR), ("TESS", TESS_DIR), ("CREMA-D", CREMAD_DIR)]:
        count = count_wavs(directory) if directory.exists() else 0
        status = "✅" if count > 50 else "❌ MISSING"
        print(f"  {status}  {name:<10} {count:>5} WAV files   →  {directory}")

    print("═" * 55)
    total = sum(count_wavs(d) for d in [RAVDESS_DIR, TESS_DIR, CREMAD_DIR] if d.exists())
    print(f"  TOTAL: {total} audio files ready for feature extraction")
    print("═" * 55)
    print("\n  ✅  Next step: python data/prepare_data.py\n")


# ══════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n🚨 Emergency Voice Triage — Dataset Downloader")
    print("=" * 55)

    install_kagglehub()

    errors = []

    try:
        download_ravdess()
    except Exception as e:
        print(f"[RAVDESS] ❌  Failed: {e}")
        errors.append("RAVDESS")

    try:
        download_tess()
    except Exception as e:
        print(f"[TESS]    ❌  Failed: {e}")
        errors.append("TESS")

    try:
        download_crema_d()
    except Exception as e:
        print(f"[CREMA-D] ❌  Failed: {e}")
        errors.append("CREMA-D")

    print_summary()

    if errors:
        print(f"⚠️  These datasets had errors: {', '.join(errors)}")
        print("   Check your internet connection and Kaggle credentials.")
        sys.exit(1)
