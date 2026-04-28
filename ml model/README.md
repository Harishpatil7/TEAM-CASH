Setup and run

1. Create (or activate) a Python environment (recommended):

```bash
python -m venv venv
# Windows PowerShell
venv\Scripts\Activate.ps1
# or cmd
venv\Scripts\activate.bat
```

2. Install requirements:

```bash
pip install -r requirements.txt
```

3. Ensure your datasets are placed under `data/crema-d`, `data/ravdess`, and `data/tess` (see project root `data/` folder).

4. Run training (feature extraction may take time):

```bash
python main.py
```

Outputs

- Trained classifier saved as `classifier.pkl`
- Trained regressor saved as `regressor.pkl`

Notes

- Feature extraction can be slow depending on dataset size and CPU.
- If you want me to run training now, tell me and I'll start it and report progress.
