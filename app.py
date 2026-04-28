"""Entrypoint: run the FastAPI app defined in server.py.

Usage:
    python app.py
    uvicorn server:app --host 0.0.0.0 --port 8000 --reload
"""
import os

try:
    from server import app  # server.py defines the FastAPI instance
except Exception as e:
    raise RuntimeError("Failed to import FastAPI app from server.py") from e


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=True)