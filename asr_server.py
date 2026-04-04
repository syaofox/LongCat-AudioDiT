"""Qwen3-ASR FastAPI server for audio transcription service."""

import os
import subprocess
import sys
import tempfile
import logging
from contextlib import asynccontextmanager

import torch
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO, format="%(asctime)s [ASR] %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_PATH = os.getenv("ASR_MODEL_PATH", "./models/Qwen3-ASR-0.6B")
HOST = os.getenv("ASR_HOST", "0.0.0.0")
PORT = int(os.getenv("ASR_PORT", "8000"))

model = None


def _ensure_model() -> str:
    """Ensure model exists locally, download if needed."""
    if os.path.isdir(MODEL_PATH):
        return MODEL_PATH

    logger.info("ASR model not found at %s, starting download...", MODEL_PATH)
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "download_model.py")
    try:
        subprocess.run([sys.executable, script, "asr-0.6b"], check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ASR model download failed: {e}")

    if not os.path.isdir(MODEL_PATH):
        raise RuntimeError(f"ASR model download failed, directory still missing: {MODEL_PATH}")

    logger.info("ASR model downloaded successfully to %s", MODEL_PATH)
    return MODEL_PATH


def load_model():
    """Load Qwen3-ASR model (offline only, no network access)."""
    global model
    model_dir = _ensure_model()
    logger.info("Loading Qwen3-ASR-0.6B from %s (offline mode)...", model_dir)
    from qwen_asr import Qwen3ASRModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    model = Qwen3ASRModel.from_pretrained(
        model_dir,
        dtype=dtype,
        device_map=device,
        local_files_only=True,
        max_inference_batch_size=8,
        max_new_tokens=1024,
    )
    logger.info("Qwen3-ASR-0.6B loaded on %s (dtype=%s)", device, dtype)
    return model


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield
    logger.info("Shutting down ASR server")


app = FastAPI(title="Qwen3-ASR Service", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ok", "model": "Qwen3-ASR-0.6B"}


@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...), language: str | None = None):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    content = await audio.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty audio file")

    suffix = os.path.splitext(audio.filename)[1] if audio.filename else ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        results = model.transcribe(audio=tmp_path, language=language)
        if not results:
            raise HTTPException(status_code=500, detail="Transcription returned empty result")
        text = results[0].text
        detected_lang = getattr(results[0], "language", "unknown")
        logger.info("Transcribed (%s): %s", detected_lang, text)
        return {"text": text, "language": detected_lang}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Transcription failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
