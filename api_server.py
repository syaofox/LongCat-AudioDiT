"""LongCat-AudioDiT HTTP API Server for Open Source Reader (开源阅读).

API Endpoints:
    GET  /?text=...&speaker=...&model=...    TTS / Voice Cloning
    GET  /health                              Health check
    GET  /speakers                            List available speaker packages
    POST /speakers                            Upload new speaker package
"""

import io
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import soundfile as sf
from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.responses import StreamingResponse, JSONResponse

import audiodit  # auto-registers AudioDiTConfig/AutoDiTModel
from audiodit import AudioDiTModel
from transformers import AutoTokenizer
from utils import normalize_text, normalize_mixed_text, load_audio, approx_duration_from_text, split_text_semantic
from utils import load_polyphone_rules, apply_polyphone_rules

torch.backends.cudnn.benchmark = False

# ─── Configuration ───────────────────────────────────────────────────────────

MODEL_DIRS = {
    "3.5B": os.getenv("MODEL_DIR_3_5B", "./models/LongCat-AudioDiT-3.5B-bf16"),
    "1B": os.getenv("MODEL_DIR_1B", "./models/LongCat-AudioDiT-1B"),
}

SAMPLES_DIR = os.getenv("SAMPLES_DIR", "/app/samples")
OUTPUTS_DIR = os.getenv("OUTPUTS_DIR", "/app/outputs")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "1B")
DEFAULT_NFE_STEPS = int(os.getenv("DEFAULT_NFE_STEPS", "16"))
DEFAULT_GUIDANCE_METHOD = os.getenv("DEFAULT_GUIDANCE_METHOD", "cfg")
DEFAULT_GUIDANCE_STRENGTH = float(os.getenv("DEFAULT_GUIDANCE_STRENGTH", "4.0"))
DEFAULT_SEED = int(os.getenv("DEFAULT_SEED", "1024"))
DEFAULT_MAX_CHARS = int(os.getenv("DEFAULT_MAX_CHARS", "100"))
DEFAULT_SILENCE_DURATION = float(os.getenv("DEFAULT_SILENCE_DURATION", "0.3"))

os.makedirs(SAMPLES_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)


# ─── Model Manager ───────────────────────────────────────────────────────────

class ModelManager:
    """Singleton model manager for caching loaded models."""

    _instance = None
    model = None
    tokenizer = None
    current_model_key = None
    device = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return cls._instance

    def unload(self) -> None:
        if self.model is not None:
            print(f"正在卸载模型 {self.current_model_key}...")
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None
            self.current_model_key = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("模型已卸载。")

    def load(self, model_key: str) -> tuple[AudioDiTModel, AutoTokenizer]:
        if model_key == self.current_model_key and self.model is not None:
            return self.model, self.tokenizer

        if self.model is not None:
            self.unload()

        model_dir = MODEL_DIRS.get(model_key)
        if model_dir is None:
            raise ValueError(f"模型 {model_key} 配置未找到。")
        if not os.path.isdir(model_dir):
            raise ValueError(f"模型目录不存在: {model_dir}")

        tokenizer_path = os.path.join(os.path.dirname(model_dir), "umt5-base")
        if not os.path.isdir(tokenizer_path):
            raise ValueError(f"Tokenizer 目录不存在: {tokenizer_path}")

        print(f"正在加载模型 {model_key}，路径: {model_dir}...")
        self.model = AudioDiTModel.from_pretrained(model_dir, local_files_only=True).to(self.device)
        self.model.vae.to_half()
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        self.current_model_key = model_key
        print(f"模型 {model_key} 加载完成。")

        return self.model, self.tokenizer

    def get_status(self) -> str:
        if self.current_model_key:
            return f"模型: {self.current_model_key} (已加载)"
        return "未加载模型"


model_manager = ModelManager()


# ─── Reference Package Management ────────────────────────────────────────────

def list_speaker_packages() -> list[str]:
    if not os.path.isdir(SAMPLES_DIR):
        return []
    packages = []
    for filename in os.listdir(SAMPLES_DIR):
        if filename.endswith(".zip"):
            packages.append(filename[:-4])
    return sorted(packages)


def load_speaker_package(package_name: str) -> tuple[str, str]:
    """Load speaker package. Returns (audio_path, reference_text)."""
    zip_path = os.path.join(SAMPLES_DIR, f"{package_name}.zip")
    if not os.path.isfile(zip_path):
        raise ValueError(f"参考包未找到: {package_name}")

    temp_dir = os.path.join(SAMPLES_DIR, f"_load_{package_name}_{int(time.time())}")
    os.makedirs(temp_dir, exist_ok=True)

    try:
        with zipfile.ZipFile(zip_path, "r") as zipf:
            zipf.extractall(temp_dir)

        audio_file = None
        for ext in [".wav", ".mp3", ".flac", ".ogg", ".m4a"]:
            candidate = os.path.join(temp_dir, f"reference_audio{ext}")
            if os.path.isfile(candidate):
                audio_file = candidate
                break

        if audio_file is None:
            raise ValueError("参考包中未找到音频文件。")

        text_file = os.path.join(temp_dir, "reference_text.txt")
        if not os.path.isfile(text_file):
            raise ValueError("参考包中未找到文本文件。")

        with open(text_file, "r", encoding="utf-8") as f:
            text_content = f.read().strip()

        persistent_dir = os.path.join(SAMPLES_DIR, "_loaded")
        os.makedirs(persistent_dir, exist_ok=True)
        persistent_audio = os.path.join(persistent_dir, f"{package_name}_audio{os.path.splitext(audio_file)[1]}")
        shutil.copy2(audio_file, persistent_audio)

        shutil.rmtree(temp_dir, ignore_errors=True)

        return persistent_audio, text_content

    except ValueError:
        raise
    except Exception as e:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise ValueError(f"加载参考包失败: {e}")


def save_speaker_package(audio_path: str, reference_text: str, package_name: str) -> str:
    temp_dir = os.path.join(SAMPLES_DIR, f"_temp_{package_name}")
    os.makedirs(temp_dir, exist_ok=True)

    try:
        audio_ext = os.path.splitext(audio_path)[1] or ".wav"
        audio_dest = os.path.join(temp_dir, f"reference_audio{audio_ext}")
        shutil.copy2(audio_path, audio_dest)

        text_dest = os.path.join(temp_dir, "reference_text.txt")
        with open(text_dest, "w", encoding="utf-8") as f:
            f.write(reference_text.strip())

        zip_path = os.path.join(SAMPLES_DIR, f"{package_name}.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, temp_dir)
                    zipf.write(file_path, arcname)

        return zip_path

    except Exception as e:
        raise ValueError(f"保存参考包失败: {e}")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# ─── Core Inference ──────────────────────────────────────────────────────────

def generate_tts_audio(
    text: str,
    model: AudioDiTModel,
    tokenizer: AutoTokenizer,
    device: torch.device,
    nfe_steps: int = DEFAULT_NFE_STEPS,
    guidance_method: str = DEFAULT_GUIDANCE_METHOD,
    guidance_strength: float = DEFAULT_GUIDANCE_STRENGTH,
    seed: int = DEFAULT_SEED,
    silence_duration: float = DEFAULT_SILENCE_DURATION,
    max_chars: int = DEFAULT_MAX_CHARS,
    country: str = "auto",
) -> tuple[np.ndarray, int]:
    """Generate TTS audio. Returns (waveform, sample_rate)."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    sr = model.config.sampling_rate
    full_hop = model.config.latent_hop
    max_duration = model.config.max_wav_duration

    paragraphs = text.split("\n\n")
    print(f"[TTS推理] 文本总长度: {len(text)}字 | 段落数: {len(paragraphs)}")

    wav_segments = []
    segment_count = 0
    polyphone_rules = load_polyphone_rules()

    for para_idx, para in enumerate(paragraphs):
        para = para.strip()
        if not para:
            if silence_duration > 0:
                silence_wav = np.zeros(int(sr * silence_duration), dtype=np.float32)
                wav_segments.append(silence_wav)
            continue

        chunks = split_text_semantic(para, max_chars=max_chars)
        print(f"  -> 共分割为 {len(chunks)} 个语义块")

        for chunk in chunks:
            polyphone_text, polyphone_logs = apply_polyphone_rules(chunk, rules=polyphone_rules)
            normalized_text = normalize_mixed_text(polyphone_text, country=country)

            inputs = tokenizer([normalized_text], padding="longest", return_tensors="pt")

            dur_sec = approx_duration_from_text(chunk, max_duration=max_duration)
            duration = int(dur_sec * sr // full_hop)
            duration = min(duration, int(max_duration * sr // full_hop))

            print(f"  [推理 {segment_count+1}] 文本: \"{chunk}\" -> 归一化: \"{normalized_text}\"")

            with torch.no_grad():
                output = model(
                    input_ids=inputs.input_ids.to(device),
                    attention_mask=inputs.attention_mask.to(device),
                    prompt_audio=None,
                    duration=duration,
                    steps=nfe_steps,
                    cfg_strength=guidance_strength,
                    guidance_method=guidance_method,
                )

            wav = output.waveform.squeeze().detach().cpu().numpy()
            wav_segments.append(wav)
            segment_count += 1

        if para_idx < len(paragraphs) - 1 and silence_duration > 0:
            silence_wav = np.zeros(int(sr * silence_duration), dtype=np.float32)
            wav_segments.append(silence_wav)

    if not wav_segments:
        raise ValueError("没有有效的文本可合成。")

    final_wav = np.concatenate(wav_segments)
    return final_wav, sr


def generate_clone_audio(
    text: str,
    prompt_audio_path: str,
    prompt_text: str,
    model: AudioDiTModel,
    tokenizer: AutoTokenizer,
    device: torch.device,
    nfe_steps: int = DEFAULT_NFE_STEPS,
    guidance_method: str = DEFAULT_GUIDANCE_METHOD,
    guidance_strength: float = DEFAULT_GUIDANCE_STRENGTH,
    seed: int = DEFAULT_SEED,
    silence_duration: float = DEFAULT_SILENCE_DURATION,
    max_chars: int = DEFAULT_MAX_CHARS,
    country: str = "auto",
) -> tuple[np.ndarray, int]:
    """Generate voice-cloned audio. Returns (waveform, sample_rate)."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    sr = model.config.sampling_rate
    full_hop = model.config.latent_hop
    max_duration = model.config.max_wav_duration

    prompt_wav = load_audio(prompt_audio_path, sr).unsqueeze(0)

    off = 3
    pw = load_audio(prompt_audio_path, sr)
    if pw.shape[-1] % full_hop != 0:
        pw = F.pad(pw, (0, full_hop - pw.shape[-1] % full_hop))
    pw = F.pad(pw, (0, full_hop * off))
    with torch.no_grad():
        plt = model.vae.encode(pw.unsqueeze(0).to(device))
    if off:
        plt = plt[..., :-off]
    prompt_dur = plt.shape[-1]

    prompt_time = prompt_dur * full_hop / sr
    norm_prompt_text = normalize_text(prompt_text)

    paragraphs = text.split("\n\n")
    print(f"[克隆推理] 目标文本长度: {len(text)}字 | 参考音频时长: {prompt_time:.2f}s")

    wav_segments = []
    segment_count = 0
    prompt_wav_device = prompt_wav.to(device)
    polyphone_rules = load_polyphone_rules()

    for para_idx, para in enumerate(paragraphs):
        para = para.strip()
        if not para:
            if silence_duration > 0:
                silence_wav = np.zeros(int(sr * silence_duration), dtype=np.float32)
                wav_segments.append(silence_wav)
            continue

        chunks = split_text_semantic(para, max_chars=max_chars)
        print(f"  -> 共分割为 {len(chunks)} 个语义块")

        for chunk in chunks:
            polyphone_text, polyphone_logs = apply_polyphone_rules(chunk, rules=polyphone_rules)
            norm_target_text = normalize_mixed_text(polyphone_text, country=country)
            full_text = f"{norm_prompt_text} {norm_target_text}"
            inputs = tokenizer([full_text], padding="longest", return_tensors="pt")

            dur_sec = approx_duration_from_text(chunk, max_duration=max_duration - prompt_time)
            approx_pd = approx_duration_from_text(prompt_text, max_duration=max_duration)
            ratio = np.clip(prompt_time / approx_pd, 1.0, 1.5)
            dur_sec = dur_sec * ratio
            duration = int(dur_sec * sr // full_hop)
            duration = min(duration + prompt_dur, int(max_duration * sr // full_hop))

            print(f"  [推理 {segment_count+1}] 文本: \"{chunk}\" -> 归一化: \"{norm_target_text}\"")

            with torch.no_grad():
                output = model(
                    input_ids=inputs.input_ids.to(device),
                    attention_mask=inputs.attention_mask.to(device),
                    prompt_audio=prompt_wav_device,
                    duration=duration,
                    steps=nfe_steps,
                    cfg_strength=guidance_strength,
                    guidance_method=guidance_method,
                )

            wav = output.waveform.squeeze().detach().cpu().numpy()
            wav_segments.append(wav)
            segment_count += 1

        if para_idx < len(paragraphs) - 1 and silence_duration > 0:
            silence_wav = np.zeros(int(sr * silence_duration), dtype=np.float32)
            wav_segments.append(silence_wav)

    if not wav_segments:
        raise ValueError("没有有效的目标文本可合成。")

    final_wav = np.concatenate(wav_segments)
    return final_wav, sr


def wav_to_mp3_bytes(wav: np.ndarray, sr: int) -> bytes:
    """Convert waveform to MP3 bytes using ffmpeg."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
        wav_path = tmp_wav.name

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_mp3:
        mp3_path = tmp_mp3.name

    try:
        sf.write(wav_path, wav, sr)
        subprocess.run(
            ["ffmpeg", "-y", "-i", wav_path, "-b:a", "128k", mp3_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        with open(mp3_path, "rb") as f:
            return f.read()
    finally:
        if os.path.exists(wav_path):
            os.remove(wav_path)
        if os.path.exists(mp3_path):
            os.remove(mp3_path)


def wav_to_wav_bytes(wav: np.ndarray, sr: int) -> bytes:
    """Convert waveform to WAV bytes."""
    buf = io.BytesIO()
    sf.write(buf, wav, sr, format="WAV")
    buf.seek(0)
    return buf.read()


# ─── FastAPI Application ─────────────────────────────────────────────────────

app = FastAPI(title="LongCat-AudioDiT TTS API", version="1.0.0")


@app.on_event("startup")
async def startup_event():
    if torch.cuda.is_available():
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: No GPU detected. Inference will be very slow on CPU.")

    auto_load = os.getenv("AUTO_LOAD_MODEL", "true").lower() == "true"
    if auto_load:
        try:
            model_manager.load(DEFAULT_MODEL)
        except Exception as e:
            print(f"自动加载模型失败: {e}")


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "model": model_manager.get_status(),
        "speakers": list_speaker_packages(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }


@app.get("/speakers")
async def list_speakers():
    return {"speakers": list_speaker_packages()}


@app.post("/speakers")
async def upload_speaker(
    package_name: str = Query(..., description="参考包名称"),
    reference_text: str = Query(..., description="参考音频文本"),
    audio: UploadFile = File(...),
):
    with tempfile.NamedTemporaryFile(suffix=os.path.splitext(audio.filename)[1] or ".wav", delete=False) as f:
        content = await audio.read()
        f.write(content)
        audio_path = f.name

    try:
        zip_path = save_speaker_package(audio_path, reference_text, package_name)
        return {"status": "ok", "package": package_name, "path": zip_path}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)


@app.get("/")
async def synthesize(
    text: str = Query(..., description="要合成的文本"),
    speaker: str | None = Query(None, description="参考包名称（用于声音克隆）"),
    model: str = Query(None, description="模型选择 (1B 或 3.5B)"),
    format: str = Query("mp3", description="输出格式 (mp3 或 wav)"),
    nfe_steps: int = Query(None, description="ODE步数"),
    guidance_method: str = Query(None, description="引导方法 (cfg 或 apg)"),
    guidance_strength: float = Query(None, description="引导强度"),
    seed: int = Query(None, description="随机种子"),
    max_chars: int = Query(None, description="最大分割字符数"),
    silence_duration: float = Query(None, description="段落间静音时长"),
    country: str = Query("auto", description="说话人语言 (auto/zh/en/ja/ko/fr/de/es/ru)"),
):
    model_key = model or DEFAULT_MODEL
    nfe = nfe_steps or DEFAULT_NFE_STEPS
    g_method = guidance_method or DEFAULT_GUIDANCE_METHOD
    g_strength = guidance_strength if guidance_strength is not None else DEFAULT_GUIDANCE_STRENGTH
    s = seed if seed is not None else DEFAULT_SEED
    m_chars = max_chars or DEFAULT_MAX_CHARS
    s_dur = silence_duration if silence_duration is not None else DEFAULT_SILENCE_DURATION

    try:
        model_obj, tokenizer = model_manager.load(model_key)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"模型加载失败: {e}")

    try:
        if speaker:
            audio_path, prompt_text = load_speaker_package(speaker)
            wav, sr = generate_clone_audio(
                text=text,
                prompt_audio_path=audio_path,
                prompt_text=prompt_text,
                model=model_obj,
                tokenizer=tokenizer,
                device=model_manager.device,
                nfe_steps=nfe,
                guidance_method=g_method,
                guidance_strength=g_strength,
                seed=s,
                silence_duration=s_dur,
                max_chars=m_chars,
                country=country,
            )
        else:
            wav, sr = generate_tts_audio(
                text=text,
                model=model_obj,
                tokenizer=tokenizer,
                device=model_manager.device,
                nfe_steps=nfe,
                guidance_method=g_method,
                guidance_strength=g_strength,
                seed=s,
                silence_duration=s_dur,
                max_chars=m_chars,
                country=country,
            )

        if format == "wav":
            audio_bytes = wav_to_wav_bytes(wav, sr)
            media_type = "audio/wav"
        else:
            audio_bytes = wav_to_mp3_bytes(wav, sr)
            media_type = "audio/mpeg"

        return StreamingResponse(
            io.BytesIO(audio_bytes),
            media_type=media_type,
            headers={"Content-Disposition": f"attachment; filename=\"tts_{re.sub(r'[^\\w\\-]', '_', speaker or 'default')}.{format}\""},
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"推理失败: {e}")


@app.post("/synthesize")
async def synthesize_post(
    text: str = Query(..., description="要合成的文本"),
    speaker: str | None = Query(None, description="参考包名称（用于声音克隆）"),
    model: str = Query(None, description="模型选择 (1B 或 3.5B)"),
    format: str = Query("mp3", description="输出格式 (mp3 或 wav)"),
    nfe_steps: int = Query(None, description="ODE步数"),
    guidance_method: str = Query(None, description="引导方法 (cfg 或 apg)"),
    guidance_strength: float = Query(None, description="引导强度"),
    seed: int = Query(None, description="随机种子"),
    max_chars: int = Query(None, description="最大分割字符数"),
    silence_duration: float = Query(None, description="段落间静音时长"),
    country: str = Query("auto", description="说话人语言"),
):
    """POST alias for / endpoint (same parameters)."""
    return await synthesize(
        text=text,
        speaker=speaker,
        model=model,
        format=format,
        nfe_steps=nfe_steps,
        guidance_method=guidance_method,
        guidance_strength=guidance_strength,
        seed=seed,
        max_chars=max_chars,
        silence_duration=silence_duration,
        country=country,
    )


if __name__ == "__main__":
    import subprocess
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "7860"))
    workers = int(os.getenv("WORKERS", "1"))

    uvicorn.run(
        "api_server:app",
        host=host,
        port=port,
        workers=workers,
        log_level="info",
    )
