"""LongCat-AudioDiT Gradio WebUI for TTS and Voice Cloning."""

import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime
import numpy as np
import torch
import torch.nn.functional as F
import soundfile as sf
import gradio as gr

import audiodit  # auto-registers AudioDiTConfig/AutoDiTModel
from audiodit import AudioDiTModel
from transformers import AutoTokenizer
from utils import normalize_text, normalize_mixed_text, load_audio, approx_duration_from_text, split_text_semantic
from utils import load_polyphone_rules, save_polyphone_rules, apply_polyphone_rules, _DEFAULT_POLYPHONE_RULES

torch.backends.cudnn.benchmark = False

# Output directory
OUTPUTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# ASR configuration
ASR_MODEL_PATH = os.getenv("ASR_MODEL_PATH", "./models/Qwen3-ASR-0.6B-ONNX")
ASR_QUANT = os.getenv("ASR_QUANT", "int4")

_asr_model = None


def _load_asr_model():
    """Load ASR model locally via ONNX."""
    global _asr_model
    if _asr_model is not None:
        return _asr_model

    if not os.path.isdir(ASR_MODEL_PATH):
        print(f"ASR model not found at {ASR_MODEL_PATH}, skipping local ASR")
        return None

    try:
        from qwen_asr_onnx import Qwen3ASRONNX
        print(f"Loading Qwen3-ASR-0.6B ONNX from {ASR_MODEL_PATH} (quant={ASR_QUANT})...")
        _asr_model = Qwen3ASRONNX(
            ASR_MODEL_PATH,
            quant=ASR_QUANT if ASR_QUANT != "none" else None,
            max_new_tokens=1024,
        )
        _asr_model.load()
        print("Local ASR model loaded successfully")
        return _asr_model
    except Exception as e:
        print(f"Failed to load local ASR model: {e}")
        return None


def _sanitize_filename(text: str, max_len: int = 15) -> str:
    """Sanitize and truncate text for use in filenames."""
    text = re.sub(r'[^\w\u4e00-\u9fff\-\s]', '', text)
    text = re.sub(r'\s+', '_', text)
    return text[:max_len]


def _transcribe_audio(audio_path: str) -> tuple[str, str]:
    """Transcribe audio file via local ONNX ASR model. Returns (text, status)."""
    if not audio_path or not os.path.isfile(audio_path):
        return "", "未上传音频文件"

    model = _load_asr_model()
    if model is not None:
        try:
            results = model.transcribe(audio=audio_path)
            if not results:
                return "", "识别结果为空"
            text = results[0].text
            detected_lang = getattr(results[0], "language", "unknown")
            status = f"识别成功 (语言: {detected_lang})" if detected_lang else "识别成功"
            return text, status
        except Exception as e:
            return "", f"本地 ASR 识别失败: {e}"
    return "", "ASR 模型未加载，请检查模型路径"


def _save_mp3(wav: np.ndarray, sr: int, prefix: str, text: str) -> str:
    """Save waveform as 320kbps MP3 in outputs/ directory."""
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_text = _sanitize_filename(text)
    filename = f"{prefix}_{timestamp}_{safe_text}.mp3"
    filepath = os.path.join(OUTPUTS_DIR, filename)

    # Use system temp directory to avoid any filesystem permission issues
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = tmp.name

    try:
        sf.write(wav_path, wav, sr)
        subprocess.run(
            ["ffmpeg", "-y", "-i", wav_path, "-b:a", "320k", filepath],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    finally:
        if os.path.exists(wav_path):
            os.remove(wav_path)

    return filepath

# Verify GPU availability at startup
if torch.cuda.is_available():
    print(f"GPU available: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
else:
    print("WARNING: No GPU detected. Inference will be very slow on CPU.")

# Model configurations
MODEL_DIRS = {
    "3.5B": "./models/LongCat-AudioDiT-3.5B-bf16",
    "1B": "./models/LongCat-AudioDiT-1B",
}


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

    def _ensure_model(self, model_key: str) -> str:
        """Ensure model exists locally, download if needed."""
        model_dir = MODEL_DIRS.get(model_key)
        if model_dir is None:
            raise gr.Error(f"模型 {model_key} 配置未找到。")

        if not os.path.isdir(model_dir):
            print(f"模型 {model_key} 未找到，开始自动下载...")
            script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "download_model.py")
            dl_key = model_key.lower().replace("3.5b", "3.5b-bf16").replace("1b", "1b")
            subprocess.run([sys.executable, script, dl_key], check=True)
            if not os.path.isdir(model_dir):
                raise gr.Error(f"模型 {model_key} 下载失败，请手动下载。")

        return model_dir

    def _ensure_tokenizer(self, model_dir: str) -> str:
        """Ensure tokenizer exists locally, download if needed."""
        tokenizer_path = os.path.join(os.path.dirname(model_dir), "umt5-base")
        if os.path.isdir(tokenizer_path):
            return tokenizer_path

        print("Tokenizer 未找到，开始自动下载...")
        script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "download_model.py")
        subprocess.run([sys.executable, script, "umt5"], check=True)
        if not os.path.isdir(tokenizer_path):
            raise gr.Error("Tokenizer 下载失败，请手动运行: python download_model.py umt5")

        return tokenizer_path

    def unload(self) -> None:
        """Unload the current model to free VRAM."""
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

        # Unload current model if switching
        if self.model is not None:
            self.unload()

        model_dir = self._ensure_model(model_key)
        tokenizer_path = self._ensure_tokenizer(model_dir)

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


def get_available_models() -> list[str]:
    available = []
    for key, path in MODEL_DIRS.items():
        if os.path.isdir(path):
            available.append(key)
    return available if available else ["3.5B"]


def generate_tts(
    text: str,
    model_choice: str,
    nfe_steps: int,
    guidance_method: str,
    guidance_strength: float,
    seed: int,
    silence_duration: float,
    max_chars: int,
) -> tuple[tuple[int, np.ndarray] | None, str]:
    if not text or not text.strip():
        raise gr.Error("请输入要合成的文本。")

    try:
        model, tokenizer = model_manager.load(model_choice)
    except gr.Error:
        raise
    except Exception as e:
        raise gr.Error(f"加载模型失败: {e}")

    device = model_manager.device
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    sr = model.config.sampling_rate
    full_hop = model.config.latent_hop
    max_duration = model.config.max_wav_duration

    # Split text by double newlines into paragraphs, then semantically split each paragraph
    paragraphs = text.split("\n\n")
    print(f"\n{'='*60}")
    print(f"[TTS推理] 开始推理 | 文本总长度: {len(text)}字 | 段落数: {len(paragraphs)}")
    print(f"[TTS推理] 静音时长: {silence_duration}s | 最大分割字符数: {max_chars}")
    print(f"{'='*60}")

    wav_segments = []
    segment_count = 0

    # Load polyphone rules once
    polyphone_rules = load_polyphone_rules()

    for para_idx, para in enumerate(paragraphs):
        para = para.strip()
        if not para:
            if silence_duration > 0:
                print(f"\n[段落 {para_idx+1}/{len(paragraphs)}] 空段落, 插入 {silence_duration}s 静音")
                silence_wav = np.zeros(int(sr * silence_duration), dtype=np.float32)
                wav_segments.append(silence_wav)
            continue

        print(f"\n[段落 {para_idx+1}/{len(paragraphs)}] 长度: {len(para)}字")
        chunks = split_text_semantic(para, max_chars=max_chars)
        print(f"  -> 共分割为 {len(chunks)} 个语义块")

        for chunk_idx, chunk in enumerate(chunks):
            # Apply polyphone replacement before normalization
            polyphone_text, polyphone_logs = apply_polyphone_rules(chunk, rules=polyphone_rules)
            for log_msg in polyphone_logs:
                print(f"  {log_msg}")

            normalized_text = normalize_mixed_text(polyphone_text)
            print(f"  [推理 {segment_count+1}] 文本: \"{chunk}\"")
            if polyphone_text != chunk:
                print(f"  [推理 {segment_count+1}] 多音字替换: \"{chunk}\" → \"{polyphone_text}\"")
            print(f"  [推理 {segment_count+1}] 归一化: \"{normalized_text}\"")

            inputs = tokenizer([normalized_text], padding="longest", return_tensors="pt")

            dur_sec = approx_duration_from_text(chunk, max_duration=max_duration)
            duration = int(dur_sec * sr // full_hop)
            duration = min(duration, int(max_duration * sr // full_hop))

            print(f"  [推理 {segment_count+1}] 预估时长: {dur_sec:.2f}s | duration参数: {duration} | 开始推理...")

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
            wav_duration = len(wav) / sr
            wav_segments.append(wav)
            segment_count += 1
            print(f"  [推理 {segment_count}] 完成 | 音频时长: {wav_duration:.2f}s | 采样点数: {len(wav)}")

        if para_idx < len(paragraphs) - 1 and silence_duration > 0:
            print(f"  [段落间] 插入 {silence_duration}s 静音")
            silence_wav = np.zeros(int(sr * silence_duration), dtype=np.float32)
            wav_segments.append(silence_wav)

    if not wav_segments:
        raise gr.Error("没有有效的文本可合成。")

    # Concatenate all segments
    final_wav = np.concatenate(wav_segments)
    total_duration = len(final_wav) / sr

    # Save to MP3
    mp3_path = _save_mp3(final_wav, sr, "tts", text.strip())

    info = (
        f"生成: {total_duration:.2f}秒 | 分段数: {segment_count} | "
        f"模型: {model_choice} | 步数: {nfe_steps} | 保存: {mp3_path}"
    )

    return (sr, final_wav), info


def generate_clone(
    prompt_audio: str | None,
    prompt_text: str,
    target_text: str,
    model_choice: str,
    nfe_steps: int,
    guidance_method: str,
    guidance_strength: float,
    seed: int,
    silence_duration: float,
    max_chars: int,
    country: str,
) -> tuple[tuple[int, np.ndarray] | None, str]:
    if prompt_audio is None:
        raise gr.Error("请上传参考音频文件。")
    if not prompt_text or not prompt_text.strip():
        raise gr.Error("请输入参考音频的文本内容。")
    if not target_text or not target_text.strip():
        raise gr.Error("请输入要合成的文本。")

    try:
        model, tokenizer = model_manager.load(model_choice)
    except gr.Error:
        raise
    except Exception as e:
        raise gr.Error(f"加载模型失败: {e}")

    device = model_manager.device
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    sr = model.config.sampling_rate
    full_hop = model.config.latent_hop
    max_duration = model.config.max_wav_duration

    # Process prompt audio once
    prompt_wav = load_audio(prompt_audio, sr).unsqueeze(0)

    off = 3
    pw = load_audio(prompt_audio, sr)
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

    # Split target text by double newlines into paragraphs, then semantically split each paragraph
    paragraphs = target_text.split("\n\n")
    print(f"\n{'='*60}")
    print(f"[克隆推理] 开始推理 | 目标文本长度: {len(target_text)}字 | 段落数: {len(paragraphs)}")
    print(f"[克隆推理] 参考音频时长: {prompt_time:.2f}s | 参考文本: \"{norm_prompt_text}\"")
    print(f"[克隆推理] 静音时长: {silence_duration}s | 最大分割字符数: {max_chars} | 语言: {country}")
    print(f"{'='*60}")

    wav_segments = []
    segment_count = 0

    # Move prompt_wav to device once for reuse
    prompt_wav_device = prompt_wav.to(device)

    # Load polyphone rules once
    polyphone_rules = load_polyphone_rules()

    for para_idx, para in enumerate(paragraphs):
        para = para.strip()
        if not para:
            if silence_duration > 0:
                print(f"\n[段落 {para_idx+1}/{len(paragraphs)}] 空段落, 插入 {silence_duration}s 静音")
                silence_wav = np.zeros(int(sr * silence_duration), dtype=np.float32)
                wav_segments.append(silence_wav)
            continue

        print(f"\n[段落 {para_idx+1}/{len(paragraphs)}] 长度: {len(para)}字")
        chunks = split_text_semantic(para, max_chars=max_chars)
        print(f"  -> 共分割为 {len(chunks)} 个语义块")

        for chunk_idx, chunk in enumerate(chunks):
            # Apply polyphone replacement before normalization
            polyphone_text, polyphone_logs = apply_polyphone_rules(chunk, rules=polyphone_rules)
            for log_msg in polyphone_logs:
                print(f"  {log_msg}")

            norm_target_text = normalize_mixed_text(polyphone_text, country=country)
            full_text = f"{norm_prompt_text} {norm_target_text}"
            inputs = tokenizer([full_text], padding="longest", return_tensors="pt")

            dur_sec = approx_duration_from_text(chunk, max_duration=max_duration - prompt_time)
            approx_pd = approx_duration_from_text(prompt_text, max_duration=max_duration)
            ratio = np.clip(prompt_time / approx_pd, 1.0, 1.5)
            dur_sec = dur_sec * ratio
            duration = int(dur_sec * sr // full_hop)
            duration = min(duration + prompt_dur, int(max_duration * sr // full_hop))

            print(f"  [推理 {segment_count+1}] 文本: \"{chunk}\"")
            if polyphone_text != chunk:
                print(f"  [推理 {segment_count+1}] 多音字替换: \"{chunk}\" → \"{polyphone_text}\"")
            print(f"  [推理 {segment_count+1}] 归一化: \"{norm_target_text}\"")
            print(f"  [推理 {segment_count+1}] 预估时长: {dur_sec:.2f}s | duration参数: {duration} | 速率比: {ratio:.2f} | 开始推理...")

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
            wav_duration = len(wav) / sr
            wav_segments.append(wav)
            segment_count += 1
            print(f"  [推理 {segment_count}] 完成 | 音频时长: {wav_duration:.2f}s | 采样点数: {len(wav)}")

        if para_idx < len(paragraphs) - 1 and silence_duration > 0:
            print(f"  [段落间] 插入 {silence_duration}s 静音")
            silence_wav = np.zeros(int(sr * silence_duration), dtype=np.float32)
            wav_segments.append(silence_wav)

    if not wav_segments:
        raise gr.Error("没有有效的目标文本可合成。")

    # Concatenate all segments
    final_wav = np.concatenate(wav_segments)
    total_duration = len(final_wav) / sr

    # Save to MP3
    mp3_path = _save_mp3(final_wav, sr, "clone", target_text.strip())

    info = (
        f"生成: {total_duration:.2f}秒 | 分段数: {segment_count} | "
        f"模型: {model_choice} | 步数: {nfe_steps} | 保存: {mp3_path}"
    )

    return (sr, final_wav), info


def save_reference_package(
    prompt_audio: str | None,
    prompt_text: str,
    package_name: str,
) -> str:
    """保存参考音频和文本为 zip 包到 samples 目录。"""
    if prompt_audio is None:
        raise gr.Error("请上传参考音频文件。")
    if not prompt_text or not prompt_text.strip():
        raise gr.Error("请输入参考音频的文本内容。")

    samples_dir = "/app/samples"
    os.makedirs(samples_dir, exist_ok=True)

    # Generate package name if not provided
    if not package_name or not package_name.strip():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        package_name = f"reference_{timestamp}"

    # Clean package name (remove invalid characters)
    package_name = "".join(c for c in package_name if c.isalnum() or c in "_-")
    if not package_name:
        package_name = f"reference_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Create temporary directory for packaging
    temp_dir = os.path.join(samples_dir, f"_temp_{package_name}")
    os.makedirs(temp_dir, exist_ok=True)

    try:
        # Copy reference audio file
        audio_ext = os.path.splitext(prompt_audio)[1] or ".wav"
        audio_dest = os.path.join(temp_dir, f"reference_audio{audio_ext}")
        shutil.copy2(prompt_audio, audio_dest)

        # Save reference text
        text_dest = os.path.join(temp_dir, "reference_text.txt")
        with open(text_dest, "w", encoding="utf-8") as f:
            f.write(prompt_text.strip())

        # Create zip file
        zip_path = os.path.join(samples_dir, f"{package_name}.zip")
        file_exists = os.path.isfile(zip_path)
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, temp_dir)
                    zipf.write(file_path, arcname)

        # Clean up temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)

        action = "更新" if file_exists else "保存"
        return f"{action}成功: {zip_path}"

    except Exception as e:
        # Clean up temp directory on error
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise gr.Error(f"保存参考包失败: {e}")


def list_reference_packages() -> list[str]:
    """列出 samples 目录中所有已保存的参考包。"""
    samples_dir = "/app/samples"
    if not os.path.isdir(samples_dir):
        return []
    
    packages = []
    for filename in os.listdir(samples_dir):
        if filename.endswith(".zip"):
            packages.append(filename[:-4])  # Remove .zip extension
    
    return sorted(packages)


def load_reference_package(package_name: str) -> tuple[str | None, str]:
    """从已保存的包中加载参考音频和文本。"""
    if not package_name:
        raise gr.Error("请选择一个参考包。")
    
    samples_dir = "/app/samples"
    zip_path = os.path.join(samples_dir, f"{package_name}.zip")
    
    if not os.path.isfile(zip_path):
        raise gr.Error(f"参考包未找到: {package_name}")
    
    # Create temporary directory for extraction
    temp_dir = os.path.join(samples_dir, f"_load_temp_{package_name}")
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        # Extract zip file
        with zipfile.ZipFile(zip_path, "r") as zipf:
            zipf.extractall(temp_dir)
        
        # Find audio file (support common extensions)
        audio_file = None
        for ext in [".wav", ".mp3", ".flac", ".ogg", ".m4a"]:
            candidate = os.path.join(temp_dir, f"reference_audio{ext}")
            if os.path.isfile(candidate):
                audio_file = candidate
                break
        
        if audio_file is None:
            raise gr.Error("参考包中未找到音频文件。")
        
        # Read text file
        text_file = os.path.join(temp_dir, "reference_text.txt")
        if not os.path.isfile(text_file):
            raise gr.Error("参考包中未找到文本文件。")
        
        with open(text_file, "r", encoding="utf-8") as f:
            text_content = f.read().strip()
        
        # Copy audio to a persistent location (Gradio needs the file to exist)
        persistent_dir = os.path.join(samples_dir, "_loaded")
        os.makedirs(persistent_dir, exist_ok=True)
        persistent_audio = os.path.join(persistent_dir, f"{package_name}_audio{os.path.splitext(audio_file)[1]}")
        shutil.copy2(audio_file, persistent_audio)
        
        # Clean up temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        return persistent_audio, text_content
        
    except gr.Error:
        raise
    except Exception as e:
        # Clean up temp directory on error
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise gr.Error(f"加载参考包失败: {e}")


def refresh_package_list() -> gr.Dropdown:
    """刷新参考包下拉列表。"""
    packages = list_reference_packages()
    return gr.Dropdown(choices=packages, value=None)


def build_ui() -> gr.Blocks:
    available_models = get_available_models()

    with gr.Blocks(
        title="LongCat-AudioDiT 语音合成",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown("# LongCat-AudioDiT 语音合成 & 声音克隆")

        with gr.Row():
            model_dropdown = gr.Dropdown(
                choices=available_models,
                value=available_models[0],
                label="模型",
                scale=3,
            )
            model_status = gr.Textbox(
                value="未加载模型",
                label="状态",
                interactive=False,
                scale=4,
            )
            load_btn = gr.Button("加载模型", scale=1, variant="primary")
            unload_btn = gr.Button("卸载模型", scale=1, variant="stop")

        def on_load_model(model_key: str) -> str:
            try:
                model_manager.load(model_key)
                return model_manager.get_status()
            except Exception as e:
                return f"错误: {e}"

        def on_unload_model() -> str:
            model_manager.unload()
            return model_manager.get_status()

        load_btn.click(
            fn=lambda: (gr.Button(value="加载中...", interactive=False), gr.Button(interactive=False)),
            outputs=[load_btn, unload_btn],
        ).then(
            fn=on_load_model,
            inputs=[model_dropdown],
            outputs=[model_status],
        ).then(
            fn=lambda: (gr.Button(value="加载模型", interactive=True), gr.Button(interactive=True)),
            outputs=[load_btn, unload_btn],
        )

        unload_btn.click(
            fn=lambda: (gr.Button(interactive=False), gr.Button(value="卸载中...", interactive=False)),
            outputs=[load_btn, unload_btn],
        ).then(
            fn=on_unload_model,
            inputs=[],
            outputs=[model_status],
        ).then(
            fn=lambda: (gr.Button(interactive=True), gr.Button(value="卸载模型", interactive=True)),
            outputs=[load_btn, unload_btn],
        )

        with gr.Tabs():
            with gr.Tab("语音合成"):
                with gr.Row():
                    with gr.Column(scale=2):
                        tts_text = gr.Textbox(
                            label="合成文本",
                            placeholder="请输入要合成的文本...",
                            lines=4,
                        )
                        with gr.Accordion("高级设置", open=False):
                            tts_nfe = gr.Slider(
                                minimum=4, maximum=32, value=16, step=1,
                                label="NFE 步数 (ODE 步数)",
                            )
                            with gr.Row():
                                tts_guidance = gr.Dropdown(
                                    choices=["cfg", "apg"],
                                    value="cfg",
                                    label="引导方法",
                                )
                                tts_strength = gr.Slider(
                                    minimum=1.0, maximum=10.0, value=4.0, step=0.5,
                                    label="引导强度",
                                )
                            tts_seed = gr.Number(value=1024, label="随机种子", precision=0)
                            with gr.Row():
                                tts_silence = gr.Slider(
                                    minimum=0.0, maximum=2.0, value=0.5, step=0.1,
                                    label="段落间静音时长 (秒)",
                                )
                                tts_max_chars = gr.Number(
                                    value=100, label="最大分割字符数", precision=0,
                                )
                        tts_btn = gr.Button("生成", variant="primary")
                    with gr.Column(scale=2):
                        tts_output = gr.Audio(label="输出音频", type="numpy")
                        tts_info = gr.Textbox(label="信息", interactive=False)

                tts_btn.click(
                    fn=lambda: gr.Button(value="生成中...", interactive=False),
                    outputs=[tts_btn],
                ).then(
                    fn=generate_tts,
                    inputs=[tts_text, model_dropdown, tts_nfe, tts_guidance, tts_strength, tts_seed, tts_silence, tts_max_chars],
                    outputs=[tts_output, tts_info],
                ).then(
                    fn=lambda: gr.Button(value="生成", interactive=True),
                    outputs=[tts_btn],
                )

            with gr.Tab("声音克隆"):
                with gr.Row():
                    with gr.Column(scale=1):
                        clone_target_text = gr.Textbox(
                            label="目标文本",
                            placeholder="要合成的文本...",
                            lines=6,
                        )
                        with gr.Accordion("加载已保存的参考", open=True):
                            with gr.Row():
                                package_dropdown = gr.Dropdown(
                                    choices=list_reference_packages(),
                                    label="已保存的参考包",
                                    scale=4,
                                )
                                refresh_btn = gr.Button("刷新", scale=1, variant="secondary")
                            load_btn = gr.Button("加载选中的参考包", variant="secondary")
                        with gr.Accordion("保存参考", open=False):
                            package_name = gr.Textbox(
                                label="参考包名称",
                                placeholder="可选的参考包名称...",
                            )
                            save_btn = gr.Button("保存参考包", variant="secondary")
                            save_info = gr.Textbox(label="保存状态", interactive=False)
                    with gr.Column(scale=1):
                        clone_audio = gr.Audio(
                            label="参考音频",
                            type="filepath",
                        )
                        clone_prompt_text = gr.Textbox(
                            label="参考音频文本",
                            placeholder="参考音频的文字内容 (上传音频后自动识别)...",
                            lines=2,
                        )
                        asr_info = gr.Textbox(
                            label="识别状态",
                            interactive=False,
                        )
                        with gr.Accordion("高级设置", open=False):
                            clone_nfe = gr.Slider(
                                minimum=4, maximum=32, value=16, step=1,
                                label="NFE 步数 (ODE 步数)",
                            )
                            with gr.Row():
                                clone_guidance = gr.Dropdown(
                                    choices=["cfg", "apg"],
                                    value="apg",
                                    label="引导方法",
                                )
                                clone_strength = gr.Slider(
                                    minimum=1.0, maximum=10.0, value=4.0, step=0.5,
                                    label="引导强度",
                                )
                            clone_seed = gr.Number(value=1024, label="随机种子", precision=0)
                            clone_country = gr.Dropdown(
                                choices=["auto", "zh", "en", "ja", "ko", "fr", "de", "es", "ru"],
                                value="auto",
                                label="说话人语言/国家 (自动检测)",
                            )
                            with gr.Row():
                                clone_silence = gr.Slider(
                                    minimum=0.0, maximum=2.0, value=0.5, step=0.1,
                                    label="段落间静音时长 (秒)",
                                )
                                clone_max_chars = gr.Number(
                                    value=100, label="最大分割字符数", precision=0,
                                )
                        clone_btn = gr.Button("克隆声音", variant="primary")
                    with gr.Column(scale=1):
                        clone_output = gr.Audio(label="输出音频", type="numpy")
                        clone_info = gr.Textbox(label="信息", interactive=False)

                clone_btn.click(
                    fn=lambda: gr.Button(value="克隆中...", interactive=False),
                    outputs=[clone_btn],
                ).then(
                    fn=generate_clone,
                    inputs=[
                        clone_audio, clone_prompt_text, clone_target_text,
                        model_dropdown, clone_nfe, clone_guidance, clone_strength, clone_seed,
                        clone_silence, clone_max_chars, clone_country,
                    ],
                    outputs=[clone_output, clone_info],
                ).then(
                    fn=lambda: gr.Button(value="克隆声音", interactive=True),
                    outputs=[clone_btn],
                )

                save_btn.click(
                    save_reference_package,
                    inputs=[clone_audio, clone_prompt_text, package_name],
                    outputs=[save_info],
                )

                refresh_btn.click(
                    refresh_package_list,
                    inputs=[],
                    outputs=[package_dropdown],
                )

                load_btn.click(
                    load_reference_package,
                    inputs=[package_dropdown],
                    outputs=[clone_audio, clone_prompt_text],
                )

                package_dropdown.change(
                    load_reference_package,
                    inputs=[package_dropdown],
                    outputs=[clone_audio, clone_prompt_text],
                )

                def on_audio_change(audio_path: str | None, existing_text: str) -> tuple[str, str]:
                    if audio_path is None:
                        return "", ""
                    if existing_text and existing_text.strip():
                        return existing_text, "已存在参考文本，跳过识别"
                    text, status = _transcribe_audio(audio_path)
                    return text, status

                clone_audio.change(
                    on_audio_change,
                    inputs=[clone_audio, clone_prompt_text],
                    outputs=[clone_prompt_text, asr_info],
                )

            with gr.Tab("多音字规则"):
                with gr.Row():
                    with gr.Column(scale=3):
                        gr.Markdown(
                            "### 多音字替换规则\n\n"
                            "**规则格式说明：**\n\n"
                            "- `[行长] 航长`：无上下文，直接替换整个词\n"
                            "- `银行[行]长 航`：有上下文，只替换括号内的字\n\n"
                            "**示例：**\n\n"
                            "- `[行长] 航长`：匹配到行长就替换为航长\n"
                            "- `银行[行]长 航`：匹配到银行行长时，只将行替换为航，结果为银行航长\n"
                            "- `银[行]卡 航`：匹配到银行卡时，替换为银航卡\n\n"
                            "**优先级：** 规则按匹配长度降序应用，长的（带上下文的）规则优先。\n\n"
                            "**格式：** 每行一条规则，用空格分隔：`模式 替换值`"
                        )
                        polyphone_rules_text = gr.Textbox(
                            label="规则列表",
                            placeholder="每行一条规则，用空格分隔：模式 替换值",
                            lines=20,
                            max_lines=50,
                        )
                        with gr.Row():
                            polyphone_save_btn = gr.Button("保存规则", variant="primary")
                            polyphone_reload_btn = gr.Button("重新加载", variant="secondary")
                            polyphone_reset_btn = gr.Button("重置为默认", variant="stop")
                        polyphone_save_info = gr.Textbox(label="保存状态", interactive=False)
                    with gr.Column(scale=2):
                        gr.Markdown(
                            "### 测试\n\n"
                            "输入文本，查看多音字替换效果："
                        )
                        polyphone_test_input = gr.Textbox(
                            label="测试文本",
                            placeholder="输入包含多音字的文本...",
                            lines=3,
                        )
                        polyphone_test_btn = gr.Button("测试替换")
                        polyphone_test_output = gr.Textbox(
                            label="替换结果",
                            interactive=False,
                            lines=6,
                        )

                def _parse_rules_text(rules_text: str) -> dict[str, str]:
                    rules = {}
                    for line in rules_text.strip().split("\n"):
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split(None, 1)
                        if len(parts) == 2:
                            pattern, replacement = parts[0].strip(), parts[1].strip()
                            if pattern and replacement:
                                rules[pattern] = replacement
                    return rules

                def _format_rules_text(rules: dict[str, str]) -> str:
                    lines = []
                    for pattern, replacement in sorted(rules.items(), key=lambda x: len(x[0]), reverse=True):
                        lines.append(f"{pattern} {replacement}")
                    return "\n".join(lines)

                def _load_rules_to_ui() -> str:
                    rules = load_polyphone_rules()
                    return _format_rules_text(rules)

                def _save_rules_from_ui(rules_text: str) -> str:
                    rules = _parse_rules_text(rules_text)
                    try:
                        save_polyphone_rules(rules)
                        return f"保存成功，共 {len(rules)} 条规则"
                    except Exception as e:
                        return f"保存失败: {e}"

                def _reset_rules() -> tuple[str, str]:
                    try:
                        save_polyphone_rules(dict(_DEFAULT_POLYPHONE_RULES))
                        return _format_rules_text(_DEFAULT_POLYPHONE_RULES), "已重置为默认规则"
                    except Exception as e:
                        return "", f"重置失败: {e}"

                def _test_rules(rules_text: str, test_text: str) -> str:
                    rules = _parse_rules_text(rules_text)
                    if not test_text:
                        return "请输入测试文本"
                    result, logs = apply_polyphone_rules(test_text, rules=rules)
                    output_lines = [f"原文: {test_text}", f"结果: {result}", ""]
                    if logs:
                        output_lines.append("--- 替换日志 ---")
                        output_lines.extend(logs)
                    else:
                        output_lines.append("(无替换)")
                    return "\n".join(output_lines)

                polyphone_rules_text.value = _load_rules_to_ui()
                polyphone_save_btn.click(
                    _save_rules_from_ui,
                    inputs=[polyphone_rules_text],
                    outputs=[polyphone_save_info],
                )
                polyphone_reload_btn.click(
                    _load_rules_to_ui,
                    inputs=[],
                    outputs=[polyphone_rules_text],
                )
                polyphone_reset_btn.click(
                    _reset_rules,
                    inputs=[],
                    outputs=[polyphone_rules_text, polyphone_save_info],
                )
                polyphone_test_btn.click(
                    _test_rules,
                    inputs=[polyphone_rules_text, polyphone_test_input],
                    outputs=[polyphone_test_output],
                )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
    )
