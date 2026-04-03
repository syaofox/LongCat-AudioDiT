"""LongCat-AudioDiT Gradio WebUI for TTS and Voice Cloning."""

import os
import shutil
import zipfile
from datetime import datetime
import numpy as np
import torch
import torch.nn.functional as F
import soundfile as sf
import gradio as gr

import audiodit  # auto-registers AudioDiTConfig/AudioDiTModel
from audiodit import AudioDiTModel
from transformers import AutoTokenizer
from utils import normalize_text, normalize_mixed_text, load_audio, approx_duration_from_text, split_text_semantic

torch.backends.cudnn.benchmark = False

# Verify GPU availability at startup
if torch.cuda.is_available():
    print(f"GPU available: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
else:
    print("WARNING: No GPU detected. Inference will be very slow on CPU.")

# Model configurations
MODEL_DIRS = {
    "1B": "/models/LongCat-AudioDiT-1B",
    "3.5B": "/models/LongCat-AudioDiT-3.5B",
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

    def load(self, model_key: str) -> tuple[AudioDiTModel, AutoTokenizer]:
        if model_key == self.current_model_key and self.model is not None:
            return self.model, self.tokenizer

        model_dir = MODEL_DIRS.get(model_key)
        if model_dir is None or not os.path.isdir(model_dir):
            raise gr.Error(f"模型 {model_key} 未找到，请先下载。")

        print(f"正在加载模型 {model_key}，路径: {model_dir}...")
        self.model = AudioDiTModel.from_pretrained(model_dir).to(self.device)
        self.model.vae.to_half()
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model.config.text_encoder_model)
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
    return available if available else ["1B"]


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

    for para_idx, para in enumerate(paragraphs):
        para = para.strip()
        if not para:
            print(f"\n[段落 {para_idx+1}/{len(paragraphs)}] 空段落, 插入 {silence_duration}s 静音")
            silence_wav = np.zeros(int(sr * silence_duration), dtype=np.float32)
            wav_segments.append(silence_wav)
            continue

        print(f"\n[段落 {para_idx+1}/{len(paragraphs)}] 长度: {len(para)}字")
        chunks = split_text_semantic(para, max_chars=max_chars)
        print(f"  -> 共分割为 {len(chunks)} 个语义块")

        for chunk_idx, chunk in enumerate(chunks):
            normalized_text = normalize_mixed_text(chunk)
            print(f"  [推理 {segment_count+1}] 文本: \"{chunk}\"")
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

        if para_idx < len(paragraphs) - 1:
            print(f"  [段落间] 插入 {silence_duration}s 静音")
            silence_wav = np.zeros(int(sr * silence_duration), dtype=np.float32)
            wav_segments.append(silence_wav)

    if not wav_segments:
        raise gr.Error("没有有效的文本可合成。")

    # Concatenate all segments
    final_wav = np.concatenate(wav_segments)
    total_duration = len(final_wav) / sr
    info = (
        f"生成: {total_duration:.2f}秒 | 分段数: {segment_count} | "
        f"模型: {model_choice} | 步数: {nfe_steps}"
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
    print(f"[克隆推理] 静音时长: {silence_duration}s | 最大分割字符数: {max_chars}")
    print(f"{'='*60}")

    wav_segments = []
    segment_count = 0

    # Move prompt_wav to device once for reuse
    prompt_wav_device = prompt_wav.to(device)

    for para_idx, para in enumerate(paragraphs):
        para = para.strip()
        if not para:
            print(f"\n[段落 {para_idx+1}/{len(paragraphs)}] 空段落, 插入 {silence_duration}s 静音")
            silence_wav = np.zeros(int(sr * silence_duration), dtype=np.float32)
            wav_segments.append(silence_wav)
            continue

        print(f"\n[段落 {para_idx+1}/{len(paragraphs)}] 长度: {len(para)}字")
        chunks = split_text_semantic(para, max_chars=max_chars)
        print(f"  -> 共分割为 {len(chunks)} 个语义块")

        for chunk_idx, chunk in enumerate(chunks):
            norm_target_text = normalize_mixed_text(chunk)
            full_text = f"{norm_prompt_text} {norm_target_text}"
            inputs = tokenizer([full_text], padding="longest", return_tensors="pt")

            dur_sec = approx_duration_from_text(chunk, max_duration=max_duration - prompt_time)
            approx_pd = approx_duration_from_text(prompt_text, max_duration=max_duration)
            ratio = np.clip(prompt_time / approx_pd, 1.0, 1.5)
            dur_sec = dur_sec * ratio
            duration = int(dur_sec * sr // full_hop)
            duration = min(duration + prompt_dur, int(max_duration * sr // full_hop))

            print(f"  [推理 {segment_count+1}] 文本: \"{chunk}\"")
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

        if para_idx < len(paragraphs) - 1:
            print(f"  [段落间] 插入 {silence_duration}s 静音")
            silence_wav = np.zeros(int(sr * silence_duration), dtype=np.float32)
            wav_segments.append(silence_wav)

    if not wav_segments:
        raise gr.Error("没有有效的目标文本可合成。")

    # Concatenate all segments
    final_wav = np.concatenate(wav_segments)
    total_duration = len(final_wav) / sr
    info = (
        f"生成: {total_duration:.2f}秒 | 分段数: {segment_count} | "
        f"模型: {model_choice} | 步数: {nfe_steps}"
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

        def on_load_model(model_key: str) -> str:
            try:
                model_manager.load(model_key)
                return model_manager.get_status()
            except Exception as e:
                return f"错误: {e}"

        load_btn.click(on_load_model, inputs=[model_dropdown], outputs=[model_status])

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
                    generate_tts,
                    inputs=[tts_text, model_dropdown, tts_nfe, tts_guidance, tts_strength, tts_seed, tts_silence, tts_max_chars],
                    outputs=[tts_output, tts_info],
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
                            placeholder="参考音频的文字内容...",
                            lines=2,
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
                    generate_clone,
                    inputs=[
                        clone_audio, clone_prompt_text, clone_target_text,
                        model_dropdown, clone_nfe, clone_guidance, clone_strength, clone_seed,
                        clone_silence, clone_max_chars,
                    ],
                    outputs=[clone_output, clone_info],
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

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
    )
