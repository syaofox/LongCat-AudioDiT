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
from utils import normalize_text, load_audio, approx_duration_from_text

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
            raise gr.Error(f"Model {model_key} not found at {model_dir}. Please download it first.")

        print(f"Loading model {model_key} from {model_dir}...")
        self.model = AudioDiTModel.from_pretrained(model_dir).to(self.device)
        self.model.vae.to_half()
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model.config.text_encoder_model)
        self.current_model_key = model_key
        print(f"Model {model_key} loaded.")

        return self.model, self.tokenizer

    def get_status(self) -> str:
        if self.current_model_key:
            return f"Model: {self.current_model_key} (loaded)"
        return "No model loaded"


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
) -> tuple[tuple[int, np.ndarray] | None, str]:
    if not text or not text.strip():
        raise gr.Error("Please enter text to synthesize.")

    try:
        model, tokenizer = model_manager.load(model_choice)
    except gr.Error:
        raise
    except Exception as e:
        raise gr.Error(f"Failed to load model: {e}")

    device = model_manager.device
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    sr = model.config.sampling_rate
    full_hop = model.config.latent_hop
    max_duration = model.config.max_wav_duration

    # Split text by newlines (keep empty lines for silence insertion)
    lines = text.split("\n")
    
    wav_segments = []
    segment_count = 0
    silence_duration = 0.5  # 500ms silence for empty lines

    for line in lines:
        line = line.strip()
        if not line:
            # Insert 500ms silence for empty lines
            silence_wav = np.zeros(int(sr * silence_duration), dtype=np.float32)
            wav_segments.append(silence_wav)
        else:
            normalized_text = normalize_text(line)
            inputs = tokenizer([normalized_text], padding="longest", return_tensors="pt")

            dur_sec = approx_duration_from_text(line, max_duration=max_duration)
            duration = int(dur_sec * sr // full_hop)
            duration = min(duration, int(max_duration * sr // full_hop))

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

    if not wav_segments:
        raise gr.Error("No valid text to synthesize.")

    # Concatenate all segments
    final_wav = np.concatenate(wav_segments)
    total_duration = len(final_wav) / sr
    info = (
        f"Generated: {total_duration:.2f}s | Segments: {segment_count} | "
        f"Model: {model_choice} | Steps: {nfe_steps}"
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
) -> tuple[tuple[int, np.ndarray] | None, str]:
    if prompt_audio is None:
        raise gr.Error("Please upload a reference audio file.")
    if not prompt_text or not prompt_text.strip():
        raise gr.Error("Please enter the reference audio text.")
    if not target_text or not target_text.strip():
        raise gr.Error("Please enter the target text to synthesize.")

    try:
        model, tokenizer = model_manager.load(model_choice)
    except gr.Error:
        raise
    except Exception as e:
        raise gr.Error(f"Failed to load model: {e}")

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

    # Split target text by newlines (keep empty lines for silence insertion)
    lines = target_text.split("\n")
    
    wav_segments = []
    segment_count = 0
    silence_duration = 0.5  # 500ms silence for empty lines

    # Move prompt_wav to device once for reuse
    prompt_wav_device = prompt_wav.to(device)
    norm_prompt_text = normalize_text(prompt_text)

    for line in lines:
        line = line.strip()
        if not line:
            # Insert 500ms silence for empty lines
            silence_wav = np.zeros(int(sr * silence_duration), dtype=np.float32)
            wav_segments.append(silence_wav)
        else:
            norm_target_text = normalize_text(line)
            full_text = f"{norm_prompt_text} {norm_target_text}"
            inputs = tokenizer([full_text], padding="longest", return_tensors="pt")

            dur_sec = approx_duration_from_text(line, max_duration=max_duration - prompt_time)
            approx_pd = approx_duration_from_text(prompt_text, max_duration=max_duration)
            ratio = np.clip(prompt_time / approx_pd, 1.0, 1.5)
            dur_sec = dur_sec * ratio
            duration = int(dur_sec * sr // full_hop)
            duration = min(duration + prompt_dur, int(max_duration * sr // full_hop))

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

    if not wav_segments:
        raise gr.Error("No valid target text to synthesize.")

    # Concatenate all segments
    final_wav = np.concatenate(wav_segments)
    total_duration = len(final_wav) / sr
    info = (
        f"Generated: {total_duration:.2f}s | Segments: {segment_count} | "
        f"Model: {model_choice} | Steps: {nfe_steps}"
    )

    return (sr, final_wav), info


def save_reference_package(
    prompt_audio: str | None,
    prompt_text: str,
    package_name: str,
) -> str:
    """Save reference audio and text as a zip package in samples directory."""
    if prompt_audio is None:
        raise gr.Error("Please upload a reference audio file.")
    if not prompt_text or not prompt_text.strip():
        raise gr.Error("Please enter the reference audio text.")

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
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, temp_dir)
                    zipf.write(file_path, arcname)

        # Clean up temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)

        return f"Saved to: {zip_path}"

    except Exception as e:
        # Clean up temp directory on error
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise gr.Error(f"Failed to save package: {e}")


def list_reference_packages() -> list[str]:
    """List all saved reference packages in samples directory."""
    samples_dir = "/app/samples"
    if not os.path.isdir(samples_dir):
        return []
    
    packages = []
    for filename in os.listdir(samples_dir):
        if filename.endswith(".zip"):
            packages.append(filename[:-4])  # Remove .zip extension
    
    return sorted(packages)


def load_reference_package(package_name: str) -> tuple[str | None, str]:
    """Load reference audio and text from a saved package."""
    if not package_name:
        raise gr.Error("Please select a reference package.")
    
    samples_dir = "/app/samples"
    zip_path = os.path.join(samples_dir, f"{package_name}.zip")
    
    if not os.path.isfile(zip_path):
        raise gr.Error(f"Package not found: {package_name}")
    
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
            raise gr.Error("No audio file found in package.")
        
        # Read text file
        text_file = os.path.join(temp_dir, "reference_text.txt")
        if not os.path.isfile(text_file):
            raise gr.Error("No text file found in package.")
        
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
        raise gr.Error(f"Failed to load package: {e}")


def refresh_package_list() -> gr.Dropdown:
    """Refresh the dropdown list of reference packages."""
    packages = list_reference_packages()
    return gr.Dropdown(choices=packages, value=None)


def build_ui() -> gr.Blocks:
    available_models = get_available_models()

    with gr.Blocks(
        title="LongCat-AudioDiT TTS",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown("# LongCat-AudioDiT TTS & Voice Cloning")

        with gr.Row():
            model_dropdown = gr.Dropdown(
                choices=available_models,
                value=available_models[0],
                label="Model",
                scale=3,
            )
            model_status = gr.Textbox(
                value="No model loaded",
                label="Status",
                interactive=False,
                scale=4,
            )
            load_btn = gr.Button("Load Model", scale=1, variant="primary")

        def on_load_model(model_key: str) -> str:
            try:
                model_manager.load(model_key)
                return model_manager.get_status()
            except Exception as e:
                return f"Error: {e}"

        load_btn.click(on_load_model, inputs=[model_dropdown], outputs=[model_status])

        with gr.Tabs():
            with gr.Tab("TTS Synthesis"):
                with gr.Row():
                    with gr.Column(scale=2):
                        tts_text = gr.Textbox(
                            label="Text to synthesize",
                            placeholder="Enter text here...",
                            lines=4,
                        )
                        with gr.Accordion("Advanced Settings", open=False):
                            tts_nfe = gr.Slider(
                                minimum=4, maximum=32, value=16, step=1,
                                label="NFE Steps (ODE steps)",
                            )
                            with gr.Row():
                                tts_guidance = gr.Dropdown(
                                    choices=["cfg", "apg"],
                                    value="cfg",
                                    label="Guidance Method",
                                )
                                tts_strength = gr.Slider(
                                    minimum=1.0, maximum=10.0, value=4.0, step=0.5,
                                    label="Guidance Strength",
                                )
                            tts_seed = gr.Number(value=1024, label="Seed", precision=0)
                        tts_btn = gr.Button("Generate", variant="primary")
                    with gr.Column(scale=2):
                        tts_output = gr.Audio(label="Output Audio", type="numpy")
                        tts_info = gr.Textbox(label="Info", interactive=False)

                tts_btn.click(
                    generate_tts,
                    inputs=[tts_text, model_dropdown, tts_nfe, tts_guidance, tts_strength, tts_seed],
                    outputs=[tts_output, tts_info],
                )

            with gr.Tab("Voice Cloning"):
                with gr.Row():
                    with gr.Column(scale=2):
                        with gr.Accordion("Load Saved Reference", open=True):
                            with gr.Row():
                                package_dropdown = gr.Dropdown(
                                    choices=list_reference_packages(),
                                    label="Saved Reference Packages",
                                    scale=4,
                                )
                                refresh_btn = gr.Button("Refresh", scale=1, variant="secondary")
                            load_btn = gr.Button("Load Selected Package", variant="secondary")
                            load_info = gr.Textbox(label="Load Status", interactive=False)
                        clone_audio = gr.Audio(
                            label="Reference Audio",
                            type="filepath",
                        )
                        clone_prompt_text = gr.Textbox(
                            label="Reference Audio Text",
                            placeholder="Text content of the reference audio...",
                            lines=2,
                        )
                        clone_target_text = gr.Textbox(
                            label="Target Text",
                            placeholder="Text to synthesize in the cloned voice...",
                            lines=3,
                        )
                        with gr.Accordion("Advanced Settings", open=False):
                            clone_nfe = gr.Slider(
                                minimum=4, maximum=32, value=16, step=1,
                                label="NFE Steps (ODE steps)",
                            )
                            with gr.Row():
                                clone_guidance = gr.Dropdown(
                                    choices=["cfg", "apg"],
                                    value="apg",
                                    label="Guidance Method",
                                )
                                clone_strength = gr.Slider(
                                    minimum=1.0, maximum=10.0, value=4.0, step=0.5,
                                    label="Guidance Strength",
                                )
                            clone_seed = gr.Number(value=1024, label="Seed", precision=0)
                        clone_btn = gr.Button("Clone Voice", variant="primary")
                        with gr.Accordion("Save Reference", open=False):
                            package_name = gr.Textbox(
                                label="Package Name",
                                placeholder="Optional name for the reference package...",
                            )
                            save_btn = gr.Button("Save Reference Package", variant="secondary")
                            save_info = gr.Textbox(label="Save Status", interactive=False)
                    with gr.Column(scale=2):
                        clone_output = gr.Audio(label="Output Audio", type="numpy")
                        clone_info = gr.Textbox(label="Info", interactive=False)

                clone_btn.click(
                    generate_clone,
                    inputs=[
                        clone_audio, clone_prompt_text, clone_target_text,
                        model_dropdown, clone_nfe, clone_guidance, clone_strength, clone_seed,
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
