"""Qwen3-ASR ONNX inference engine."""

import gc
import json
import os

import numpy as np
import onnxruntime as ort
import soundfile as sf
from transformers import AutoTokenizer

IM_START_TOKEN_ID = 151644
IM_END_TOKEN_ID = 151645
AUDIO_START_TOKEN_ID = 151669
AUDIO_END_TOKEN_ID = 151670
AUDIO_PAD_TOKEN_ID = 151676
ENDOFTEXT_TOKEN_ID = 151643
NEWLINE_TOKEN_ID = 198
EOS_TOKEN_IDS = [ENDOFTEXT_TOKEN_ID, IM_END_TOKEN_ID]


def _conv_out_len(t: int) -> int:
    return (t + 1) // 2


def get_feat_extract_output_lengths(input_lengths: int) -> int:
    CONV_WINDOW = 100
    TOKENS_PER_WINDOW = 13
    leave = input_lengths % CONV_WINDOW
    t = _conv_out_len(leave)
    t = _conv_out_len(t)
    t = _conv_out_len(t)
    return t + (input_lengths // CONV_WINDOW) * TOKENS_PER_WINDOW


def log_mel_spectrogram(audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
    """Compute log-mel spectrogram [1, 128, T] from 16kHz mono audio."""
    import librosa
    import torch

    assert sample_rate == 16000, f"Expected 16kHz audio, got {sample_rate}Hz"
    audio = audio.astype(np.float32)

    mel_filters = librosa.filters.mel(
        sr=sample_rate, n_fft=400, n_mels=128, fmin=0.0, fmax=8000.0, norm="slaney"
    )
    mel_filters_t = torch.from_numpy(mel_filters).float()

    window = torch.hann_window(400)
    audio_tensor = torch.from_numpy(audio).float()
    stft = torch.stft(audio_tensor, n_fft=400, hop_length=160, window=window, return_complex=True)
    magnitudes = stft.abs() ** 2

    mel_spec = mel_filters_t @ magnitudes
    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    log_spec = log_spec[:, :-1]

    return log_spec.unsqueeze(0).numpy()


def build_prompt_ids(audio_token_count: int) -> list[int]:
    ids = [
        IM_START_TOKEN_ID, 9125, NEWLINE_TOKEN_ID, IM_END_TOKEN_ID, NEWLINE_TOKEN_ID,
        IM_START_TOKEN_ID, 882, NEWLINE_TOKEN_ID, AUDIO_START_TOKEN_ID,
    ]
    ids.extend([AUDIO_PAD_TOKEN_ID] * audio_token_count)
    ids.extend([AUDIO_END_TOKEN_ID, IM_END_TOKEN_ID, NEWLINE_TOKEN_ID])
    ids.extend([IM_START_TOKEN_ID, 77091, NEWLINE_TOKEN_ID])
    return ids


def get_audio_pad_range(prompt_ids: list[int]) -> tuple[int, int]:
    start = end = None
    for i, tid in enumerate(prompt_ids):
        if tid == AUDIO_PAD_TOKEN_ID:
            if start is None:
                start = i
            end = i + 1
    if start is None:
        raise ValueError("No audio_pad tokens found in prompt")
    return start, end


def _resolve_model_path(model_dir: str, name: str, quant: str | None = None) -> str:
    if quant:
        path = os.path.join(model_dir, f"{name}.{quant}.onnx")
        if os.path.exists(path):
            return path
    path = os.path.join(model_dir, f"{name}.onnx")
    if os.path.exists(path):
        return path
    raise FileNotFoundError(f"No model found for {name} (quant={quant}) in {model_dir}")


def load_embed_tokens(model_dir: str) -> np.ndarray:
    with open(os.path.join(model_dir, "config.json")) as f:
        cfg = json.load(f)

    if "embed_tokens_shape" in cfg:
        shape = cfg["embed_tokens_shape"]
    else:
        shape = [cfg["decoder"]["vocab_size"], cfg["decoder"]["hidden_size"]]

    dtype_str = cfg.get("embed_tokens_dtype", "float32")
    dtype = np.float16 if dtype_str == "float16" else np.float32

    embed = np.fromfile(os.path.join(model_dir, "embed_tokens.bin"), dtype=dtype).reshape(shape)
    return embed.astype(np.float32)


class TranscriptionResult:
    def __init__(self, text: str, language: str):
        self.text = text
        self.language = language


class Qwen3ASRONNX:
    """ONNX-based Qwen3-ASR inference engine with low memory footprint."""

    def __init__(
        self,
        model_dir: str,
        quant: str | None = "int4",
        max_new_tokens: int = 1024,
    ):
        self.model_dir = model_dir
        self.quant = quant
        self.max_new_tokens = max_new_tokens
        self.sessions: dict[str, ort.InferenceSession] = {}
        self.embed_tokens: np.ndarray | None = None
        self.tokenizer: AutoTokenizer | None = None

    def load(self):
        opts = ort.SessionOptions()
        opts.log_severity_level = 3

        # int4 models are optimized for CPU inference, no GPU needed
        providers = ["CPUExecutionProvider"]

        print("Loading ONNX models on CPU (int4 is optimized for CPU inference)")

        for name in ["encoder", "decoder_init", "decoder_step"]:
            path = _resolve_model_path(self.model_dir, name, self.quant)
            self.sessions[name] = ort.InferenceSession(path, opts, providers=providers)
            actual = self.sessions[name].get_providers()
            print(f"  {name}: using {actual[0]}")

        self.embed_tokens = load_embed_tokens(self.model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, trust_remote_code=True)

    def _load_audio(self, audio_path: str) -> np.ndarray:
        audio, sr = sf.read(audio_path, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        return audio

    def _greedy_decode(
        self,
        audio_features: np.ndarray,
        prompt_ids: list[int],
    ) -> list[int]:
        position_ids = np.arange(len(prompt_ids), dtype=np.int64)[np.newaxis, :]
        init_input_names = {inp.name for inp in self.sessions["decoder_init"].get_inputs()}

        if "input_ids" in init_input_names:
            audio_start, _ = get_audio_pad_range(prompt_ids)
            input_ids = np.array(prompt_ids, dtype=np.int64)[np.newaxis, :]
            audio_offset = np.array([audio_start], dtype=np.int64)

            logits, present_keys, present_values = self.sessions["decoder_init"].run(
                ["logits", "present_keys", "present_values"],
                {
                    "input_ids": input_ids,
                    "position_ids": position_ids,
                    "audio_features": audio_features,
                    "audio_offset": audio_offset,
                },
            )
        else:
            input_embeds = self.embed_tokens[prompt_ids].copy()
            audio_start, audio_end = get_audio_pad_range(prompt_ids)
            audio_len = audio_end - audio_start
            if audio_features.shape[1] != audio_len:
                raise ValueError(
                    f"Audio feature length mismatch: expected {audio_len}, got {audio_features.shape[1]}"
                )
            input_embeds[audio_start:audio_end] = audio_features[0]
            input_embeds = input_embeds[np.newaxis, :, :]

            logits, present_keys, present_values = self.sessions["decoder_init"].run(
                ["logits", "present_keys", "present_values"],
                {"input_embeds": input_embeds, "position_ids": position_ids},
            )

        next_token = int(np.argmax(logits[0, -1, :]))
        output_tokens = [next_token]

        if next_token in EOS_TOKEN_IDS:
            return output_tokens

        pos = len(prompt_ids)
        for i in range(self.max_new_tokens - 1):
            token_embed = self.embed_tokens[next_token][np.newaxis, np.newaxis, :]
            step_pos = np.array([[pos]], dtype=np.int64)

            new_logits, new_keys, new_values = self.sessions["decoder_step"].run(
                ["logits", "present_keys", "present_values"],
                {
                    "input_embeds": token_embed,
                    "position_ids": step_pos,
                    "past_keys": present_keys,
                    "past_values": present_values,
                },
            )

            del present_keys, present_values, logits
            present_keys, present_values, logits = new_keys, new_values, new_logits

            next_token = int(np.argmax(logits[0, -1, :]))
            output_tokens.append(next_token)
            pos += 1

            if next_token in EOS_TOKEN_IDS:
                break

            if i % 50 == 0:
                gc.collect()

        return output_tokens

    def transcribe(self, audio: str, language: str | None = None) -> list[TranscriptionResult]:
        audio_np = self._load_audio(audio)
        mel = log_mel_spectrogram(audio_np)

        audio_features = self.sessions["encoder"].run(["audio_features"], {"mel": mel})[0]
        audio_token_count = audio_features.shape[1]

        if audio_token_count == 0:
            del mel, audio_np
            gc.collect()
            return []

        prompt_ids = build_prompt_ids(audio_token_count)
        output_tokens = self._greedy_decode(audio_features, prompt_ids)

        del audio_features, mel, audio_np, prompt_ids
        gc.collect()

        while output_tokens and output_tokens[-1] in EOS_TOKEN_IDS:
            output_tokens.pop()

        text = self.tokenizer.decode(output_tokens, skip_special_tokens=True)
        text = self._clean_asr_output(text)
        return [TranscriptionResult(text, language or "auto")]

    @staticmethod
    def _clean_asr_output(text: str) -> str:
        if "<asr_text>" in text:
            text = text.split("<asr_text>", 1)[1]
        if "</asr_text>" in text:
            text = text.split("</asr_text>", 1)[0]
        text = text.strip()
        if text.lower().startswith("language"):
            parts = text.split("\n", 1)
            if len(parts) > 1:
                text = parts[1]
            else:
                idx = text.find(" ")
                if idx != -1:
                    second_space = text.find(" ", idx + 1)
                    if second_space != -1:
                        text = text[second_space + 1:]
                    else:
                        text = ""
        return text.strip()
