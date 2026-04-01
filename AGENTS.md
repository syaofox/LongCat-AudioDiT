# AGENTS.md — Guidelines for Coding Agents

## Project Overview

LongCat-AudioDiT is a conditional-flow-matching TTS model with a DiT transformer backbone, UMT5 text encoder, and WAV-VAE audio autoencoder. The codebase is a HuggingFace-compatible inference library (no training code).

## Repository Structure

```
├── audiodit/                   # Core package (HF-compatible model definitions)
│   ├── __init__.py             # Auto-registers with transformers AutoConfig/AutoModel
│   ├── configuration_audiodit.py  # AudioDiTConfig, AudioDiTVaeConfig
│   └── modeling_audiodit.py    # AudioDiTModel, AudioDiTTransformer, AudioDiTVae
├── inference.py                # Single-sample CLI inference
├── batch_inference.py          # Batch inference (SeedTTS eval format)
├── utils.py                    # Shared helpers: normalize_text, load_audio, approx_duration_from_text
├── requirements.txt            # Dependencies
└── assets/                     # Prompt audio and images
```

## Build / Install / Run

```bash
# Install dependencies
pip install -r requirements.txt

# Single-sample inference (TTS)
python inference.py --text "hello world" --output_audio output.wav --model_dir meituan-longcat/LongCat-AudioDiT-1B

# Single-sample inference (voice cloning)
python inference.py --text "要合成的文本" --prompt_text "参考文本" --prompt_audio prompt.wav --output_audio out.wav --model_dir meituan-longcat/LongCat-AudioDiT-1B --guidance_method apg

# Batch inference
python batch_inference.py --lst /path/to/meta.lst --output_dir /path/to/output --model_dir meituan-longcat/LongCat-AudioDiT-1B
```

There is **no test suite, linter, or type-checking** configured in this repo. If you add tests, use `pytest`:

```bash
pytest tests/              # all tests
pytest tests/test_foo.py -k test_name  # single test
```

## Code Style

### Python Version
- Python 3.10+ (uses `int | None`, `list[int]`, `tuple[...]` union syntax throughout)

### Imports
- Standard library first, then third-party, then local — separated by blank lines.
- Relative imports within the `audiodit/` package (e.g., `from .configuration_audiodit import ...`).
- Top-level scripts use absolute imports (`from audiodit import ...`, `from utils import ...`).
- The `import audiodit` line in scripts triggers HuggingFace AutoConfig/AutoModel registration as a side effect — preserve it.

### Types & Annotations
- Use modern union syntax: `int | None` not `Optional[int]`.
- Use `list[int]`, `dict[str, ...]`, `tuple[torch.Tensor, ...]` not `List`, `Dict`, `Tuple`.
- `torch.Tensor` type annotations on `forward()` signatures.
- Return types on all public functions and methods.
- Config classes inherit from `PreTrainedConfig`; model classes inherit from `PreTrainedModel`.

### Naming Conventions
- Classes: `AudioDiT` prefix, PascalCase (e.g., `AudioDiTModel`, `AudioDiTVaeEncoder`).
- Private/helper classes/modules: leading underscore (e.g., `_MomentumBuffer`, `_project`, `_VaeEncoderBlock`, `_wn_conv1d`).
- Config parameter names: `dit_` prefix for transformer params (e.g., `dit_dim`, `dit_depth`).
- Constants: UPPER_SNAKE_CASE (e.g., `EN_DUR_PER_CHAR`).
- Config class attributes: `model_type` string must match the HF model type.

### Formatting
- 4-space indentation, no trailing whitespace.
- Max line length ~120 chars (no strict enforcement in repo — be reasonable).
- Docstrings: Google style with `Args:` / `Returns:` sections. Keep concise.
- Use `@dataclass` for output types, inheriting from `ModelOutput`.

### Error Handling
- Use `raise ValueError(...)` for invalid config values.
- No bare `except:` — use specific exception types.
- In batch inference, `except Exception as e` with `print` is acceptable for fault tolerance.

### PyTorch Conventions
- Use `torch.no_grad()` context manager or `@torch.no_grad()` decorator for inference.
- Use `F.scaled_dot_product_attention` for attention (not manual softmax).
- Use `nn.utils.weight_norm` for conv layers in the VAE (prefixed `_wn_conv1d`).
- Lazy initialization for buffers that could be corrupted by meta-device construction (see `AudioDiTRotaryEmbedding`).
- Preserve numerical compatibility: VAE runs in `float16` via `model.vae.to_half()`.

### HuggingFace Integration
- Model configs use `model_type = "audiodit"` / `"audiodit_vae"`.
- `sub_configs` dict links nested configs (VAE, text encoder).
- Registration in `__init__.py`: `AutoConfig.register(...)` and `AutoModel.register(...)`.
- `__all__` lists all exported names.

### Key Design Patterns
- Functional helpers at module level: `odeint_euler`, `lens_to_mask`, `_modulate`, `_rotate_half`, `_apply_rotary_emb`.
- Single-file monolith: all modules (DiT, VAE, attention, APG) in `modeling_audiodit.py`.
- No global state or singletons — everything flows through `config`.
