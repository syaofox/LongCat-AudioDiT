#!/usr/bin/env python3
"""Download LongCat-AudioDiT models from HuggingFace to local directory."""

import os
import sys
from huggingface_hub import snapshot_download


MODELS = {
    "1b": "meituan-longcat/LongCat-AudioDiT-1B",
    "3.5b-bf16": "drbaph/LongCat-AudioDiT-3.5B-bf16",
    "umt5": "google/umt5-base",
    "asr-0.6b": "Qwen/Qwen3-ASR-0.6B",
}


def download_model(model_id: str, local_dir: str) -> None:
    print(f"Downloading {model_id} to {local_dir}...")
    os.makedirs(local_dir, exist_ok=True)
    snapshot_download(
        repo_id=model_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
    )
    print(f"Done: {local_dir}")


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python download_model.py 1b          # Download 1B model")
        print("  python download_model.py 3.5b-bf16   # Download 3.5B bf16 model")
        print("  python download_model.py umt5        # Download UMT5 tokenizer/model")
        print("  python download_model.py all          # Download all models")
        print("  python download_model.py <repo_id> <local_dir>  # Custom")
        sys.exit(1)

    target = sys.argv[1].lower()

    if target == "1b":
        download_model(MODELS[target], "./models/LongCat-AudioDiT-1B")
    elif target == "3.5b-bf16":
        download_model(MODELS[target], "./models/LongCat-AudioDiT-3.5B-bf16")
    elif target == "umt5":
        download_model(MODELS[target], "./models/umt5-base")
    elif target == "asr-0.6b":
        download_model(MODELS[target], "./models/Qwen3-ASR-0.6B")
    elif target == "all":
        for name, repo_id in MODELS.items():
            if name == "umt5":
                download_model(repo_id, "./models/umt5-base")
            elif name == "1b":
                download_model(repo_id, "./models/LongCat-AudioDiT-1B")
            elif name == "3.5b-bf16":
                download_model(repo_id, "./models/LongCat-AudioDiT-3.5B-bf16")
            elif name == "asr-0.6b":
                download_model(repo_id, "./models/Qwen3-ASR-0.6B")
    else:
        # Custom repo_id and local_dir
        local_dir = sys.argv[2] if len(sys.argv) > 2 else "./models/custom"
        download_model(target, local_dir)


if __name__ == "__main__":
    main()