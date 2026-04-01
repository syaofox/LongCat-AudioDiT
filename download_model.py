#!/usr/bin/env python3
"""Download LongCat-AudioDiT models from HuggingFace to local directory."""

import os
import sys
from huggingface_hub import snapshot_download


MODELS = {
    "1b": "meituan-longcat/LongCat-AudioDiT-1B",
    "3.5b": "meituan-longcat/LongCat-AudioDiT-3.5B",
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
        print("  python download_model.py 3.5b        # Download 3.5B model")
        print("  python download_model.py all          # Download both models")
        print("  python download_model.py <repo_id> <local_dir>  # Custom")
        sys.exit(1)

    target = sys.argv[1].lower()

    if target in MODELS:
        download_model(MODELS[target], f"./models/LongCat-AudioDiT-{target.upper()}")
    elif target == "all":
        for name, repo_id in MODELS.items():
            download_model(repo_id, f"./models/LongCat-AudioDiT-{name.upper()}")
    else:
        # Custom repo_id and local_dir
        local_dir = sys.argv[2] if len(sys.argv) > 2 else "./models/custom"
        download_model(target, local_dir)


if __name__ == "__main__":
    main()