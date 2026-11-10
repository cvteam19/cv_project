"""
Download pre-trained weights for Faster R-CNN and YOLOv8s.
"""
import sys
from pathlib import Path

WEIGHTS = {
    "faster_rcnn_best.pth": "https://huggingface.co/MariKarrp/cvteam19-fire-smoke-detection/resolve/main/faster_rcnn_best.pth",
    "selected_best.pt": "https://huggingface.co/MariKarrp/cvteam19-fire-smoke-detection/resolve/main/selected_best.pt",
}

CKPT_DIR = Path("checkpoints")

def download_from_hf(url: str, dest: Path):
    try:
        from huggingface_hub import hf_hub_download
        parts = url.split("/")
        repo_id = f"{parts[3]}/{parts[4]}"
        filename = parts[-1]
        print(f"Downloading {filename} from Hugging Face...")
        path = hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=str(dest))
        import shutil
        shutil.copy(path, dest / filename)
    except ImportError:
        print("Install huggingface_hub: pip install huggingface_hub", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Failed to download from HF: {e}", file=sys.stderr)
        sys.exit(1)

def download_from_gdrive(url: str, dest: Path):
    try:
        import gdown
        gdown.download(url, str(dest), quiet=False)
    except ImportError:
        print("Install gdown: pip install gdown", file=sys.stderr)
        sys.exit(1)

def main():
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    for filename, url in WEIGHTS.items():
        dest = CKPT_DIR / filename
        if dest.exists():
            print(f"{filename} already exists, skipping.")
            continue
        
        if "huggingface.co" in url:
            download_from_hf(url, CKPT_DIR)
        elif "drive.google.com" in url:
            download_from_gdrive(url, dest)
        else:
            print(f"Unknown source for {filename}")
            continue

    print("\nAll weights are ready in checkpoints/")

if __name__ == "__main__":
    main()