"""
Download and prepare the D-Fire dataset.
"""
import argparse
import shutil
import sys
from pathlib import Path

DATASET_ID = "linxel/dfire-prepared"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data", type=Path,
                    help="Destination root; dataset ends up in <out>/data/{train,val,test}")
    ap.add_argument("--symlink", action="store_true",
                    help="Symlink instead of copy (saves disk space, needs admin on Windows)")
    args = ap.parse_args()

    dest = (args.out / "data").resolve()
    if (dest / "train" / "images").is_dir():
        print(f"[skip] Dataset already present at {dest}")
        return

    try:
        import kagglehub
    except ImportError:
        print("Install kagglehub: pip install kagglehub", file=sys.stderr)
        sys.exit(1)

    print(f"[1/2] Downloading {DATASET_ID} via kagglehub...")
    cache_path = Path(kagglehub.dataset_download(DATASET_ID))
    print(f"      cache: {cache_path}")

    # kagglehub layout: <cache>/data/{train,val,test}/...
    src = cache_path / "data"
    if not src.is_dir():
        # fallback: some versions don't have 'data' wrapper
        if (cache_path / "train").is_dir():
            src = cache_path
        else:
            print(f"Cannot find train/val/test inside {cache_path}", file=sys.stderr)
            sys.exit(1)

    dest.parent.mkdir(parents=True, exist_ok=True)
    if args.symlink:
        print(f"[2/2] Symlinking {src} -> {dest}")
        if dest.exists():
            dest.unlink() if dest.is_symlink() else shutil.rmtree(dest)
        try:
            dest.symlink_to(src, target_is_directory=True)
        except OSError as e:
            print(f"Symlink failed ({e}); fallback to copy", file=sys.stderr)
            shutil.copytree(src, dest, dirs_exist_ok=True)
    else:
        print(f"[2/2] Copying {src} -> {dest} (may take a few minutes)")
        shutil.copytree(src, dest, dirs_exist_ok=True)

    # sanity check
    for split in ("train", "val", "test"):
        n = len(list((dest / split / "images").glob("*")))
        print(f"      {split}: {n} images")

    print(f"\nDone. Dataset ready at {dest}")
    print(f"Run: python -m src.train --data-root {args.out / 'data'}")


if __name__ == "__main__":
    main()