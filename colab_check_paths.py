"""Check expected Colab demo paths before running the notebook."""

from __future__ import annotations

from pathlib import Path


ROOT = Path.cwd().resolve()
EXPECTED_PATHS = {
    "notebook": ROOT / "gesture_inference_analysis.ipynb",
    "env_toml": ROOT / "colab_env.toml",
    "bootstrap": ROOT / "demo" / "demo_bootstrap.py",
    "demo_utils": ROOT / "demo" / "demo_utils.py",
    "data_yaml": ROOT / "datasets" / "HAGRID-YOLO" / "data.yaml",
    "batch_images": ROOT / "datasets" / "HAGRID-YOLO" / "test" / "images",
    "single_demo_images": ROOT / "assets" / "inference_images",
    "curated_test_set": ROOT / "assets" / "curated_test_set",
    "weights": ROOT / "artifacts" / "runs" / "train" / "exp_batchsize_32" / "weights" / "best.pt",
    "yolov5_runtime": ROOT / "yolov5_runtime",
    "models": ROOT / "yolov5_runtime" / "models",
    "utils": ROOT / "yolov5_runtime" / "utils",
}


def count_images(path: Path) -> int:
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    if not path.is_dir():
        return 0
    return sum(1 for p in path.iterdir() if p.is_file() and p.suffix.lower() in suffixes)


print(f"Repo root: {ROOT}")
for name, path in EXPECTED_PATHS.items():
    exists = path.exists()
    extra = ""
    if name in {"batch_images", "single_demo_images", "curated_test_set"}:
        extra = f" | images={count_images(path)}"
    print(f"{name:18s} exists={str(exists):5s} {path}{extra}")

missing = [name for name, path in EXPECTED_PATHS.items() if not path.exists()]
if missing:
    raise SystemExit(f"Missing required paths: {missing}")

print("Colab demo paths look ready.")
