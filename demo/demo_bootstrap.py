from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable


SHARED_FOLDER_URL = "https://drive.google.com/drive/folders/19J9mf99svPelodM0Xuvs7OrChEzmyPqu?usp=sharing"
DEMO_DIR = Path("/content/hagrid_yolov5_demo")
SHARED_DOWNLOAD_DIR = Path("/content/hagrid_yolov5_shared")
ENV_MARKER = Path("/content/.hagrid_yolov5_demo_env_ready")


def run_command(command: Iterable[str | Path]) -> None:
    subprocess.check_call([str(x) for x in command])


def find_repo_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    while current.parent != current:
        if (current / "gesture_inference_analysis.ipynb").exists() and (current / "colab_env.toml").exists():
            return current
        current = current.parent
    if (current / "gesture_inference_analysis.ipynb").exists() and (current / "colab_env.toml").exists():
        return current
    raise FileNotFoundError("YOLOv5 demo repo root not found.")


def ensure_demo_files(folder_url: str = SHARED_FOLDER_URL) -> Path:
    if (DEMO_DIR / "gesture_inference_analysis.ipynb").exists() and (DEMO_DIR / "colab_check_paths.py").exists():
        return DEMO_DIR

    try:
        return find_repo_root(Path.cwd())
    except FileNotFoundError:
        pass

    run_command([sys.executable, "-m", "pip", "install", "-q", "gdown"])
    run_command(["rm", "-rf", SHARED_DOWNLOAD_DIR])
    SHARED_DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    run_command([sys.executable, "-m", "gdown", "--folder", folder_url, "-O", SHARED_DOWNLOAD_DIR])

    zip_candidates = sorted(SHARED_DOWNLOAD_DIR.rglob("hagrid_yolov5_demo.zip"))
    if not zip_candidates:
        raise FileNotFoundError("hagrid_yolov5_demo.zip was not found in the shared Google Drive folder.")

    run_command(["rm", "-rf", DEMO_DIR])
    run_command(["unzip", "-q", zip_candidates[0], "-d", "/content"])

    if (DEMO_DIR / "gesture_inference_analysis.ipynb").exists():
        return DEMO_DIR
    nested = DEMO_DIR / "hagrid_yolov5_demo"
    if (nested / "gesture_inference_analysis.ipynb").exists():
        return nested
    raise FileNotFoundError("Unzipped demo folder does not contain the notebook at the expected path.")


def load_toml(path: Path) -> dict:
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib
    with path.open("rb") as f:
        return tomllib.load(f)


def environment_ready() -> bool:
    if not ENV_MARKER.exists():
        return False
    try:
        import matplotlib  # noqa: F401
        import numpy  # noqa: F401
        import pandas  # noqa: F401
        import scipy  # noqa: F401
        import seaborn  # noqa: F401
    except Exception:
        return False
    return True


def ensure_colab_environment(repo_root: Path) -> None:
    run_command([sys.executable, "-m", "pip", "install", "-q", "tomli"])
    env = load_toml(repo_root / "colab_env.toml")

    if environment_ready():
        return

    packages = env.get("pip", {}).get("packages", [])
    if packages:
        run_command(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                "--upgrade",
                "--force-reinstall",
                "--no-cache-dir",
                *packages,
            ]
        )
    ENV_MARKER.write_text("installed", encoding="utf-8")
    print("Environment installed. Restarting runtime. Run this cell again after reconnecting.")
    os.kill(os.getpid(), 9)


def setup_colab_demo(folder_url: str = SHARED_FOLDER_URL) -> Path:
    in_colab = "COLAB_RELEASE_TAG" in os.environ or "google.colab" in sys.modules
    if in_colab:
        repo_root = ensure_demo_files(folder_url)
        os.chdir(repo_root)
        ensure_colab_environment(repo_root)
        return repo_root
    return find_repo_root(Path.cwd())
