from __future__ import annotations

import os
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET_DIR = PROJECT_ROOT / "dataset_deepdetect" / "ddata"
DEFAULT_RUNS_DIR = PROJECT_ROOT / "runs" / "classify" / "runs_yolo"


def get_dataset_dir(dataset_dir: str | Path | None = None) -> Path:
    if dataset_dir is not None:
        return Path(dataset_dir).expanduser().resolve()

    env_value = os.getenv("DATA_DIR")
    if env_value:
        return Path(env_value).expanduser().resolve()

    return DEFAULT_DATASET_DIR


def get_runs_dir(runs_dir: str | Path | None = None) -> Path:
    if runs_dir is not None:
        return Path(runs_dir).expanduser().resolve()
    return DEFAULT_RUNS_DIR


def ensure_yolo_dataset_layout(dataset_dir: str | Path | None = None) -> Path:
    dataset_path = get_dataset_dir(dataset_dir)
    test_dir = dataset_path / "test"
    val_dir = dataset_path / "val"

    if not val_dir.exists() and test_dir.exists():
        shutil.copytree(test_dir, val_dir)

    return dataset_path
