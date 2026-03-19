from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class CleaningSummary:
    scanned_files: int = 0
    removed_files: int = 0


def clean_dataset(dataset_dir: str | Path, dry_run: bool = False) -> CleaningSummary:
    import cv2

    dataset_path = Path(dataset_dir)
    valid_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    summary = CleaningSummary()

    for file_path in (path for path in dataset_path.rglob("*") if path.is_file()):
        summary.scanned_files += 1

        if file_path.suffix.lower() not in valid_suffixes:
            if not dry_run:
                file_path.unlink(missing_ok=True)
            summary.removed_files += 1
            continue

        try:
            image = cv2.imread(str(file_path))
        except Exception:
            image = None

        if image is None:
            if not dry_run:
                file_path.unlink(missing_ok=True)
            summary.removed_files += 1

    return summary
