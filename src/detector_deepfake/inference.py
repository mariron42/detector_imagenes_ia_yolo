from __future__ import annotations

from pathlib import Path

from detector_deepfake.config import get_runs_dir


def find_latest_best_model(runs_dir: str | Path | None = None) -> Path | None:
    base_dir = get_runs_dir(runs_dir)
    candidates = sorted(base_dir.glob("*/weights/best.pt"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        return None
    return candidates[0]


def resolve_model_path(model_path: str | Path | None = None, runs_dir: str | Path | None = None) -> Path:
    if model_path is not None:
        resolved = Path(model_path).expanduser().resolve()
    else:
        resolved = find_latest_best_model(runs_dir)

    if resolved is None or not resolved.exists():
        raise FileNotFoundError("No se encontro ningun best.pt para inferencia.")

    return resolved


def classify_image(
    image_path: str | Path,
    model_path: str | Path | None = None,
    runs_dir: str | Path | None = None,
) -> dict[str, object]:
    from ultralytics import YOLO

    resolved_image = Path(image_path).expanduser().resolve()
    if not resolved_image.exists():
        raise FileNotFoundError(f"No existe la imagen: {resolved_image}")

    resolved_model = resolve_model_path(model_path, runs_dir)
    result = YOLO(str(resolved_model))(str(resolved_image))[0]
    probs = getattr(result, "probs", None)
    if probs is None:
        raise RuntimeError("El modelo no devolvio probabilidades.")

    scores: list[tuple[str, float]] = []
    for index, label in result.names.items():
        scores.append((label, float(probs.data[index])))

    scores.sort(key=lambda item: item[1], reverse=True)
    best_index = int(probs.top1)

    return {
        "image_path": resolved_image,
        "model_path": resolved_model,
        "scores": scores,
        "label": result.names[best_index],
        "confidence": float(probs.top1conf),
    }