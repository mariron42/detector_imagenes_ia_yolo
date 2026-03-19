from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from detector_deepfake.config import ensure_yolo_dataset_layout, get_runs_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Entrena el clasificador YOLO del proyecto.")
    parser.add_argument("--dataset", type=str, default=None, help="Ruta al dataset con carpetas train y test/val.")
    parser.add_argument("--model", type=str, default="yolov8n-cls.pt", help="Peso base de YOLO para clasificación.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", type=str, default=None, help="Directorio donde guardar runs.")
    parser.add_argument("--name", type=str, default="deepfake_det_public")
    parser.add_argument("--fraction", type=float, default=1.0, help="Fraccion del dataset a usar, util para smoke tests.")
    parser.add_argument("--exist-ok", action="store_true", help="Permite reutilizar el nombre de corrida si ya existe.")
    parser.add_argument("--cache", action="store_true", help="Activa cache de imagenes en entrenamiento.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    from ultralytics import YOLO

    dataset_path = ensure_yolo_dataset_layout(args.dataset)
    runs_dir = get_runs_dir(args.project)

    print(f"[*] Dataset: {dataset_path}")
    print(f"[*] Runs: {runs_dir}")
    print(f"[*] Modelo base: {args.model}")

    model = YOLO(args.model)
    model.train(
        data=str(dataset_path),
        epochs=args.epochs,
        batch=args.batch,
        workers=args.workers,
        fraction=args.fraction,
        cache=args.cache,
        imgsz=args.imgsz,
        project=str(runs_dir),
        name=args.name,
        device=args.device,
        exist_ok=args.exist_ok,
    )



if __name__ == "__main__":
    main()
