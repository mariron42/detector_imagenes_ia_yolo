from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from detector_deepfake.inference import classify_image


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ejecuta inferencia sobre una imagen con el ultimo best.pt disponible.")
    parser.add_argument("image", type=str, help="Ruta a la imagen a clasificar.")
    parser.add_argument("--model", type=str, default=None, help="Ruta explicita al modelo best.pt.")
    parser.add_argument("--runs-dir", type=str, default=None, help="Directorio base de corridas YOLO.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    prediction = classify_image(args.image, model_path=args.model, runs_dir=args.runs_dir)

    print(f"[*] Imagen: {prediction['image_path']}")
    print(f"[*] Modelo: {prediction['model_path']}")

    for label, score in prediction["scores"]:
        print(f"{label}: {score:.4f}")

    print(f"Prediccion final: {prediction['label']} ({prediction['confidence']:.2%})")


if __name__ == "__main__":
    main()
