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
    parser = argparse.ArgumentParser(description="Hace una demostracion simple clasificando una imagen como real o fake.")
    parser.add_argument("image", type=str, help="Ruta a la imagen a evaluar.")
    parser.add_argument("--model", type=str, default=None, help="Ruta explicita a best.pt.")
    parser.add_argument("--runs-dir", type=str, default=None, help="Directorio base de corridas YOLO.")
    parser.add_argument("--top-k", type=int, default=2, help="Cuantas clases mostrar en el resumen.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    prediction = classify_image(args.image, model_path=args.model, runs_dir=args.runs_dir)

    print("=== Demo Deepfake ===")
    print(f"Imagen: {prediction['image_path']}")
    print(f"Modelo: {prediction['model_path']}")
    print()
    print(f"Resultado: {prediction['label']} ({prediction['confidence']:.2%})")
    print()
    print("Top clases:")
    for label, score in prediction["scores"][: max(args.top_k, 1)]:
        print(f"- {label}: {score:.2%}")

    if str(prediction["label"]).lower() == "fake":
        print()
        print("Interpretacion: la imagen se parece mas a la clase fake.")
    elif str(prediction["label"]).lower() == "real":
        print()
        print("Interpretacion: la imagen se parece mas a la clase real.")


if __name__ == "__main__":
    main()