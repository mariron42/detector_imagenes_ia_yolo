from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from detector_deepfake.config import get_dataset_dir
from detector_deepfake.utils import clean_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Elimina archivos invalidos o corruptos del dataset.")
    parser.add_argument("--dataset", type=str, default=None, help="Ruta al dataset a validar.")
    parser.add_argument("--dry-run", action="store_true", help="Solo cuenta archivos invalidos sin borrarlos.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    dataset_path = get_dataset_dir(args.dataset)
    summary = clean_dataset(dataset_path, dry_run=args.dry_run)

    print(f"[*] Dataset: {dataset_path}")
    print(f"[*] Archivos revisados: {summary.scanned_files}")
    print(f"[*] Archivos invalidos: {summary.removed_files}")
    if args.dry_run:
        print("[*] Modo simulacion activado: no se eliminaron archivos.")


if __name__ == "__main__":
    main()
