# Detector de Deepfakes

Clasificador binario de imagenes `real` vs `fake` orientado a deteccion de contenido generado por IA.

El proyecto nacio con un baseline en ConvNeXt y hoy utiliza YOLO Classification como flujo principal. Esta reorganizacion deja el repositorio listo para publicarse con una estructura mas limpia, scripts reproducibles y notebooks como apoyo.

## Estado actual

- Flujo principal: `scripts/train.py`
- Inferencia puntual: `scripts/infer.py`
- Notebook de apoyo: `notebooks/02-training-yolo.ipynb`
- Baseline legacy: `scripts/train_legacy_convnext.py`

## Estructura

```text
.
├── docs/
├── notebooks/
├── scripts/
├── src/
├── data/
├── models/
└── results/
```

## Requisitos

- Python 3.11 o 3.12
- GPU NVIDIA recomendada para entrenamiento
- PyTorch nightly con CUDA 12.8 si vas a reproducir el entorno validado en RTX 5060

Instalacion base:

```powershell
python -m pip install -r requirements.txt
```

## Dataset esperado

El dataset no se sube al repositorio. Debe existir localmente con esta estructura:

```text
dataset_deepdetect/ddata/
    train/
        fake/
        real/
    test/
        fake/
        real/
```

El script de entrenamiento crea `val/` a partir de `test/` si hace falta.

## Entrenamiento

```powershell
python scripts/train.py --epochs 5 --batch 64 --workers 0 --imgsz 224 --device 0
```

## Inferencia

```powershell
python scripts/infer.py ruta/a/imagen.jpg
```

## Demo rapida con una imagen

```powershell
python scripts/demo_image.py ruta/a/imagen.jpg
```

## Documentacion

- `docs/TRAINING.md`: flujo recomendado del proyecto
- `docs/hardware/RTX5060-CUDA-SETUP.md`: notas especificas de la maquina validada
- `results/EXPERIMENTS.md`: resumen liviano de corridas historicas

## Que no se versiona

Este repositorio excluye deliberadamente:

- datasets locales
- pesos `.pt` y `.pth`
- salidas de entrenamiento en `runs/`
- entornos virtuales
- secretos en `.env`

## Notas

Si vas a publicar este proyecto en GitHub, rota cualquier credencial que haya vivido en `.env` antes de hacer el primer push.
