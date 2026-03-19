# Datos

El dataset no se versiona en Git.

Estructura esperada:

```text
dataset_deepdetect/ddata/
  train/
    fake/
    real/
  test/
    fake/
    real/
```

El script `scripts/train.py` copia `test/` a `val/` si hace falta para compatibilidad con YOLO Classification.
