# Flujo de entrenamiento

El flujo recomendado del repositorio ya no parte del notebook, sino de scripts reproducibles.

## Entrenamiento YOLO

```powershell
python scripts/train.py --epochs 5 --batch 64 --workers 0 --imgsz 224 --device 0
```

Puntos importantes:

- En Windows, `workers=0` es la configuracion segura para esta maquina y evita errores de memoria en procesos secundarios.
- Si tu dataset solo tiene `test/`, el script crea `val/` copiando esa carpeta para cumplir el layout de YOLO Classification.
- Los resultados quedan en `runs/classify/runs_yolo/` y no deben subirse a Git.

## Inferencia

```powershell
python scripts/infer.py ruta/a/imagen.jpg
```

Si no pasas `--model`, el script busca el `best.pt` mas reciente dentro de las corridas del proyecto.

## Notebooks

El notebook recomendado para trabajar el flujo completo es `notebooks/01-yolo-entrenamiento-y-prueba.ipynb`.
El cuaderno `notebooks/02-training-yolo.ipynb` queda como transicion y apoyo.
Los notebooks en `notebooks/archive/` se conservan como referencia historica.
