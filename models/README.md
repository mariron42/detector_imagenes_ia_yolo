# Modelos

Los pesos grandes no forman parte del repositorio.

- Pesos base de Ultralytics: se descargan automaticamente al ejecutar `scripts/train.py` si no estan disponibles.
- Pesos entrenados (`best.pt`, `last.pt`): se generan en `runs/` y deben publicarse mediante releases, almacenamiento externo o Git LFS si decides conservarlos.
- Los pesos legacy `.pth` del baseline ConvNeXt se consideran historicos y no deben quedar en la raiz del repo publico.
