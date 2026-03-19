# Guia de entorno validado en RTX 5060

Fecha de validacion: 2026-02-28
Proyecto: Detector de Deepfakes

Esta guia queda como documentacion avanzada de hardware. El onboarding general del repositorio esta en `README.md` y el flujo recomendado en `docs/TRAINING.md`.

## 1) Hardware y entorno detectados

- GPU: NVIDIA GeForce RTX 5060 Laptop GPU
- VRAM: 7.96 GB
- Python (kernel del notebook): d:\proyectos\Detector de ia\.venv312\Scripts\python.exe
- Entorno virtual: .venv312

## 2) Problemas encontrados y causa raíz

Durante la configuración aparecieron tres problemas principales:

1. Error ModuleNotFoundError: No module named torch
   - Causa: instalación incompleta o conflicto de dependencias en el kernel activo.

2. Error de pip con torchaudio
   - Mensaje: No matching distribution found for torchaudio
   - Causa: para esta combinación de nightly/cuda/python no siempre hay wheel de torchaudio.
   - Nota: para este proyecto de imágenes, torchaudio no es necesario.

3. Error CUDA: no kernel image is available for execution on the device
   - Causa: build antigua de PyTorch sin soporte efectivo para arquitectura Blackwell (sm_120).

## 3) Stack compatible que SÍ funciona en esta computadora

Versiones verificadas y funcionales:

- torch: 2.12.0.dev20260228+cu128
- torchvision: 0.26.0.dev20260221+cu128
- CUDA en PyTorch: 12.8

Índice utilizado:

- https://download.pytorch.org/whl/nightly/cu128

## 4) Comandos recomendados de instalación (limpia)

Ejecutar en PowerShell dentro del proyecto:

1) Limpiar versiones previas:

python -m pip uninstall -y torch torchvision torchaudio

2) Instalar stack compatible para RTX 5060:

python -m pip install --upgrade --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128

3) Instalar utilidades del notebook:

python -m pip install tensorboard tqdm pillow pandas matplotlib

4) Verificación rápida:

python -m pip show torch torchvision

## 5) Flujo correcto dentro del notebook

Archivo: Detector_Deepfake_RTX5060.ipynb

Orden de ejecución:

1. Celda 1: instalación
2. Reiniciar kernel
3. Celda 2: verificación CUDA real
4. Celda 3: dataset y dataloaders
5. Celda 4: modelo
6. Celda 5: entrenamiento
7. Celda 6: iniciar entrenamiento

## 6) Señales de que todo está bien

En la celda de verificación debe aparecer:

- Torch 2.12.0.dev...+cu128
- Torchvision 0.26.0.dev...+cu128
- torch.cuda.is_available(): True
- Test de kernel CUDA superado (compatibilidad OK)

## 7) Estructura de dataset esperada

- dataset_deepdetect/ddata/train/fake
- dataset_deepdetect/ddata/train/real
- dataset_deepdetect/ddata/test/fake
- dataset_deepdetect/ddata/test/real

## 8) Ajustes sugeridos para esta GPU

- Batch size inicial recomendado: 24 (estable en Windows con 8 GB VRAM)
- AMP: activado (acelera y reduce uso de VRAM)
- En Windows: usar num_workers = 0 (modo estable)
- En Linux: workers sugeridos hasta 4

Si aparece error de memoria (OOM):

1. Bajar batch size de 24 a 16, luego 12
2. Mantener AMP activado
3. Cerrar procesos que usen GPU en paralelo

## 9) Estabilidad en VS Code (cambios de intérprete)

En esta PC se observó que VS Code puede alternar temporalmente entre varios Python (por ejemplo 3.14 global, WindowsApps y .venv312), lo que puede provocar:

- pausas largas,
- GPU en cero uso,
- cuaderno aparentemente congelado.

Para evitarlo:

1. Seleccionar explícitamente el kernel `.venv312` del proyecto.
2. No cambiar de intérprete mientras entrena.
3. Cerrar terminales activadas repetidas veces si no se usan.
4. Mantener un solo notebook entrenando al mismo tiempo.
5. Si se pone “rara” la PC: detener entrenamiento, reiniciar kernel, ejecutar celdas 2→6 otra vez.

## 10) Errores típicos y solución rápida

A) ModuleNotFoundError: torch
- Ejecutar celda 1
- Reiniciar kernel
- Repetir celda 2

B) No matching distribution found for torchaudio
- Quitar torchaudio de la instalación (no se usa para imágenes)

C) no kernel image is available
- Verificar que torch y torchvision sean cu128 y nightly recientes
- Reinstalar con el índice cu128
- Reiniciar kernel

D) La GPU deja de usarse y el entrenamiento se queda parado
- Verificar que sigues en `.venv312` (celda 2 imprime el python del kernel)
- En Windows mantener `num_workers = 0`
- Reducir batch size si hay presión de VRAM
- Reiniciar kernel y relanzar desde celda 2

## 11) Comprobación final de salud

Antes de entrenar, confirmar:

- Kernel correcto seleccionado (.venv312)
- torch y torchvision en cu128
- CUDA disponible
- Test de kernel CUDA en la celda 2 exitoso

Si se cumplen esos puntos, el entrenamiento en esta RTX 5060 debe funcionar correctamente.
