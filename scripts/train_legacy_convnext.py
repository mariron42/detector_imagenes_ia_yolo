from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from detector_deepfake.config import get_dataset_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Baseline legacy con ConvNeXt para comparacion historica.")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=4)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    dataset_dir = get_dataset_dir(args.dataset)
    train_dir = dataset_dir / "train"
    test_dir = dataset_dir / "test"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Usando dispositivo: {device}")
    if device.type == "cuda":
        print(f"[*] GPU detectada: {torch.cuda.get_device_name(0)}")

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = ImageFolder(train_dir, transform=train_transform)
    test_dataset = ImageFolder(test_dir, transform=test_transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=args.workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=True)

    class_names = train_dataset.classes
    print(f"[*] Clases detectadas: {class_names}")

    model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
    num_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(num_features, len(class_names))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.05)
    scaler = torch.amp.GradScaler(device="cuda") if device.type == "cuda" else None

    def evaluate() -> None:
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="[Validacion]"):
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        print(f"==> Precision en validacion: {(correct / total) * 100:.2f}%\n")

    def train() -> None:
        for epoch in range(args.epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [Entrenamiento]")
            for images, labels in loop:
                images = images.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                if scaler is not None:
                    with torch.amp.autocast("cuda"):
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                loop.set_postfix({"Loss": running_loss / total, "Acc": f"{(correct / total) * 100:.2f}%"})

            evaluate()
            torch.save(model.state_dict(), ROOT / f"convnext_v2_epoch{epoch + 1}.pth")

    print("[*] Baseline legacy ConvNeXt. Para el flujo actual, usa scripts/train.py")
    start_time = time.time()
    train()
    print(f"[*] Entrenamiento completado en {(time.time() - start_time) / 60:.2f} minutos.")


if __name__ == "__main__":
    main()
