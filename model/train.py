"""Etapa 2 — Entrenamiento de la MLP 784→32→10 sobre MNIST.

Uso (desde la raíz del repo):
    .venv/bin/python model/train.py
    .venv/bin/python model/train.py --no-augment    # para comparar sin augmentation

Salidas en model/artifacts/:
    mlp.pt    state_dict de PyTorch
    mlp.npz   los mismos pesos en float32 para NumPy: W1 (32,784), b1 (32,), W2 (10,32), b2 (10,)
    mlp.json  hiperparámetros, métricas finales y curva de entrenamiento
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets

HERE = Path(__file__).resolve().parent
N_VAL = 5000
SPLIT_SEED = 0  # separado de --seed: cambiar la semilla no cambia qué imágenes van a validación


def load_mnist(root=HERE / "data"):
    """Devuelve (x_train, y_train, x_test, y_test), con las imágenes en uint8 y forma (N, 784)."""
    train = datasets.MNIST(root, train=True, download=True)
    test = datasets.MNIST(root, train=False, download=True)
    return train.data.reshape(-1, 784), train.targets, test.data.reshape(-1, 784), test.targets


def to_float(x_u8):
    # Pixel uint8 → float en [0, 1], sin normalizar por media/desvío. Así, en la Etapa 3
    # la entrada queda cuantizada con escala 1/255 y zero-point 0: el pixel crudo ya ES
    # el valor cuantizado, y la primera capa (la de 25.088 MACs) no necesita corrección
    # por zero-point.
    return x_u8.float() / 255.0


def build_mlp():
    # Sin softmax al final: CrossEntropyLoss lo aplica internamente al entrenar, y en
    # inferencia alcanza con el argmax de los logits (softmax no altera el orden), así
    # que el hardware nunca necesita calcular exponenciales.
    return nn.Sequential(nn.Linear(784, 32), nn.ReLU(), nn.Linear(32, 10))


def augment(x, max_shift_px=1.0, max_rot_deg=10.0):
    """Traslación y rotación aleatorias, distintas para cada imagen del batch."""
    n = x.shape[0]
    angle = (torch.rand(n) * 2 - 1) * math.radians(max_rot_deg)
    # En las coordenadas normalizadas de affine_grid la imagen va de -1 a 1: 1 px = 2/28.
    shift = (torch.rand(n, 2) * 2 - 1) * max_shift_px * 2 / 28
    cos, sin = angle.cos(), angle.sin()
    theta = torch.stack([torch.stack([cos, -sin, shift[:, 0]], dim=1),
                         torch.stack([sin, cos, shift[:, 1]], dim=1)], dim=1)
    img = x.view(n, 1, 28, 28)
    grid = F.affine_grid(theta, img.shape, align_corners=False)
    return F.grid_sample(img, grid, align_corners=False).view(n, 784)


@torch.no_grad()
def accuracy(model, x, y):
    return (model(x).argmax(dim=1) == y).float().mean().item() * 100


def main():
    ap = argparse.ArgumentParser(description="Entrena la MLP 784→32→10 sobre MNIST.")
    # Con augmentation la red no memoriza, así que sigue mejorando al entrenar más, pero
    # se satura: en validación, +0,31 puntos de 30 a 60 épocas y +0,13 de 60 a 120.
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-3)
    # Probado con 0.1: no mejoró validación ni la pérdida al cuantizar los pesos a int8.
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=HERE / "artifacts")
    args = ap.parse_args()

    x_train_u8, y_train, x_test_u8, y_test = load_mnist()
    x_train, x_test = to_float(x_train_u8), to_float(x_test_u8)

    # Validación: 5.000 imágenes apartadas del set de train, para seguir el entrenamiento
    # y comparar hiperparámetros. El set de test se evalúa una única vez, al final.
    perm = torch.randperm(len(x_train), generator=torch.Generator().manual_seed(SPLIT_SEED))
    x_val, y_val = x_train[perm[:N_VAL]], y_train[perm[:N_VAL]]
    x_tr, y_tr = x_train[perm[N_VAL:]], y_train[perm[N_VAL:]]
    print(f"MLP 784→32→10 | {len(x_tr)} train / {len(x_val)} val / {len(x_test)} test | "
          f"augmentation: {'no' if args.no_augment else 'sí'}")

    torch.manual_seed(args.seed)
    model = build_mlp()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = math.ceil(len(x_tr) / args.batch_size)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * steps_per_epoch)
    loss_fn = nn.CrossEntropyLoss()

    history = []
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(len(x_tr))
        loss_sum = 0.0
        for i in range(0, len(x_tr), args.batch_size):
            idx = order[i:i + args.batch_size]
            xb = x_tr[idx] if args.no_augment else augment(x_tr[idx])
            loss = loss_fn(model(xb), y_tr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            loss_sum += loss.item() * len(idx)
        model.eval()
        epoch_loss, val_acc = loss_sum / len(x_tr), accuracy(model, x_val, y_val)
        history.append({"epoch": epoch, "loss": epoch_loss, "val_acc": val_acc})
        print(f"época {epoch:2d}/{args.epochs}  loss {epoch_loss:.4f}  val {val_acc:.2f}%")

    metrics = {
        "train_acc": accuracy(model, x_tr, y_tr),
        "val_acc": accuracy(model, x_val, y_val),
        "test_acc": accuracy(model, x_test, y_test),
    }
    print(f"\nfinal: train {metrics['train_acc']:.2f}%  val {metrics['val_acc']:.2f}%  "
          f"test {metrics['test_acc']:.2f}%  ({time.time() - t0:.0f} s)")

    args.out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.out / "mlp.pt")
    # Mismo layout que PyTorch, (salidas, entradas): una fila por neurona. Es el orden
    # natural del loop MAC en C, con los pesos de cada neurona contiguos en memoria.
    fc1, fc2 = model[0], model[2]
    weights = {"W1": fc1.weight, "b1": fc1.bias, "W2": fc2.weight, "b2": fc2.bias}
    weights = {k: v.detach().numpy() for k, v in weights.items()}
    np.savez(args.out / "mlp.npz", **weights)
    info = {
        "hparams": {k: v for k, v in vars(args).items() if k != "out"},
        "metrics": metrics,
        "max_abs_weight": {"W1": float(np.abs(weights["W1"]).max()),
                           "W2": float(np.abs(weights["W2"]).max())},
        "torch_version": torch.__version__,
        "history": history,
    }
    (args.out / "mlp.json").write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n")
    print(f"pesos guardados en {args.out}/ (mlp.pt, mlp.npz, mlp.json)")


if __name__ == "__main__":
    main()
