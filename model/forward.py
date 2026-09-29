"""Etapas 1 y 2 — Forward pass de la MLP en NumPy, verificado contra PyTorch.

Uso (desde la raíz del repo):
    .venv/bin/python model/forward.py

Con los pesos de model/artifacts/ hace tres cosas:
    1. Imprime las activaciones intermedias de 10 imágenes de test, una por dígito.
    2. Mide la accuracy del forward NumPy sobre las 10.000 imágenes de test.
    3. Checkpoint de la Etapa 2: lo compara con PyTorch. En float no se puede exigir igualdad
       bit a bit (sumar en otro orden cambia el redondeo), así que el criterio es: mismas
       predicciones en las 10.000 imágenes y diferencias menores a TOL. Si no se cumple,
       sale con código 1.
"""
import sys
from pathlib import Path

import numpy as np
import torch

from train import build_mlp, load_mnist

HERE = Path(__file__).resolve().parent
TOL = 1e-4  # las diferencias medidas son de ~2e-5; los logits llegan a ~75 en valor absoluto


def load_weights(path=HERE / "artifacts" / "mlp.npz"):
    """W1 (32, 784), b1 (32,), W2 (10, 32), b2 (10,), en float32. Una fila de W por neurona."""
    return dict(np.load(path))


def forward(x, w):
    """x: (N, 784), píxeles en [0, 1].

    Devuelve h (N, 32), la salida de la capa oculta, y z (N, 10), los logits.
    """
    # Capa oculta: cada neurona suma sus 784 productos píxel × peso, le suma su bias y pasa
    # por ReLU. x @ W1.T hace las 32 sumas de todas las imágenes de una vez.
    h = np.maximum(0, x @ w["W1"].T + w["b1"])
    # Capa de salida: lo mismo con 10 neuronas de 32 entradas, sin ReLU. La predicción es el
    # argmax de los logits, así que no hace falta softmax.
    z = h @ w["W2"].T + w["b2"]
    return h, z


def show_image(i, x_u8, label, h, z):
    print(f"\nImagen {i} del test: dígito {label}, "
          f"{np.count_nonzero(x_u8)} de {x_u8.size} píxeles con trazo")
    print(f"  capa oculta, {np.count_nonzero(h)} de {len(h)} neuronas activas "
          "(· = apagada por la ReLU):")
    for start in range(0, len(h), 8):
        values = "".join("      ·" if v == 0 else f"{v:7.2f}" for v in h[start:start + 8])
        print(f"    {start:2d}-{start + 7:2d}:{values}")
    print("  logits: " + "  ".join(f"{d}:{v:+.1f}" for d, v in enumerate(z)))
    second, first = np.sort(z)[-2:]
    pred = int(z.argmax())
    print(f"  → predicción {pred} ({'bien' if pred == label else 'MAL'}), "
          f"margen {first - second:.1f} sobre el segundo logit")


def main():
    w = load_weights()
    print("Pesos: " + "  ".join(f"{k} {w[k].shape}" for k in ("W1", "b1", "W2", "b2")))
    print(f"MACs por imagen: {w['W1'].size} + {w['W2'].size} = {w['W1'].size + w['W2'].size}")

    _, _, x_u8, y = load_mnist()
    x_u8, y = x_u8.numpy(), y.numpy()
    x = x_u8.astype(np.float32) / 255  # igual que to_float() de train.py, bit a bit
    h, z = forward(x, w)
    pred = z.argmax(axis=1)

    # 1. Activaciones de 10 imágenes: la primera de cada dígito en el set de test.
    for digit in range(10):
        i = int(np.argmax(y == digit))
        show_image(i, x_u8[i], y[i], h[i], z[i])

    # 2. Accuracy.
    print(f"\nAccuracy del forward NumPy en test: {(pred == y).mean() * 100:.2f}%")

    # 3. Checkpoint: el mismo cálculo en PyTorch, con la misma entrada.
    model = build_mlp()
    model.load_state_dict(torch.load(HERE / "artifacts" / "mlp.pt"))
    with torch.no_grad():
        h_pt = model[1](model[0](torch.from_numpy(x)))  # Linear + ReLU
        z_pt = model[2](h_pt)
    h_pt, z_pt = h_pt.numpy(), z_pt.numpy()
    same = int((z_pt.argmax(axis=1) == pred).sum())
    dh, dz = float(np.abs(h - h_pt).max()), float(np.abs(z - z_pt).max())
    ok = same == len(y) and max(dh, dz) < TOL
    print(f"\nCheckpoint de la Etapa 2, NumPy contra PyTorch ({len(y)} imágenes de test):")
    print(f"  predicciones iguales:        {same} / {len(y)}")
    print(f"  máx. diferencia capa oculta: {dh:.1e}")
    print(f"  máx. diferencia logits:      {dz:.1e}  (tolerancia {TOL:.0e})")
    print(f"  logits idénticos bit a bit:  {(z == z_pt).mean() * 100:.1f}%")
    print(f"  → {'OK' if ok else 'FALLA'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
