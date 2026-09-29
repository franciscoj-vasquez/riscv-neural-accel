"""Etapa 3 — Cuantización post-entrenamiento de la MLP e inferencia 100% entera en NumPy.

Uso (desde la raíz del repo):
    .venv/bin/python model/quantize.py

Esquema (todo con zero-point 0):
    entrada   uint8   píxel crudo, escala Sx = 1/255
    pesos     int8    simétricos, una escala por matriz: Sw = max|W| / 127
    bias      int32   con la escala del acumulador: Sx · Sw
    capa 1    acumulador int32 → requantización (× M0, >> n) → uint8, escala Sh
              la saturación a [0, 255] hace de ReLU
    capa 2    acumulador int32 → argmax, sin requantizar

Sh se calibra con el máximo de la capa oculta sobre las 55.000 imágenes de train (nunca test).

Salidas en model/artifacts/:
    mlp_int8.npz   los enteros que va a usar el C: W1q, b1q, M0, n, W2q, b2q
    mlp_int8.json  escalas en float (solo para documentar) y accuracy float contra entera
"""
import json
from pathlib import Path

import numpy as np
import torch

from forward import forward, load_weights
from train import N_VAL, SPLIT_SEED, load_mnist

HERE = Path(__file__).resolve().parent
SX = 1 / 255  # escala de la entrada: el píxel uint8 ya es el valor cuantizado
M0_BITS = 31  # M0 en [2^30, 2^31): entra en un int32 positivo con 31 bits de precisión


def split_mnist():
    """Mismo reparto que train.py: 55.000 train / 5.000 validación / 10.000 test, en uint8."""
    x_train, y_train, x_test, y_test = (t.numpy() for t in load_mnist())
    perm = torch.randperm(len(x_train),
                          generator=torch.Generator().manual_seed(SPLIT_SEED)).numpy()
    val, tr = perm[:N_VAL], perm[N_VAL:]
    return (x_train[tr], y_train[tr]), (x_train[val], y_train[val]), (x_test, y_test)


def to_fixed_point(m):
    """Escribe el real m (0 < m < 1) como M0 · 2^-n, con M0 entero de M0_BITS bits.

    Se elige el n más grande con M0 < 2^M0_BITS: así M0 usa todos sus bits y el error
    relativo de la aproximación (a lo sumo 0,5 / M0) queda por debajo de 2^-M0_BITS.
    """
    n = 0
    while m * 2 ** (n + 1) < 2 ** M0_BITS:
        n += 1
    m0 = round(m * 2 ** n)
    assert 2 ** (M0_BITS - 1) <= m0 < 2 ** M0_BITS
    return m0, n


def quantize(w, x_cal):
    """Cuantiza los pesos float de `w` y calibra la capa oculta con las imágenes uint8 x_cal.

    Devuelve (q, s): q son los enteros de la inferencia; s las escalas en float, que solo
    se usan para documentar y para volver a float (valor real = escala × entero).
    """
    # Calibración: el máximo de la capa oculta sobre x_cal pasa a valer 255.
    h_cal, _ = forward(x_cal.astype(np.float32) / 255, w)
    s_h = float(h_cal.max()) / 255
    w = {k: v.astype(np.float64) for k, v in w.items()}  # escalas en doble precisión
    s_w1 = float(np.abs(w["W1"]).max()) / 127
    s_w2 = float(np.abs(w["W2"]).max()) / 127

    q = {
        # Pesos: el entero más cercano en unidades de Sw. El máximo cae justo en ±127.
        "W1q": np.round(w["W1"] / s_w1).astype(np.int8),
        "W2q": np.round(w["W2"] / s_w2).astype(np.int8),
        # Bias: en la escala del acumulador de su capa, para sumarlo directo, sin reescalar.
        "b1q": np.round(w["b1"] / (SX * s_w1)).astype(np.int32),
        "b2q": np.round(w["b2"] / (s_h * s_w2)).astype(np.int32),
    }
    m = SX * s_w1 / s_h  # factor de la requantización: escala del acumulador → escala de h
    q["M0"], q["n"] = to_fixed_point(m)
    s = {"Sx": SX, "Sw1": s_w1, "Sw2": s_w2, "Sh": s_h, "M": m,
         "acc1": SX * s_w1, "acc2": s_h * s_w2}
    return q, s


def requantize(acc, m0, n):
    """int32 → uint8: round(acc · M0 / 2^n), saturado a [0, 255].

    El redondeo es sumar medio paso (2^(n-1)) y correr n bits: redondea los .5 hacia +∞.
    El producto necesita 64 bits (|acc| < 2^20, M0 < 2^31). La saturación por abajo hace
    de ReLU: un acumulador negativo da 0.
    """
    rounded = (acc.astype(np.int64) * m0 + (1 << (n - 1))) >> n
    return np.clip(rounded, 0, 255).astype(np.uint8)


def forward_int(x_u8, q):
    """Inferencia 100% entera. x_u8: (N, 784) uint8.

    Devuelve acc1 (N, 32) int32, h (N, 32) uint8 y acc2 (N, 10) int32. La predicción es
    el argmax de acc2 (ante empate, el primero, igual que un loop con `>` en C).
    """
    # Las cuentas van en int64 para que NumPy no desborde en silencio; después se verifica
    # que todo entra en int32, que es lo que va a usar el C.
    acc1 = x_u8.astype(np.int64) @ q["W1q"].T.astype(np.int64) + q["b1q"]
    h = requantize(acc1, q["M0"], q["n"])
    acc2 = h.astype(np.int64) @ q["W2q"].T.astype(np.int64) + q["b2q"]
    for acc in (acc1, acc2):
        assert np.abs(acc).max() < 2 ** 31
    return acc1.astype(np.int32), h, acc2.astype(np.int32)


def show_example(i, x_u8, label, w, q, s):
    """Sigue una imagen por todo el camino entero, comparando cada paso con el float."""
    h_f, z_f = forward(x_u8[None].astype(np.float32) / 255, w)
    acc1, h, acc2 = forward_int(x_u8[None], q)
    h_f, z_f, acc1, h, acc2 = h_f[0], z_f[0], acc1[0], h[0], acc2[0]
    j = int(h_f.argmax())  # la neurona oculta más activa

    print(f"\nEjemplo: imagen {i} del test (dígito {label}), neurona oculta {j}")
    print(f"  float:     h = {h_f[j]:.4f}")
    print(f"  acc1 = Σ W1q·píxel + b1q = {acc1[j] - q['b1q'][j]:,} + {q['b1q'][j]:,} "
          f"= {acc1[j]:,}")
    print(f"     en real: acc1 × Sx·Sw1 = {acc1[j] * s['acc1']:.4f}")
    m0, n = q["M0"], q["n"]
    prod = int(acc1[j]) * m0
    print(f"  requantización, en enteros: acc1 × M0 = {int(acc1[j]):,} × {m0:,} = {prod:,}")
    print(f"     + 2^{n - 1} (medio paso) y >> {n}: {(prod + (1 << (n - 1))) >> n} "
          f"→ h = {h[j]}  (con M en float: {int(acc1[j]) * s['M']:.4f})")
    print(f"     en real: h × Sh = {h[j]} × {s['Sh']:.6f} = {h[j] * s['Sh']:.4f}")
    print(f"  capa oculta: {np.count_nonzero(h)} de {len(h)} neuronas activas; "
          f"máx. error contra float {np.abs(h * s['Sh'] - h_f).max():.4f}")
    print("  acc2:   " + "  ".join(f"{d}:{v:+,}" for d, v in enumerate(acc2)))
    print("  en real " + "  ".join(f"{d}:{v * s['acc2']:+.1f}" for d, v in enumerate(acc2)))
    print("  float   " + "  ".join(f"{d}:{v:+.1f}" for d, v in enumerate(z_f)))
    print(f"  → predicción entera {int(acc2.argmax())}, float {int(z_f.argmax())}")


def evaluate(name, x_u8, y, w, q):
    _, z_f = forward(x_u8.astype(np.float32) / 255, w)
    _, _, acc2 = forward_int(x_u8, q)
    p_f, p_i = z_f.argmax(axis=1), acc2.argmax(axis=1)
    acc_f, acc_i = (p_f == y).mean() * 100, (p_i == y).mean() * 100
    changed = p_f != p_i
    fixed, broken = int((changed & (p_i == y)).sum()), int((changed & (p_f == y)).sum())
    print(f"  {name:10s} float {acc_f:.2f}%  entera {acc_i:.2f}%  | cambian {changed.sum()} "
          f"de {len(y)} predicciones: {fixed} mal→bien, {broken} bien→mal, "
          f"{changed.sum() - fixed - broken} mal→mal")
    return {"float": round(acc_f, 2), "int": round(acc_i, 2), "changed": int(changed.sum()),
            "fixed": fixed, "broken": broken}


def main():
    w = load_weights()
    (x_tr, _), (x_val, y_val), (x_test, y_test) = split_mnist()
    q, s = quantize(w, x_tr)

    print("Escalas (valor real = escala × entero):")
    print(f"  entrada  Sx  = 1/255 = {s['Sx']:.6f}")
    print(f"  pesos    Sw1 = {s['Sw1']:.6f}  (max|W1| = {np.abs(w['W1']).max():.4f})")
    print(f"           Sw2 = {s['Sw2']:.6f}  (max|W2| = {np.abs(w['W2']).max():.4f})")
    print(f"  oculta   Sh  = {s['Sh']:.6f}  (máximo en train = {s['Sh'] * 255:.4f})")
    print(f"Requantización: M = Sx·Sw1/Sh = {s['M']:.6e} ≈ M0 / 2^n = {q['M0']:,} / 2^{q['n']}"
          f"  (error relativo {abs(q['M0'] / 2 ** q['n'] / s['M'] - 1):.1e})")
    print(f"Bias: b1q de {q['b1q'].min():,} a {q['b1q'].max():,}; "
          f"b2q de {q['b2q'].min():,} a {q['b2q'].max():,}")

    i = 0
    show_example(i, x_test[i], y_test[i], w, q, s)

    print("\nAccuracy, float contra entera:")
    results = {"val": evaluate("validación", x_val, y_val, w, q),
               "test": evaluate("test", x_test, y_test, w, q)}

    out = HERE / "artifacts"
    np.savez(out / "mlp_int8.npz", **q)
    info = {"scales": s, "M0": q["M0"], "n": q["n"], "calibration": "max, 55.000 train",
            "accuracy": results}
    (out / "mlp_int8.json").write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n")
    print(f"\nEnteros guardados en {out}/ (mlp_int8.npz, mlp_int8.json)")


if __name__ == "__main__":
    main()
