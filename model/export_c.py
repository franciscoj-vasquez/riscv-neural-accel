"""Etapa 3 — Exporta la red cuantizada a C y genera los vectores del checkpoint.

Uso (desde la raíz del repo; normalmente lo llama `make -C reference check`):
    .venv/bin/python model/export_c.py

Parte de los enteros congelados de model/artifacts/mlp_int8.npz, no los recalcula.

Salidas:
    model/artifacts/mlp_int8.h   los enteros como arreglos const de C. Se versiona: es el
                                 mismo modelo que mlp_int8.npz, en el formato que compila
                                 reference/mlp.c, en la PC y después en el PicoRV32.
    reference/build/vectors.bin  las 10.000 imágenes de test con todos los enteros que calcula
                                 forward_int(). No se versiona: lo lee reference/check.c.

Formato de vectors.bin (little-endian, sin relleno entre campos):
    encabezado   "MLPVEC01" y cuatro uint32: imágenes, entradas, ocultas, salidas
    un registro  x uint8[784] · etiqueta uint8 · acc1 int32[32] · h uint8[32] ·
    por imagen   acc2 int32[10] · predicción uint8
"""
import struct
from pathlib import Path

import numpy as np

from quantize import forward_int, load_quantized, split_mnist

HERE = Path(__file__).resolve().parent
HEADER = HERE / "artifacts" / "mlp_int8.h"
VECTORS = HERE.parent / "reference" / "build" / "vectors.bin"
MAGIC = b"MLPVEC01"


def c_rows(rows, per_line, width, labels):
    """Cuerpo de un arreglo 2D de C: una llave por fila, con `per_line` valores por renglón."""
    out = []
    for label, row in zip(labels, rows):
        out.append(f"    {{ /* {label} */")
        for start in range(0, len(row), per_line):
            values = row[start:start + per_line]
            out.append("        " + "".join(f"{int(v):{width}d}," for v in values))
        out.append("    },")
    return out


def c_header(q, s):
    w1, b1, w2, b2 = q["W1q"], q["b1q"], q["W2q"], q["b2q"]
    hidden, n_in = w1.shape
    n_out = w2.shape[0]
    side = int(round(n_in ** 0.5))  # 28: cada renglón de una neurona es una fila de la imagen
    lines = [
        f"/* Parámetros enteros de la MLP {n_in}→{hidden}→{n_out}. Generado por",
        " * model/export_c.py a partir de mlp_int8.npz: no editar a mano, regenerar con el",
        " * script.",
        " *",
        " * Lo incluye reference/mlp.c. Una fila por neurona, en el mismo orden que en PyTorch:",
        " *   mlp_w1[j][i]  peso int8 del píxel i en la neurona oculta j. Cada renglón es",
        f" *                 una fila de la imagen ({side} píxeles): se ve la forma de la",
        " *                 neurona.",
        " *   mlp_b1[j]     bias int32, en la escala del acumulador de la capa 1 (Sx·Sw1)",
        " *   mlp_w2[k][j]  peso int8 de la neurona oculta j en la salida k",
        " *   mlp_b2[k]     bias int32, en la escala del acumulador de la capa 2 (Sh·Sw2)",
        " *",
        f" * Escalas, solo como referencia: Sx = 1/255, Sw1 = {s['Sw1']:.6f}, Sh = {s['Sh']:.6f},",
        f" * Sw2 = {s['Sw2']:.6f}.",
        " */",
        "#ifndef MLP_INT8_H",
        "#define MLP_INT8_H",
        "",
        "#include <stdint.h>",
        "",
        "/* Requantización de la capa oculta: M = Sx·Sw1/Sh ≈ MLP_M0 / 2^MLP_N. */",
        f"#define MLP_M0 {q['M0']}u",
        f"#define MLP_N  {q['n']}",
        "",
        f"static const int8_t mlp_w1[{hidden}][{n_in}] = {{",
        *c_rows(w1, side, 4, [f"neurona {j}" for j in range(hidden)]),
        "};",
        "",
        f"static const int32_t mlp_b1[{hidden}] = {{",
        *["   " + " ".join(f"{int(v):7d}," for v in b1[i:i + 8]) for i in range(0, hidden, 8)],
        "};",
        "",
        f"static const int8_t mlp_w2[{n_out}][{hidden}] = {{",
        *c_rows(w2, 16, 4, [f"salida {k}" for k in range(n_out)]),
        "};",
        "",
        f"static const int32_t mlp_b2[{n_out}] = {{",
        "   " + " ".join(f"{int(v):5d}," for v in b2),
        "};",
        "",
        "#endif /* MLP_INT8_H */",
        "",
    ]
    return "\n".join(lines)


def test_vectors(x, y, q):
    """Un registro por imagen, con lo que calcula forward_int(): el formato del docstring."""
    acc1, h, acc2 = forward_int(x, q)
    n_in, hidden, n_out = x.shape[1], acc1.shape[1], acc2.shape[1]
    rec = np.zeros(len(x), dtype=[("x", "u1", n_in), ("label", "u1"), ("acc1", "<i4", hidden),
                                  ("h", "u1", hidden), ("acc2", "<i4", n_out), ("pred", "u1")])
    rec["x"], rec["label"], rec["acc1"], rec["h"], rec["acc2"] = x, y, acc1, h, acc2
    rec["pred"] = acc2.argmax(axis=1)  # ante empate, el primero: igual que mlp.c
    head = MAGIC + struct.pack("<4I", len(x), n_in, hidden, n_out)
    return head + rec.tobytes(), rec


def main():
    q, s = load_quantized()
    header = c_header(q, s)
    HEADER.write_text(header)
    params = sum(q[k].nbytes for k in ("W1q", "b1q", "W2q", "b2q"))
    print(f"model/artifacts/mlp_int8.h: {params:,} bytes de parámetros "
          f"({len(header):,} bytes de texto)")

    _, _, (x_test, y_test) = split_mnist()
    data, rec = test_vectors(x_test, y_test, q)
    VECTORS.parent.mkdir(exist_ok=True)
    VECTORS.write_bytes(data)
    acc = (rec["pred"] == rec["label"]).mean() * 100
    print(f"reference/build/vectors.bin: {len(rec):,} imágenes de test, {len(data) / 1e6:.1f} MB "
          f"(accuracy de forward_int: {acc:.2f}%)")


if __name__ == "__main__":
    main()
