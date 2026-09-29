"""Pizarra de 28×28 para probar la MLP a mano: se dibuja un dígito con el mouse y se ve qué
predice la red, con los pesos float32 o solo con enteros.

Uso (desde la raíz del repo):
    .venv/bin/python model/draw.py

Usa tkinter, que viene con Python; en Debian/Ubuntu es un paquete aparte (python3-tk).

Controles: click izquierdo dibuja, click derecho borra, Espacio limpia y M cambia de red.

Las dos redes son las de referencia del proyecto, sin copias: forward() con los pesos de
mlp.npz y forward_int() con los enteros congelados de mlp_int8.npz. Las dos reciben la misma
entrada, como en MNIST: 784 píxeles uint8, fila por fila, con trazo = 255 y fondo = 0.
"""
import tkinter as tk
from tkinter import font as tkfont

import numpy as np

from forward import forward, load_weights
from quantize import forward_int, load_quantized

SIZE = 28
CELL = 16  # píxeles de pantalla por píxel de la imagen
# Pincel: intensidad plena hasta CORE píxeles del centro, que se desvanece hasta EDGE. Deja un
# trazo de ~3 px con bordes grises, como los de MNIST.
CORE, EDGE = 0.8, 1.8
STEP = 0.25  # entre dos lecturas del mouse se pinta cada 1/4 de píxel, para no dejar huecos

# Centro de cada píxel, en unidades de píxel: el píxel (fila r, columna c) ocupa
# [c, c+1] × [r, r+1].
_ROWS, _COLS = np.mgrid[0:SIZE, 0:SIZE] + 0.5

# Paleta de las figuras de la guía: azul = float, naranja = entero. El texto va en tinta.
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, GUIDE, TRACK = "#e1e0d9", "#c3c2b7", "#f0efec"
MODES = {  # modo → (nombre, color de las barras, fondo del botón elegido)
    "float": ("float32", "#2a78d6", "#e3ecf7"),
    "int": ("entera (int8)", "#eb6834", "#faeae3"),
}
# Gris de cada valor de píxel: de la superficie (0) a la tinta (255).
_SURF_RGB, _INK_RGB = np.array([0xfc, 0xfc, 0xfb]), np.array([0x0b, 0x0b, 0x0b])
SHADES = ["#%02x%02x%02x" % tuple(np.round(_SURF_RGB + (_INK_RGB - _SURF_RGB) * v / 255)
                                   .astype(int)) for v in range(256)]

# Geometría de la ventana, en píxeles de pantalla.
MARGIN = 24
GX, GY = MARGIN, 56                  # esquina de la grilla
PX = GX + SIZE * CELL + 44           # columna del panel derecho
PANEL_W, BAR_W, ROW_H = 340, 170, 27
W, H = PX + PANEL_W + MARGIN, GY + SIZE * CELL + 56
SEGMENTS = {"float": (PX, PX + 124), "int": (PX + 124, PX + 248)}  # botones de modo
CLEAR = (PX + 262, PX + PANEL_W)                                  # botón Limpiar


def paint(img, x, y, erase=False):
    """Un toque del pincel centrado en (x, y), en unidades de píxel. Modifica img."""
    d = np.hypot(_COLS - x, _ROWS - y)
    if erase:
        img[d < EDGE] = 0
    else:
        v = np.round(np.clip((EDGE - d) / (EDGE - CORE), 0, 1) * 255).astype(np.uint8)
        np.maximum(img, v, out=img)  # repasar un trazo no lo oscurece más allá de 255


def stroke(img, p0, p1, erase=False):
    """Pinta el segmento p0 → p1 con toques cada STEP píxeles."""
    n = max(1, int(np.ceil(np.hypot(p1[0] - p0[0], p1[1] - p0[1]) / STEP)))
    for t in np.linspace(0, 1, n + 1):
        paint(img, p0[0] + t * (p1[0] - p0[0]), p0[1] + t * (p1[1] - p0[1]), erase)


def predict(img, w, q):
    """Corre las dos redes sobre la imagen uint8 (28, 28): devuelve los logits float y acc2."""
    x = img.reshape(1, -1)
    _, z = forward(x.astype(np.float32) / 255, w)  # la float divide por 255, como train.py
    _, _, acc2 = forward_int(x, q)                  # la entera usa el píxel crudo
    return z[0], acc2[0]


def describe(img):
    """Lo que más pesa para la red: cuánta tinta hay, dónde está su centro y qué tamaño
    tiene."""
    ink = img.astype(np.float64)
    rows, cols = np.nonzero(img)
    center = (ink.sum(axis=1) @ np.arange(SIZE) / ink.sum(),
              ink.sum(axis=0) @ np.arange(SIZE) / ink.sum())
    box = (rows.max() - rows.min() + 1, cols.max() - cols.min() + 1)
    return len(rows), center, box


def num(x, dec=0, sign=False):
    """Número con formato argentino: punto de miles y coma decimal."""
    s = f"{x:{'+' if sign else ''},.{dec}f}"
    return s.replace(",", "_").replace(".", ",").replace("_", ".")


class DrawApp:
    def __init__(self, root):
        self.w = load_weights()
        self.q, self.scales = load_quantized()
        self.img = np.zeros((SIZE, SIZE), np.uint8)
        self.mode = "float"
        self.last = None  # último punto del trazo en curso, en unidades de píxel

        family = tkfont.nametofont("TkDefaultFont").actual()["family"]
        self.font = {name: tkfont.Font(family=family, size=size, weight=weight)
                     for name, size, weight in [("title", 14, "bold"), ("big", 48, "bold"),
                                                ("body", 10, "normal"), ("bold", 10, "bold"),
                                                ("small", 9, "normal")]}
        root.title("MLP 784→32→10 · pizarra 28×28")
        root.resizable(False, False)
        self.cv = tk.Canvas(root, width=W, height=H, bg=SURFACE, highlightthickness=0,
                            cursor="crosshair")
        self.cv.pack()
        self._build()

        self.cv.bind("<ButtonPress-1>", lambda e: self.press(e, erase=False))
        self.cv.bind("<B1-Motion>", lambda e: self.drag(e, erase=False))
        self.cv.bind("<ButtonPress-3>", lambda e: self.press(e, erase=True))
        self.cv.bind("<B3-Motion>", lambda e: self.drag(e, erase=True))
        root.bind("<space>", lambda e: self.clear())
        for key in ("m", "M"):
            root.bind(f"<KeyPress-{key}>", lambda e: self.toggle_mode())
        self.set_mode(self.mode)

    def _text(self, x, y, text="", font="body", fill=INK, anchor="w", **kw):
        return self.cv.create_text(x, y, text=text, font=self.font[font], fill=fill,
                                   anchor=anchor, **kw)

    def _build(self):
        cv, grid_end = self.cv, GY + SIZE * CELL
        self._text(GX, 28, "Dibujá un dígito", font="title")
        self._text(W - MARGIN, 28, "izq.: dibujar · der.: borrar · Espacio: limpiar · M: red",
                   font="small", fill=MUTED, anchor="e")

        # Grilla: un rectángulo por píxel, pintado con su valor.
        self.cells = np.array([[cv.create_rectangle(GX + c * CELL, GY + r * CELL,
                                                    GX + (c + 1) * CELL, GY + (r + 1) * CELL,
                                                    fill=SURFACE, outline=GRID)
                                for c in range(SIZE)] for r in range(SIZE)])
        cv.create_rectangle(GX, GY, GX + SIZE * CELL, grid_end, outline=GUIDE)
        # Guía: la caja de 20×20 y el centro (fila 14, columna 14) de los dígitos de MNIST.
        cv.create_rectangle(GX + 4 * CELL, GY + 4 * CELL, GX + 24 * CELL, GY + 24 * CELL,
                            outline=GUIDE, dash=(4, 4))
        cx, cy = GX + 14.5 * CELL, GY + 14.5 * CELL
        cv.create_line(cx - 6, cy, cx + 7, cy, fill=GUIDE)
        cv.create_line(cx, cy - 6, cx, cy + 7, fill=GUIDE)
        self.stats = self._text(GX, grid_end + 18, font="small", fill=INK2)
        self._text(GX, grid_end + 36, "MNIST: ~150 px con trazo · centro en (14; 14) · "
                   "lado mayor de 20 px (la caja punteada)", font="small", fill=MUTED)

        # Botones: los dos modos y Limpiar.
        self.segments = {}
        for key, (x0, x1) in SEGMENTS.items():
            rect = cv.create_rectangle(x0, GY, x1, GY + 30)
            label = self._text((x0 + x1) / 2, GY + 15, MODES[key][0], anchor="center")
            self.segments[key] = (rect, label)
        cv.create_rectangle(CLEAR[0], GY, CLEAR[1], GY + 30, outline=GUIDE)
        self._text(sum(CLEAR) / 2, GY + 15, "Limpiar", fill=INK2, anchor="center")

        # Predicción: el dígito grande y, al lado, las dos redes y el margen.
        self._text(PX, GY + 52, "predicción", font="small", fill=MUTED)
        self.big = self._text(PX, GY + 96, font="big")
        self.lines = [self._text(PX + 70, GY + 64 + 18 * i) for i in range(4)]

        # Tabla de salidas: una fila por dígito.
        top = GY + 158
        self._text(PX, top, "dígito", font="small", fill=MUTED)
        self._text(PX + 36, top, "softmax", font="small", fill=MUTED)
        self.raw_head = self._text(PX + PANEL_W, top, font="small", fill=MUTED, anchor="e")
        self.rows = []
        for d in range(10):
            y = top + 24 + d * ROW_H
            cv.create_rectangle(PX + 36, y - 5, PX + 36 + BAR_W, y + 5, fill=TRACK, outline="")
            self.rows.append({
                "label": self._text(PX + 12, y, str(d), anchor="center"),
                "bar": cv.create_rectangle(PX + 36, y - 5, PX + 36, y + 5, outline=""),
                "pct": self._text(PX + 36 + BAR_W + 56, y, font="small", fill=INK2, anchor="e"),
                "raw": self._text(PX + PANEL_W, y, font="small", fill=INK2, anchor="e"),
            })
        self._text(PX, top + 24 + 10 * ROW_H, "La red elige el valor más alto; el % (softmax) "
                   "es solo para mostrar.", font="small", fill=MUTED, width=PANEL_W)

    # --- Eventos ---

    def press(self, e, erase):
        gx, gy = (e.x - GX) / CELL, (e.y - GY) / CELL
        if 0 <= gx < SIZE and 0 <= gy < SIZE:
            self.last = (gx, gy)
            self.edit(lambda img: paint(img, gx, gy, erase))
            return
        self.last = None
        if erase or not GY <= e.y <= GY + 30:
            return
        for key, (x0, x1) in SEGMENTS.items():
            if x0 <= e.x <= x1:
                self.set_mode(key)
        if CLEAR[0] <= e.x <= CLEAR[1]:
            self.clear()

    def drag(self, e, erase):
        if self.last is None:  # el click empezó fuera de la grilla
            return
        p = ((e.x - GX) / CELL, (e.y - GY) / CELL)
        self.edit(lambda img: stroke(img, self.last, p, erase))
        self.last = p

    def clear(self):
        self.edit(lambda img: img.fill(0))

    def toggle_mode(self):
        self.set_mode("int" if self.mode == "float" else "float")

    def set_mode(self, mode):
        self.mode = mode
        for key, (rect, label) in self.segments.items():
            _, color, wash = MODES[key]
            on = key == mode
            self.cv.itemconfig(rect, fill=wash if on else SURFACE,
                               outline=color if on else GUIDE, width=1.5 if on else 1)
            self.cv.itemconfig(label, fill=INK if on else INK2,
                               font=self.font["bold" if on else "body"])
        self.cv.tag_raise(self.segments[mode][0])  # que el borde elegido quede entero
        self.cv.tag_raise(self.segments[mode][1])
        for row in self.rows:
            self.cv.itemconfig(row["bar"], fill=MODES[mode][1])
        self.cv.itemconfig(self.raw_head, text="logit" if mode == "float" else "acc2 (int32)")
        self.show()

    # --- Dibujo y predicción ---

    def edit(self, change):
        before = self.img.copy()
        change(self.img)
        for r, c in zip(*np.nonzero(self.img != before)):
            self.cv.itemconfig(self.cells[r, c], fill=SHADES[self.img[r, c]])
        self.show()

    def show(self):
        cv = self.cv
        if not self.img.any():
            cv.itemconfig(self.big, text="—", fill=MUTED)
            for item, text in zip(self.lines, ["grilla vacía", "", "", ""]):
                cv.itemconfig(item, text=text, fill=MUTED, font=self.font["body"])
            for row in self.rows:
                cv.coords(row["bar"], PX + 36, 0, PX + 36, 0)
                cv.itemconfig(row["pct"], text="")
                cv.itemconfig(row["raw"], text="")
                cv.itemconfig(row["label"], font=self.font["body"])
            cv.itemconfig(self.stats, text="")
            return

        z, acc2 = predict(self.img, self.w, self.q)
        preds = {"float": int(z.argmax()), "int": int(acc2.argmax())}
        if self.mode == "float":
            real, raw = z, [num(v, 2, sign=True) for v in z]
        else:  # el acumulador pasado a real, solo para el softmax y el margen
            real, raw = acc2 * self.scales["acc2"], [num(v, sign=True) for v in acc2]
        winner = preds[self.mode]
        p = np.exp(real - real.max())
        p /= p.sum()
        second = np.sort(real)[-2]

        cv.itemconfig(self.big, text=str(winner), fill=INK)
        same = preds["float"] == preds["int"]
        texts = [f"float32  →  {preds['float']}", f"entera  →  {preds['int']}",
                 "las dos coinciden" if same else "¡difieren! (casi empate)",
                 f"margen: {num(real[winner] - second, 2)}"]
        for i, (item, text) in enumerate(zip(self.lines, texts)):
            bold = i == 2 and not same
            cv.itemconfig(item, text=text, fill=INK if bold else INK2,
                          font=self.font["bold" if bold else "body"])

        for d, row in enumerate(self.rows):
            y = cv.coords(row["label"])[1]
            cv.coords(row["bar"], PX + 36, y - 5, PX + 36 + max(p[d] * BAR_W, 0), y + 5)
            cv.itemconfig(row["pct"], text=f"{num(p[d] * 100, 1)} %")
            cv.itemconfig(row["raw"], text=raw[d])
            cv.itemconfig(row["label"], font=self.font["bold" if d == winner else "body"])

        n, (r, c), (bh, bw) = describe(self.img)
        cv.itemconfig(self.stats, text=f"Tu dibujo: {n} px con trazo · centro en "
                                       f"({num(r, 1)}; {num(c, 1)}) · caja de {bh}×{bw}")


def main():
    root = tk.Tk()
    DrawApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
