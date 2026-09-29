# riscv-neural-accel

Una NPU (unidad de procesamiento neuronal) acoplada a un core RISC-V, el PicoRV32, en una FPGA Arty A7-100T. El proyecto acelera la inferencia de una red neuronal en tres niveles, siempre con la misma red, el mismo core y la misma FPGA:

1. **C puro** en el core: el baseline.
2. **Una instrucción custom** (PCPI) que hace 4 multiplicaciones-acumulaciones por instrucción.
3. **Un acelerador memory-mapped** (la NPU) que calcula capas enteras.

La red es un perceptrón multicapa 784→32→10 que clasifica los dígitos de MNIST, cuantizado a enteros de 8 bits.

Trabajo Final Integrador de Ingeniería en Computación, Universidad Nacional de Córdoba.

## Estado

| Etapa | Qué | Estado |
|---|---|---|
| 1–2 | La red, entrenada en PyTorch y replicada en NumPy | terminada |
| 3 | Cuantización a enteros y referencia en C, idéntica bit a bit a Python | terminada |
| 4 | SoC con PicoRV32 en la FPGA y medición del baseline | siguiente |
| 5 | Instrucción custom (PCPI) | pendiente |
| 6 | NPU memory-mapped | pendiente |
| 9 | Demo: un dígito dibujado en un panel táctil → NPU → UART | pendiente |
| 8 y 7 | Generación automática del firmware; CNN (opcional) | pendiente |

El detalle de cada etapa, con sus checkpoints, está en [`docs/plan.md`](docs/plan.md).

**Resultados hasta ahora** (las 10.000 imágenes de test de MNIST):
- Accuracy de 98,09% en float y 98,13% con enteros: la diferencia es ruido.
- La referencia en C calcula exactamente los mismos 750.000 enteros intermedios que Python.
- Con la misma receta de entrenamiento y 3 semillas, el promedio en float es 97,91%.

## Estructura

```
docs/            plan de trabajo y marco teórico
model/           la red en Python: entrenamiento, verificación, cuantización y exportación a C
  artifacts/     el modelo congelado, en todos sus formatos: float32 (mlp.*) y enteros (mlp_int8.*)
reference/       la inferencia entera en C, verificada bit a bit contra Python
```

Con las próximas etapas se suman `firmware/` (programas para el PicoRV32), `rtl/` (el hardware, en Verilog), `sim/` (testbenches) y `fpga/` (pines y scripts de Vivado).

## Cadena de verificación

Cada implementación se valida contra la anterior, sobre las 10.000 imágenes de test:

| Implementación | Aritmética | Se compara contra | Criterio |
|---|---|---|---|
| `model/train.py` (PyTorch) | float32 | — | — |
| `model/forward.py` (NumPy) | float32 | PyTorch | Mismas predicciones y diferencias menores a 1e-4 |
| `model/quantize.py` (NumPy) | enteros | float32 | Accuracy sin cambio medible |
| `reference/mlp.c` (C) | enteros | `quantize.py` | Idéntico bit a bit, en todos los valores intermedios |
| PicoRV32, PCPI y NPU (próximas etapas) | enteros | `reference/mlp.c` | Idéntico bit a bit |

## Cómo reproducirlo

Hace falta Python 3.12, y gcc y make para la parte en C. Desde la raíz del repo:

```bash
python3 -m venv .venv
.venv/bin/pip install -r model/requirements.txt
```

| Comando | Qué hace |
|---|---|
| `.venv/bin/python model/forward.py` | Checkpoint de la Etapa 2: el forward en NumPy contra PyTorch |
| `.venv/bin/python model/quantize.py` | Cuantiza la red y compara la accuracy float contra la entera |
| `make -C reference check` | Checkpoint de la Etapa 3: el C contra Python, valor por valor |
| `.venv/bin/python model/draw.py` | Pizarra para dibujar un dígito y ver qué predice la red, en float o en enteros |
| `.venv/bin/python model/train.py` | Reentrena la red (~1,5 min en CPU) |

- MNIST se descarga solo la primera vez, en `model/data/`.
- `model/artifacts/` está versionado y es la referencia del proyecto. `train.py` y `quantize.py` reescriben esos archivos, y en otra máquina pueden salir levemente distintos.
- La pizarra usa tkinter; en Debian y Ubuntu viene en el paquete `python3-tk`.

## Documentación

- [`docs/plan.md`](docs/plan.md): el plan de trabajo, etapa por etapa, con sus checkpoints y las decisiones tomadas.
- [`docs/marco_teorico.md`](docs/marco_teorico.md): el marco teórico del informe.
