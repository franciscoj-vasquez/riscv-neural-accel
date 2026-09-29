# Plan de trabajo — NPU acoplada a RISC-V en FPGA

## Cambios respecto al roadmap original

1. **Reordenamiento de prioridades**: la Etapa 9 (demo funcional) se ataca apenas cierra la Etapa 6, sin esperar a la Etapa 7 (CNN). La CNN pasa a ser un estiramiento opcional, al final, solo si sobra tiempo después de todo lo demás.
   **Por qué:** el propio roadmap original ya dice que la Etapa 9 "requiere 6 (idealmente 7)" — la 7 nunca fue un bloqueante real, solo deseable. Un demo funcionando de punta a punta pesa más en una defensa de PI que una mejora incremental de accuracy vía CNN, y además así, si el tiempo se acorta, el proyecto igual cierra con una historia completa (MLP + NPU + demo en vivo) en vez de quedar con la CNN sí pero sin demo.

2. **Cambio de periférico en la Etapa 9**: se reemplaza la cámara OV7670 por un **panel táctil resistivo por SPI** (tipo ILI9341+XPT2046) sobre el que se dibuja el dígito directamente.
   **Por qué:** casi todo el pipeline de preprocesamiento que el plan original necesitaba (recorte de ROI, downscale por bloques, inversión de polaridad, umbral adaptativo de contraste) existe únicamente para compensar los problemas de una foto de cámara real (iluminación, tinta oscura sobre papel claro, mal centrado). Dibujando directo sobre una grilla lógica de 28×28, esos problemas no existen — el trazo ya nace binario, sin luz ambiente de por medio, y cae directo en el formato que la red espera. De paso, la interfaz de hardware (SPI) es bastante más simple que la de la cámara (captura paralela por PCLK + FIFO de cruce de dominio de reloj).

3. **Etapa 8 (toolchain) confirmada al final, sin cambios de contenido — pero con la razón explícita.** Automatizar la generación de firmware antes de que la interfaz de la NPU esté estable y probada a mano sería automatizar un formato que todavía no existe, y obligaría a reescribir el generador cada vez que cambie el diseño del hardware durante las Etapas 4-7. Además, la Etapa 8 **no depende de la 7 ni de la 9** — solo de la 6 (y de la 3, para el esquema de cuantización) — así que puede hacerse en paralelo con la Etapa 9 o inmediatamente después, según convenga.

**Pendiente:** el alcance formal que quedó redactado en la Solicitud de PI (Título, Objetivo, Anexo, Gantt) todavía dice "Etapas 1 a 7" con la cámara mencionada en el Anexo como fuera de alcance. Falta actualizar ese documento para reflejar este plan (demo con panel táctil en vez de cámara, CNN como estiramiento en vez de compromiso) — pendiente de una próxima sesión.

---

## Encabezado del roadmap original (sin cambios)

**Caso de uso vehicular:** clasificación de dígitos MNIST (MLP primero, CNN después, esta última ahora como estiramiento opcional).

**Duración estimada:** 4–6 meses para el núcleo (Etapas 1–6 + demo). La CNN y el toolchain, si se llega, se suman después.

**Regla de oro transversal:** nunca avanzar de etapa sin cumplir el checkpoint bit-exacto. Los mismatches numéricos de cuantización arrastrados son el bug más costoso de debuggear en RTL. Cada nivel necesita su oráculo.

---

## Etapa 1 — Fundamentos mínimos de redes neuronales

**Duración:** 1–2 semanas · **Entorno:** PC solamente

### Conceptos a incorporar
- Neurona = multiply-accumulate + bias + función de activación (ReLU).
- Capa densa (fully connected) = multiplicación matriz-vector.
- Forward pass (inferencia): la única parte que va a hardware.
- Intuición de backpropagation (solo intuición: el entrenamiento queda en la PC, no hace falta dominar la matemática).

### Tareas
1. Implementar una MLP **784→32→10** a mano en NumPy, solo forward pass, con pesos random.
2. Contar los MACs por capa (784×32 + 32×10 ≈ 25K MACs por inferencia — más precisamente, 25.408 pesos = 25.408 MACs por inferencia; los bias son solo 42, uno por neurona, no uno por peso).

### Recursos
- 3Blue1Brown, serie de redes neuronales (primeros 4 videos).
- "Neural Networks from Scratch", capítulos 1–2.
- Evitar cursos largos completos: overkill para inferencia.

### Checkpoint de salida
Poder explicar qué operaciones ejecuta una MLP en inferencia y cuántos MACs cuesta cada capa.

---

## Etapa 2 — Entrenamiento de la red

**Duración:** 1 semana · **Entorno:** PC solamente

### Conceptos a incorporar
Dataset / train / test split, función de loss, epochs, accuracy. Solo lo operativo.

### Tareas
1. Entrenar la MLP 784→32→10 con MNIST en PyTorch (~50 líneas). **Target: >97% accuracy.**
2. Correr inferencia sobre 10 imágenes sueltas imprimiendo activaciones intermedias, para familiarizarse con los tensores.
3. Exportar los pesos entrenados y cargarlos en el forward pass NumPy de la Etapa 1.

### Checkpoint de salida
El forward pass NumPy da las **mismas predicciones** que PyTorch con los pesos entrenados, en las 10.000 imágenes de test, y las activaciones y los logits difieren solo por redondeo (menos de 1e-4). En float no se puede exigir igualdad bit a bit: sumar en otro orden cambia el redondeo. La igualdad bit a bit se exige a partir de la Etapa 3, donde todas las cuentas son enteras y el orden de la suma no altera el resultado.

---

## Etapa 3 — Cuantización

**Duración:** 1–2 semanas · **Entorno:** PC solamente

### Conceptos a incorporar (los centrales del proyecto)
- Cuantización float32 → int8: escala y zero-point.
- Cuantización simétrica vs. asimétrica.
- Requantización entre capas: multiplicador de punto fijo + shift.
- Acumulación en int32, saturación.
- Pesos y activaciones → int8/uint8; bias y acumulador → int32 (no todo pasa a 8 bits por igual).

Este es el concepto que separa "saber de NN" de "saber de NN en hardware embebido".

### Tareas
1. Leer el esquema estándar: Jacob et al. (gemmlowp) o documentación de cuantización de TFLite.
2. Cuantizar la MLP con post-training quantization. Para una red así de chica, hacerlo **a mano** es más didáctico que usar Brevitas (que queda como referencia; Brevitas hace *quantization-aware training*, una técnica más pesada que no hace falta para esta escala).
3. Escribir la **inferencia int8 completa en C puro**: entrada uint8, pesos int8, acumuladores int32, requantización entre capas. Sin floats en ningún lado.
4. Validar bit-exacto contra la versión Python cuantizada, sobre las 10.000 imágenes de test.

### Checkpoint de salida
**Golden reference en C**, bit-exacta contra Python, con accuracy ≈ igual a float32 (la caída esperada por cuantización, a esta escala, es mínima — bien fundamentado en la literatura, pero la certeza real la da este mismo checkpoint, no una estimación teórica). Este archivo es el oráculo de todo el proyecto.

---

## Etapa 4 — SoC RISC-V base en FPGA

**Duración:** 2–3 semanas · **Entorno:** FPGA

### Alcance
Plataforma base sin NN: **PicoRV32** (core single-issue, in-order) + memoria + UART + contador de ciclos.

### Tareas
1. SoC mínimo en la FPGA: hello world por UART.
2. Toolchain: riscv-gcc, linker script, startup code.
3. Portar la golden reference C y correrla en el core.
4. Medir ciclos por inferencia con el contador → **este número es el baseline**.
5. Perfilar: confirmar que el grueso del tiempo está en los loops MAC.

### Checkpoint de salida
Inferencia MNIST corriendo en RISC-V en FPGA, con ciclos por inferencia medidos y documentados.

---

## Etapa 5 — Primera aceleración: instrucción custom (PCPI)

**Duración:** 2–3 semanas · **Entorno:** simulación + FPGA

### Conceptos de hardware a incorporar
Interfaz de coprocesador (**PCPI = Pico Co-Processor Interface**, mecanismo propio de PicoRV32): handshake (`pcpi_valid`/`pcpi_insn`/`pcpi_rs1`/`pcpi_rs2`/`pcpi_wait`/`pcpi_ready`/`pcpi_wr`/`pcpi_rd`), decodificación de instrucción custom (espacio de opcode `custom-0`, reservado por el propio estándar RISC-V), stall del pipeline mientras se espera el resultado.

No es una arquitectura de "dos cores": sigue siendo un solo core con un solo PC, extendido con una unidad de ejecución externa (misma idea que un coprocesador matemático clásico tipo FPU) — no hay ejecución concurrente, el pipeline se detiene mientras el PCPI calcula.

Es una técnica de tipo **SIMD / sub-word parallelism** (4 valores int8 empaquetados en un registro de 32 bits, una instrucción los procesa juntos) — no superescalar (PicoRV32 es single-issue, no tiene lógica de múltiple emisión que aprovechar).

### Diseño
Instrucción **dot-product int8×4**: toma dos registros de 32 bits (4 elementos int8 empaquetados en cada uno), ejecuta 4 MACs y acumula. El acumulador conviene que viva **dentro del propio bloque PCPI** (no en el banco de registros del RISC-V), con una instrucción separada para leer el resultado final — la interfaz PCPI no expone el valor viejo de `rd` como entrada, así que acumular vía el banco de registros no es directo.

### Tareas
1. RTL del bloque PCPI.
2. Testbench (CocoTB o Verilator) contra un modelo de referencia en Python.
3. Integración al core.
4. Reescribir el loop MAC del firmware usando la instrucción — **no se modifica el compilador**: se usa la directiva `.insn` de GNU as para emitir el word de 32 bits exacto desde una macro de C con inline assembly, y esa macro se llama en el firmware como si fuera una función normal.
5. Validar bit-exacto contra la golden reference y medir speedup.

### Resultado esperable
Speedup de **3–6×** sobre el baseline, misma accuracy exacta (el speedup medido ya incluye cualquier overhead del handshake PCPI — se mide de punta a punta con el contador de ciclos real). Riesgo a vigilar: que el empaquetado/desempaquetado de los 4 int8 en el registro (si los datos no están ya guardados empaquetados en memoria) se coma parte de la ganancia.

### Checkpoint de salida
Mismo resultado bit-exacto, N× menos ciclos, mecánica de extensión del core incorporada.

---

## Etapa 6 — La NPU: acelerador memory-mapped en el bus

**Duración:** 4–8 semanas · **Entorno:** simulación + FPGA · **El corazón del proyecto**

### Conceptos a incorporar
- Acelerador **memory-mapped**: se le habla con `sw`/`lw` normales a un rango de direcciones (igual que la UART de la Etapa 4), no con una instrucción nueva — mecanismo de extensión totalmente distinto al de la Etapa 5.
- Buffers en BRAM para pesos, activaciones de entrada y salida (BRAM = memoria de bloque dedicada de la FPGA, necesaria para poder leer varios valores por ciclo y alimentar el array de MACs).
- FSM (máquina de estados finitos) de control por capa: corre sola una vez que se le da "start", sin que la CPU la guíe paso a paso.
- Sincronización CPU–acelerador: polling primero (la CPU pregunta en loop si terminó), interrupción después.

### Arquitectura v1
- Array de **8 o 16 MACs int8 en paralelo** (paralelismo espacial real, no solo empaquetado en una instrucción como en la Etapa 5 — de ahí el salto de speedup). Es una decisión de diseño abierta: más MACs = más throughput pero más DSPs/LUTs/ancho de banda de BRAM necesario; para esta red tan chica (32 y 10 neuronas en las dos capas), 8-16 ya está bien calibrado, ir mucho más allá no ayudaría tanto.
- BRAMs dedicadas: pesos / activaciones entrada / activaciones salida.
- Registros: dimensiones de la capa, punteros, start, done.
- Modelo de ejecución: la CPU carga datos y configuración, dispara, espera done, lee resultados. **Una capa entera por invocación.**

### Flujo de trabajo
1. Modelo de referencia funcional del acelerador en Python (no hace falta ciclo a ciclo).
2. RTL + testbench por bloque: datapath, FSM, interfaz de bus.
3. Simulación integrada del SoC completo (Verilator): firmware + acelerador matcheando la golden reference.
4. FPGA: síntesis, timing closure, medición.

### Resultado esperable
Speedup de **20–50×** sobre baseline según paralelismo elegido. CPU solo orquesta; la NPU ejecuta las capas.

### Checkpoint de salida
Inferencia completa con la NPU, bit-exacta, medida en ciclos y recursos (LUT/FF/BRAM/DSP).

**A partir de acá se abren tres caminos que solo requieren esta etapa cerrada, independientes entre sí (ver más abajo el orden de prioridad elegido).**

---

## Etapa 9 — Demo final: panel táctil → NPU → UART *(reordenada: se ataca apenas cierra la Etapa 6 — prioridad 1)*

> **Revisión de viabilidad hecha en detalle antes de comprar nada** (interfaz por interfaz: eléctrica, protocolo, formato de imagen). Veredicto: viable y sigue siendo mejora neta sobre la cámara, pero con más piezas reales de trabajo que la primera pasada — quedan incorporadas abajo.

**Duración estimada (revisada):** 2,5–3,5 semanas · **Entorno:** FPGA · **Prerequisito:** Etapa 6 cerrada (ya no se espera a la Etapa 7)

### Objetivo
Dibujar un dígito sobre un panel táctil y que salga por UART: `detectado: 7`.

### Hardware
**Panel táctil resistivo por SPI con LCD integrado** (tipo ILI9341+XPT2046, 2.4"-2.8", ~240×320) — la pantalla pasó de "opcional" a **recomendada**: sin ella la persona dibuja "a ciegas", sin ver dónde cae su trazo respecto de la grilla, lo cual es un riesgo real de que el dígito quede mal encuadrado por falta de feedback visual, no por falla de la red. Se conecta a uno de los puertos Pmod de la Arty (el pinout de Pmod ya expone CS/MOSI/MISO/SCK) o al header tipo Arduino/chipKIT.

**Voltaje:** los chips ILI9341/XPT2046 son nativamente 3.3V. Las advertencias de daño que circulan en tutoriales son para el caso de meterles 5V desde un Arduino Uno — la Arty, al ser ya 3.3V, no corre ese riesgo. El riesgo residual (si el módulo trae un divisor resistivo pensado para 5V→3.3V) es, como mucho, que la señal quede algo subalimentada y la comunicación no sea confiable — no daño de hardware. Se resuelve de raíz sumando un **level shifter bidireccional** (BSS138, ~USD 1-2) a la compra, como resguardo barato.

**Periférico nuevo en el SoC:** el PicoRV32 no trae SPI ni GPIO — hace falta agregar un periférico chico de GPIO mapeado en memoria y hacer SPI por bit-banging en firmware (prender/apagar CS/MOSI/SCK a mano, leer MISO). Comparable en esfuerzo al UART que ya se construye en la Etapa 4 — no es trabajo gratis, pero es chico y de bajo riesgo.

### Pipeline de captura (más piezas que la primera idea, todas simples y bien documentadas)
1. **Lectura de posición por SPI**: comando al XPT2046, lectura de 12 bits crudos del ADC (canal X o canal Y). Protocolo simple, mucho menos riesgo que la SCCB + captura paralela + cruce de dominio de reloj que hubiera hecho falta para la cámara.
2. **Calibración**: la lectura cruda del ADC (0-4095) *no* corresponde linealmente a la pantalla — hace falta un paso de calibración (tocar las 4 esquinas conocidas una vez, calcular la transformación lineal) antes de poder mapear a píxeles. Estándar y bien documentado, pero es un paso real, no automático.
3. **Interpolación entre muestras**: si se muestrea la posición a baja frecuencia, un trazo rápido puede saltear celdas y dejar el dibujo punteado — hace falta interpolar linealmente entre dos lecturas consecutivas y rellenar las celdas intermedias.
4. **Downscale por promediado, no mapeo directo a 28×28**: mejor dibujar a la resolución nativa del panel (o una región del mismo) y aplicar *un* paso de downscale por promediado de bloques al final — la misma técnica que ya estaba prevista para la cámara, aplicada a un lienzo chico y controlado en vez de una foto con ruido. Da bordes más parecidos al antialiasing real de MNIST que un mapeo directo y duro a 28×28. No es trabajo extra: es reusar una pieza que el plan original ya iba a necesitar.
5. **Centrado por centro de masa: se mantiene, no es opcional.** La red que corre en este demo es la **MLP** (la CNN quedó como estiramiento al final) y una MLP no tiene ninguna tolerancia arquitectónica a un mal centrado — este paso sigue haciendo falta.

Lo que sí se elimina de verdad, y es la ganancia real de este cambio: el recorte de región de interés sobre una escena más grande, la inversión de polaridad, y el umbral adaptativo por iluminación — esos tres desaparecen porque el trazo nace limpio y controlado, sin cámara ni luz ambiente de por medio.

### Paso intermedio recomendado (antes de cablear el panel físico)
Probar calibración, interpolación, downscale y centrado con secuencias de coordenadas sintéticas enviadas por UART, antes de depender del hardware táctil real — separa el debug de la lógica del debug del hardware, igual que en el plan original, solo que acá el hardware de por sí ya es menos riesgoso.

### Ajuste del entrenamiento
El salto de dominio entre MNIST y un trazo dibujado con stylus es más chico que el de una foto de cámara (sin iluminación ni ruido óptico), así que el augmentation es menos urgente que en el plan original — sigue siendo un seguro barato (rotaciones leves, traslaciones), y opcionalmente se puede sumar algún trazo propio dibujado en el panel al training set.

### Desglose de tiempos (estimado, a validar en la práctica)
- Periférico GPIO + SPI bit-banged + lectura del XPT2046 + calibración: ~1-1.5 semanas.
- Driver básico del ILI9341 (inicialización + dibujar el trazo en vivo): ~3-5 días.
- Pipeline de preprocesamiento (interpolación, downscale, centrado): ~3-5 días.
- Ajuste/prueba con trazos reales: ~1 semana.

### Checkpoint de salida
Dígito dibujado en el panel táctil → NPU → clasificación correcta por UART, en vivo.

### Lista de compra
- Módulo ILI9341 + XPT2046 con LCD, 2.4" o 2.8" (ej. HiLetgo).
- Lápiz/stylus (verificar si el listado puntual lo incluye — si no, cualquier stylus resistivo genérico sirve).
- Level shifter bidireccional (BSS138 o similar) como resguardo de tensión.
- Cables jumper para el cableado al puerto Pmod.

---

## Etapa 8 — Cierre end-to-end: toolchain automatizado *(sin cambios de contenido — al final, en paralelo o después de la Etapa 9)*

**Duración:** 2–3 semanas · **Entorno:** PC

**Nota de dependencia:** esta etapa solo requiere la Etapa 6 (interfaz de la NPU ya estable) y la Etapa 3 (esquema de cuantización definido) — no depende de la Etapa 9 ni de la Etapa 7. Se construye al final a propósito: automatizar la traducción modelo→firmware antes de que el formato de esa traducción esté probado y estable sería automatizar algo que todavía puede cambiar, duplicando el mantenimiento durante las etapas más riesgosas (4-7). Para uso personal, nada impide ir usando scripts chicos y descartables desde antes (por ejemplo, para no tipear a mano el array de 25.408 pesos) — lo que se deja para el final es el generador general y prolijo.

### Objetivo
Convertir el proyecto de "un demo" a "un flujo".

### Tareas
1. Script Python (`compile_model.py`) que toma el modelo cuantizado (`modelo.pt`) y **genera automáticamente** el firmware:
   - Headers con pesos y parámetros de cuantización (sacados directo de los tensores del modelo, no re-tipeados a mano).
   - Descriptores de capas (dimensiones, tipo, activación — sacados de la arquitectura real del modelo).
   - Secuencia de llamadas a la NPU (una por capa, en orden, generada según cuántas capas tenga el modelo).
2. Uso: `python compile_model.py modelo.pt` → código C listo para compilar con `riscv-gcc`.
3. Referencias de inspiración: microTVM (Apache TVM), FINN (sin adoptarlos; un generador propio, chico y específico a esta NPU, es más didáctico que integrar una herramienta de producción).

### Entregable final del proyecto
Tabla comparativa sobre la misma red, mismo core, misma FPGA:

| Versión | Ciclos/inferencia | Speedup | LUT | FF | BRAM | DSP | Accuracy |
|---|---|---|---|---|---|---|---|
| C puro (baseline) | | 1× | | | | | |
| Instrucción custom PCPI | | | | | | | |
| NPU en bus | | | | | | | |

---

## Etapa 7 — De MLP a CNN *(reordenada: estiramiento opcional, se intenta al final si sobra tiempo)*

**Duración:** 3–4 semanas · **Entorno:** PC + simulación + FPGA · **Prioridad más baja del plan — no es requisito para dar el PI por completo**

### Conceptos de NN a incorporar
Convolución 2D, kernels, feature maps, stride/padding, pooling.

### Tareas (repetición corta de Etapas 1–3 — mismo proceso de entrenamiento/cuantización/golden-reference ya conocido, aplicado a una red distinta)
1. Entrenar una CNN chica: 2 capas conv + 1 densa (estilo LeNet reducida).
2. Cuantizar y armar la golden reference C de la CNN — mismo esquema de cuantización que la MLP, sin teoría nueva; lo nuevo es el loop más anidado en C (canales × posiciones × kernel), y el max-pooling en el dominio cuantizado es en realidad más simple de lo esperado (comparar enteros directo, sin reescalar).
3. Extender la NPU para convolución:
   - **Opción simple (recomendada dado el tiempo disponible):** im2col en la CPU + reuso del motor de matmul existente — la NPU de la Etapa 6 no cambia nada, toda la novedad queda en el firmware.
   - **Opción avanzada:** convolución directa en la FSM del acelerador — mucho más trabajo de RTL, riesgo real de exceder el tiempo disponible; queda como posible extensión, no como plan.

### Checkpoint de salida
CNN completa acelerada, bit-exacta. El acelerador pasa de "matmul engine" a "NPU".

---

## Resumen de dependencias y paralelismo (actualizado)

- **Etapas 1–3:** 100% PC. Se puede arrancar hoy, sin placa.
- **Etapa 4:** independiente de 1–3 en lo técnico → se puede hacer en paralelo si el tiempo lo permite.
- **Etapas 5–6:** requieren 3 y 4 completas.
- **A partir de la Etapa 6 cerrada, se abren tres caminos independientes entre sí** (ninguno requiere a los otros dos, los tres solo requieren la 6):
  1. **Etapa 9 (demo funcional)** — prioridad 1.
  2. **Etapa 8 (toolchain)** — prioridad 2, se puede hacer en paralelo con la 9 o inmediatamente después.
  3. **Etapa 7 (CNN)** — prioridad 3 / opcional, al final, solo si sobra tiempo después de 8 y 9.
- **Orden de ejecución recomendado:** 1 → 2 → 3 → 4 (en paralelo con 1-3) → 5 → 6 → 9 → 8 → 7 (si hay tiempo).

## Decisiones abiertas

- [ ] Variante de la placa: Arty A7-35 vs. A7-100 (definido: **A7-100T**, elegido por margen de BRAM).
- [ ] Simulador principal: Verilator vs. CocoTB+Icarus (o ambos por nivel).
- [ ] Lenguaje RTL del acelerador.
- [ ] Polling vs. interrupciones en la v1 de la NPU (recomendado: polling primero).
- [ ] Ancho del array de MACs de la NPU: 8 vs. 16 (definir con datos reales de la Etapa 5 — LUT/DSP consumidos, timing — antes de comprometerse).
- [ ] Módulo táctil SPI concreto a comprar (ILI9341+XPT2046 con LCD, 2.4" o 2.8", verificar niveles lógicos 3.3V del datasheet puntual antes de cablear — sumar level shifter BSS138 como resguardo).
- [ ] Diseño del periférico GPIO/SPI bit-banged para el SoC (nuevo, chico, comparable en esfuerzo al UART de la Etapa 4).
- [ ] Organización del paralelismo en el array de MACs de la Etapa 6: repartir entre las entradas de una neurona vs. repartir entre neuronas distintas (afecta cómo se leen las BRAM y cuántos acumuladores hacen falta).
