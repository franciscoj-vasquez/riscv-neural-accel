---
lang: es
documentclass: report
numbersections: true
toc: true
toc-depth: 3
geometry: margin=2.8cm
fontsize: 11pt
header-includes:
  - \usepackage{graphicx}
  - \usepackage{hyperref}
  - \usepackage{amsmath}
  - \usepackage{tikz}
  - \usetikzlibrary{positioning,calc}
  - \AtBeginDocument{\renewcommand{\tablename}{Tabla}}
  - \hypersetup{pdftitle={Propuesta de Proyecto Integrador},pdfauthor={Francisco Javier Vasquez Curetti}}
---

# Resumen {-}

Este documento propone el diseño, la implementación y la evaluación experimental de una unidad de procesamiento neuronal (NPU) acoplada a un procesador RISC-V embebido en FPGA, comparando de forma cuantitativa tres niveles progresivos de aceleración por hardware —software puro, instrucción custom vía interfaz de coprocesador (PCPI) y acelerador dedicado memory-mapped en el bus— sobre una red neuronal densa (MLP) cuantizada a 8 bits, entrenada para la clasificación de dígitos manuscritos (MNIST). El trabajo cerrará con una demostración funcional de clasificación en vivo.

# Acrónimos {-}

| Sigla | Significado |
|---|---|
| **MAC** | Multiply-Accumulate |
| **MLP** | Multi-Layer Perceptron |
| **CNN** | Convolutional Neural Network |
| **ISA** | Instruction Set Architecture |
| **PCPI** | Pico Co-Processor Interface |
| **SIMD** | Single Instruction, Multiple Data |
| **FSM** | Finite State Machine |
| **BRAM** | Block RAM |
| **LUT** | Look-Up Table |
| **FF** | Flip-Flop |
| **DSP** | Digital Signal Processing slice |
| **SoC** | System on Chip |
| **NPU** | Neural Processing Unit |
| **SPI** | Serial Peripheral Interface |
| **ADC** | Analog-to-Digital Converter |
| **GPIO** | General Purpose Input/Output |

# Resumen Ejecutivo

## Idea central

Una red neuronal entrenada en punto flotante no puede ejecutarse eficientemente en hardware embebido de bajo costo: los multiplicadores de punto flotante son caros en área y potencia. Una técnica extendida para resolver este problema es la **cuantización**, que consiste en representar pesos y activaciones con números enteros en lugar de valores de punto flotante (_floats_), reduciendo sustancialmente el costo de cómputo a cambio de una pérdida de precisión numérica que, para redes de tamaño moderado, puede mantenerse reducida.

Este proyecto usa como caso de estudio la clasificación de dígitos manuscritos mediante red densa de tamaño reducido (el detalle capa por capa se muestra en la Tabla 3.1). Se iniciará con una implementación de referencia sobre un procesador RISC-V embebido (PicoRV32), a partir de la cual se construirán y compararán dos niveles sucesivos de aceleración por hardware:

1. **Nivel 1**: una instrucción custom de dot-product int8×4, integrada al procesador mediante su interfaz de coprocesador (PCPI), que reemplaza el loop escalar de multiplicación-acumulación por una operación empaquetada.
2. **Nivel 2**: un acelerador dedicado (NPU) conectado como periférico memory-mapped en el bus, con un arreglo de 8 a 16 unidades MAC en paralelo, que ejecuta una capa completa de la red por invocación sin intervención de la CPU durante el cómputo.

Cada nivel se valida, sobre el mismo core y la misma FPGA, con el mismo conjunto de métricas: ciclos por inferencia, speedup, uso de recursos lógicos (LUT/FF/BRAM/DSP) y exactitud de clasificación.

## Motivación y relevancia

Como Proyecto Integrador de la carrera de Ingeniería en Computación, este trabajo busca articular en un único diseño conocimientos de al menos tres áreas: Arquitectura de Computadoras (el procesador y su extensión), Inteligencia Artificial (el modelo y su cuantización) y Electrónica Digital.

Tecnológicamente, esta integración es posible en concreto porque RISC-V es una arquitectura de conjunto de instrucciones (ISA) abierta: a diferencia de las ISAs propietarias, permite extender el procesador con instrucciones y periféricos a medida, sin autorización de terceros ni acceso a un diseño cerrado. Esa apertura es la que habilita, en este proyecto, tanto la instrucción custom del Nivel 1 como el periférico memory-mapped del Nivel 2.

# Introducción

## Contexto del problema

El costo de realizar inferencias mediante redes neuronales en microcontroladores y FPGAs puede medirse, al menos, por las siguientes dos metricas: 

 - Uso de la memoria disponible, para almacenar los pesos de las conexiones entre neuronas
 - Computacionalmente cuántas multiplicaciones-acumulación (MACs) por segundo puede sostener el hardware.

La cuantización a enteros de 8 bits nos servirá como mecanismo para optimizar ambas metricas; la primera de ellas de manera trivial, y para la segunda permitiendo las posteriores aceleraciones ya introducidas. El costo de introducir esta tecnica seria esperable que se manifieste mediante una perdida en la precision del modelo - la cual se medira empiricamente iniciado el proyecto.

// TODOs Conexion? Faltaria aqui alguna separacion u algo por el estilo

El resto de este documento se organiza en cuatro capítulos. El Marco Teórico (Capítulo 3) desarrolla los fundamentos necesarios: redes neuronales para inferencia embebida, el esquema de cuantización entera, y los dos mecanismos de extensión de hardware —PCPI y memory-mapped— sobre PicoRV32. Trabajos Relacionados (Capítulo 4) sitúa este proyecto respecto de sus antecedentes directos e indirectos. Planteo del Problema y Propuesta de Solución (Capítulo 5) formaliza el problema, delimita el alcance, define el modelo de verificación y detalla la arquitectura propuesta. Estado del Anteproyecto y Plan de Trabajo (Capítulo 6) cierra el documento con un resumen honesto de las decisiones ya tomadas y las que permanecen abiertas.

# Marco Teórico

## Redes neuronales para inferencia embebida

### La neurona, MACs y funciones de activación

Una neurona de una capa densa (_fully connected_) calcula

$$y = \phi\!\left(\sum_{i=1}^{N} x_i\, w_i + b\right),$$

donde $x_i$ son las entradas, $w_i$ los pesos, $b$ el sesgo (_bias_) y $\phi$ la función de activación. 

Este proyecto utilizará **ReLU** (_Rectified Linear Unit_), $\phi(x) = \max(0, x)$ como funcion de activacion, lo cual es un estándar en redes de inferencia embebida por ser trivial de implementar en hardware, a diferencia de activaciones como sigmoide o tangente hiperbólica.

La operación que se repite una vez por cada entrada —multiplicar y sumar al acumulador— se conoce como **MAC** (_multiply-accumulate_), y es el tipo de operación dominante en el calculo del costo computacional de una red neuronal: una capa con $N$ entradas y $M$ neuronas ejecuta $N \times M$ MACs, $M$ sumas de bias y $M$ funciones de activación. 

### La red que se utilizará

Se utilizá como red de referencia una MLP de una unica capa oculta, 784→32→10, entrenada sobre MNIST. El tamaño de entrada (784 = 28×28 píxeles) y el de salida (10 clases) están determinados por el dataset y el problema, no son un parámetro de diseño; el tamaño de la capa oculta (32 neuronas) sí es una decisión deliberada, apoyada en los siguientes criterios: 
- El conteo total de pesos y bias moderadamente reducido, el cual mantiene acotado el presupuesto de memoria.
- Se espera un buen nivel de exactitud con esas dimensiones de red, de manera tal que agrandar la red podria no suponer una mejora proporcional al costo de hardware adicional.
- Al ser una potencia de dos ($2^5$), resultará en que los anchos de arreglo de MACs contemplados más adelante para la NPU (8 y 16) sean divisibles sin resto, evitando rondas parciales.

La Tabla 3.1 detalla el costo exacto por capa. El total de 25.408 MACs es la cifra de referencia usada en el resto de este documento para estimar tiempos de ejecución y paralelismo de hardware; los 42 bias, al ser dos órdenes de magnitud menos numerosos que los pesos, se acumulan en 32 bits sin impacto apreciable en el presupuesto de memoria (ver «Acumulación en int32 y requantización entre capas», más adelante).

| Capa | Entradas | Neuronas | Pesos (= MACs) | Bias | Parámetros |
|---|---:|---:|---:|---:|---:|
| Oculta (784→32) | 784 | 32 | 25.088 | 32 | 25.120 |
| Salida (32→10) | 32 | 10 | 320 | 10 | 330 |
| **Total** | — | — | **25.408** | **42** | **25.450** |

: Parámetros y MACs por capa de la MLP 784→32→10.

## Cuantización de redes neuronales

Este proyecto explora una solución pensada para procesadores minimalistas —como PicoRV32—, en los que correr una red neuronal en punto flotante de forma eficiente no es viable. La cuantización permite ejecutarla de todos modos, a cambio de resignar potencialmente algo de exactitud —la magnitud real de esa pérdida es algo a determinar mediante la validación— a favor de una ganancia sustancial en eficiencia de cómputo: menos ciclos de reloj por inferencia, menos instrucciones ejecutadas, menor consumo de energía.

El PicoRV32 implementa el ISA entero RV32I, por lo que no cuenta con hardware de punto flotante. Es por ello que cada operación en _float_ se emularía por software mediante una secuencia de varias decenas de instrucciones enteras. Incluso en procesadores que sí incluyen una FPU, multiplicar en punto flotante exige manejar signo, exponente y mantisa por separado, normalizar el resultado y redondearlo correctamente.

### Escala, zero-point y cuantización simétrica/asimétrica

// TODOs no entiendo bien esta seccion.. ver de entenderla y redactarla mejor!

La cuantización mapea un rango real $[x_{\min}, x_{\max}]$ a un rango entero de 8 bits mediante una escala $S$ y un _zero-point_ $Z$, según

$$r = S\,(q - Z), \qquad q = \mathrm{round}\!\left(\frac{r}{S}\right) + Z,$$

donde $r$ es el valor real y $q$ su representación entera, saturada al rango de 8 bits ($[-128,127]$ con signo, o $[0,255]$ sin signo, según corresponda). Los pesos suelen cuantizarse de forma **simétrica** ($Z = 0$), lo cual simplifica el hardware de multiplicación al eliminar el término de corrección cruzada que introduce un _zero-point_ distinto de cero; las activaciones posteriores a una ReLU (que solo toman valores no negativos) suelen cuantizarse de forma **asimétrica** ($Z \neq 0$), aprovechando mejor el rango de 8 bits disponible en vez de desperdiciar la mitad en valores negativos que nunca ocurren.

### Acumulación en int32 y requantización entre capas

La suma de $N$ productos de 8×8 bits se acumula en un registro de 32 bits para evitar desborde, y el bias se representa también en 32 bits. Al finalizar la suma de una capa, el acumulador de 32 bits debe volver a comprimirse a 8 bits para alimentar la siguiente capa. Este paso, llamado **requantización**, reescala el acumulador por el factor

$$M = \frac{S_{\text{entrada}} \cdot S_{\text{peso}}}{S_{\text{salida}}},$$

que en punto flotante sería una simple multiplicación, pero que en hardware entero se aproxima descomponiendo $M = M_0 \cdot 2^{-n}$, con $M_0 \in [0.5, 1)$ representado como entero de punto fijo y $n$ un corrimiento de bits (_shift_) hacia la derecha — de modo que la requantización completa se reduce a una multiplicación entera y un _shift_, sin usar nunca una unidad de punto flotante.

### El esquema de referencia: Jacob et al.

El esquema de cuantización entera adoptado en este proyecto sigue el trabajo de Jacob et al. [1], que describe el método sobre el cual se basa la cuantización por defecto de TensorFlow Lite: cuantización post-entrenamiento (sin reentrenar la red), acumulación en 32 bits, y requantización mediante multiplicador de punto fijo. Para una red del tamaño de la usada en este proyecto, la pérdida de exactitud esperada frente a la versión en punto flotante es mínima — el propio checkpoint de validación bit-exacta contra el conjunto de test completo de MNIST provee la cifra real, en lugar de depender de una estimación teórica.

## RISC-V y el procesador PicoRV32

### RISC-V como ISA abierta y extensible

A diferencia de ISAs propietarias, la especificación RISC-V reserva explícitamente rangos de opcode (`custom-0`, `custom-1`, y `custom-2`/`custom-3` en variantes de 128 bits) para que los implementadores agreguen instrucciones propias sin colisionar con el conjunto estándar ni requerir autorización de terceros. Esta extensibilidad es la que habilita el mecanismo de aceleración por instrucción custom descripto más adelante.

### PicoRV32: un core minimalista

PicoRV32 [2], del autor Clifford Wolf, es una implementación abierta y sintetizable del ISA RV32I, de arquitectura **single-issue e in-order**: decodifica y ejecuta una instrucción por ciclo, sin lógica de emisión múltiple ni ejecución fuera de orden. Esta simplicidad deliberada (el proyecto se describe a sí mismo como _size-optimized_) es la razón central por la que la vía de aceleración elegida en este proyecto es de tipo SIMD/empaquetado y no una técnica de tipo superescalar, que requeriría hardware de emisión múltiple ausente en este core.

## Extensión de ISA vía coprocesador: la interfaz PCPI

### Protocolo de handshake

PicoRV32 expone una interfaz de coprocesador liviana llamada **PCPI** (_Pico Co-Processor Interface_). Cuando el core decodifica una instrucción del espacio `custom-0` que no reconoce internamente, levanta la señal `pcpi_valid` y expone la instrucción cruda (`pcpi_insn`) y los valores de los registros fuente (`pcpi_rs1`, `pcpi_rs2`) a cualquier unidad externa conectada al puerto. Dicha unidad decodifica si la instrucción le corresponde, ejecuta el cómputo, y responde con `pcpi_ready` (más `pcpi_wr`/`pcpi_rd` si corresponde escribir un resultado al registro destino). Si ninguna unidad responde dentro de 16 ciclos, el core levanta una excepción de instrucción ilegal, preservando la compatibilidad con el ISA base cuando el periférico no está presente.

Es importante notar que esto **no constituye una arquitectura de dos procesadores**: sigue existiendo un único program counter y un único flujo de instrucciones; la unidad PCPI es una unidad de ejecución adicional —análoga en concepto a un coprocesador matemático clásico (FPU)— y no hay ejecución concurrente: el pipeline se detiene mientras la unidad PCPI calcula.

### Por qué SIMD y no superescalar

La instrucción propuesta para este proyecto —un dot-product de 4 elementos int8, empaquetados en dos registros de 32 bits— es una técnica de paralelismo de datos dentro de una palabra (_sub-word parallelism_ / SIMD), no de paralelismo de instrucciones: dado que PicoRV32 es, como se estableció más arriba, _single-issue_ e _in-order_, empaquetar más datos en los operandos de una única instrucción es la única forma de aumentar el trabajo útil por ciclo sin modificar la microarquitectura del core.

Un problema de diseño no trivial es dónde reside el acumulador de la operación, dado que el protocolo PCPI no expone el valor previo del registro destino como entrada: la solución adoptada (siguiendo el patrón usado por trabajos con interfaces equivalentes, ver Trabajos Relacionados) es mantener el acumulador dentro de la propia unidad PCPI, con una instrucción separada para leerlo y resetearlo.

### Toolchain: generación de la instrucción sin modificar el compilador

La instrucción custom se invoca desde C mediante la directiva `.insn` del ensamblador GNU, que permite emitir el word de 32 bits exacto de una instrucción (especificando opcode, funct3, funct7 y registros) sin que el ensamblador necesite reconocer el mnemónico de antemano. Esta directiva se envuelve en una macro de C con _inline assembly_, de forma que el firmware la invoca como si fuera una función común, sin requerir ningún parche al compilador `riscv-gcc`.

## Aceleradores memory-mapped y arquitectura de NPU embebida

### Memory-mapped vs. extensión de ISA

A diferencia de la instrucción custom, un acelerador **memory-mapped** no introduce ninguna instrucción nueva: se le asigna un rango de direcciones de memoria, y el procesador interactúa con él mediante operaciones `load`/`store` ordinarias — el mismo mecanismo que ya usa cualquier periférico de E/S convencional (por ejemplo, una UART). La CPU no despacha una operación por dato, como en la instrucción custom, sino que configura el acelerador (dimensiones de la capa, punteros a pesos y activaciones), lo dispara con una escritura a un registro de control, y espera a que termine — de modo que el acelerador ejecuta autónomamente el cómputo de una capa completa por invocación, sin más intervención del procesador durante ese lapso. La Figura 3.1 esquematiza, lado a lado, los tres modelos de ejecución comparados en este proyecto.

\begin{figure}[htbp]
\centering
\begin{tikzpicture}[
  font=\small,
  node distance=1.9cm and 3.0cm,
  cpu/.style={draw, rectangle, rounded corners, minimum width=2.6cm, minimum height=1.0cm, align=center},
  unit/.style={draw, rectangle, minimum width=4.8cm, minimum height=1.2cm, align=center},
  tag/.style={font=\scriptsize\bfseries},
  lbl/.style={font=\scriptsize, align=center}
]

\node[cpu] (cpuA) {PicoRV32};
\node[unit, right=of cpuA] (memA) {Memoria de datos};
\draw[<->] (cpuA) -- (memA);
\draw[->] (cpuA) edge[loop above] node[lbl, above] {loop MAC escalar\\(sin unidad externa)} (cpuA);
\node[tag, left=0.3cm of cpuA] {(a)};

\node[cpu, below=of cpuA] (cpuB) {PicoRV32};
\node[unit, right=of cpuB] (pcpiB) {Unidad PCPI\\dot-product int8$\times$4 + acumulador};
\draw[<->] (cpuB) -- (pcpiB) node[lbl, midway, above] {handshake PCPI\\(pipeline detenido)};
\node[tag, left=0.3cm of cpuB] {(b)};

\node[cpu, below=of cpuB] (cpuC) {PicoRV32};
\node[unit, right=of cpuC] (npuC) {NPU: FSM + array de MACs\\BRAM pesos / activ. entrada / activ. salida};
\draw[->] ($(cpuC.east)+(0,0.15)$) -- ($(npuC.west)+(0,0.15)$) node[lbl, midway, above] {config / start (store)};
\draw[->] ($(npuC.west)+(0,-0.15)$) -- ($(cpuC.east)+(0,-0.15)$) node[lbl, midway, below] {done (poll)};
\node[tag, left=0.3cm of cpuC] {(c)};

\end{tikzpicture}
\caption{Los tres niveles de aceleración comparados en este proyecto. (a) baseline en software puro: la propia CPU ejecuta el loop MAC contra la memoria de datos. (b) Nivel 1: instrucción custom vía PCPI, con el pipeline de la CPU detenido durante el cómputo. (c) Nivel 2: NPU memory-mapped; la CPU configura y dispara el acelerador y queda libre hasta sondear la señal de fin.}
\label{fig:niveles}
\end{figure}

### Arreglo de MACs, BRAM y máquina de estados

La arquitectura propuesta para la NPU consiste en un arreglo de 8 a 16 unidades MAC int8 operando en paralelo —paralelismo espacial real, en contraste con el empaquetado temporal de la instrucción custom—, alimentadas por BRAMs (memoria de bloque dedicada de la FPGA, necesaria porque un array de $N$ MACs requiere leer $N$ pares de operandos por ciclo, algo que una memoria compartida con la CPU no podría sostener sin convertirse en cuello de botella) para pesos, activaciones de entrada y activaciones de salida. Una máquina de estados finitos (FSM) de control recorre las rondas necesarias para completar la capa, aplica bias y requantización, y señaliza finalización mediante un bit de estado (`done`), que la CPU puede consultar por _polling_ (consulta repetida) o mediante una interrupción.

El ancho del arreglo (8 vs. 16 MACs) y la forma de organizar el paralelismo (repartido entre las entradas de una misma neurona, o entre neuronas distintas procesadas en simultáneo) son decisiones de diseño abiertas, que determinan directamente el consumo de DSPs y el ancho de banda de lectura requerido de las BRAM — se detallan como decisión pendiente en el capítulo «Estado del Anteproyecto y Plan de Trabajo».

## Recursos de FPGA relevantes

A los fines de medir y comparar los tres niveles de aceleración, este proyecto reporta consistentemente cuatro tipos de recursos de la FPGA: **LUT** (_Look-Up Table_, el bloque básico de lógica combinacional), **FF** (_Flip-Flop_, el bloque básico de almacenamiento de 1 bit, usado por ejemplo para el estado de la FSM o los acumuladores), **BRAM** (memoria de bloque dedicada, distinta de la lógica general) y **DSP** (bloque de multiplicación-acumulación ya implementado en silicio, mucho más eficiente que construir un multiplicador a partir de LUTs).

## Interfaz de entrada para la demostración funcional

Para la demostración final se optó por un panel táctil resistivo con interfaz SPI (controlador de touch tipo XPT2046, con controlador de pantalla LCD tipo ILI9341) en lugar de una cámara. Esta decisión y su justificación se desarrollan en el capítulo «Planteo del Problema y Propuesta de Solución»; a nivel de marco teórico, cabe señalar dos detalles de bajo nivel relevantes para el diseño del firmware de captura. Primero, PicoRV32 no incluye controladores SPI ni **GPIO** nativos, por lo que la comunicación con el panel se implementa mediante un periférico GPIO mapeado en memoria y _bit-banging_ por firmware de las líneas del bus SPI (`CS`/`MOSI`/`MISO`/`SCK`) — un esfuerzo de diseño comparable al del periférico UART ya requerido para el baseline. Segundo, un controlador táctil resistivo entrega una lectura cruda de 12 bits de un **ADC** interno por cada eje, que requiere una calibración lineal (mapeo de esquinas conocidas) antes de poder interpretarse como coordenadas de pantalla — un detalle bien documentado en la literatura de estos controladores [3], pero necesario para el diseño del firmware de captura.

# Trabajos Relacionados

## PicoRV32 y el ecosistema de cores RISC-V minimalistas

PicoRV32 [2] es el core sobre el que se apoya este proyecto, y su interfaz PCPI es, en sí misma, un antecedente de diseño: es un mecanismo de extensión ya probado y documentado, usado en múltiples trabajos de terceros para acoplar coprocesadores a medida sin modificar el núcleo del procesador.

## Rosso, Zerbini y Riva: aceleración de CORDIC sobre PicoRV32

El antecedente más directo de este proyecto proviene del grupo de investigación GInTEA (Grupo de Investigación y Transferencia en Electrónica Avanzada), de la Universidad Tecnológica Nacional, Facultad Regional Córdoba. Rosso, Zerbini y Riva [4] comparan, sobre el mismo core PicoRV32 y la misma familia de placas Arty A7 consideradas en este proyecto, tres niveles de aceleración para el cómputo de funciones trigonométricas mediante el algoritmo CORDIC: firmware puro (baseline), extensión de ISA sin coprocesador dedicado (instrucciones aritméticas agregadas directamente a la ALU del core) e interfaz PCPI con acelerador dedicado.

Los resultados muestran una progresión no lineal especialmente relevante para las decisiones de diseño de este proyecto: la extensión de ISA sin coprocesador logra apenas 1,19× de speedup, con 978 LUT (+13,9 %) y 524 FF (+1,4 %) sobre el baseline, mientras que la interfaz PCPI alcanza 6,1× con 1992 LUT (+132 %) y 1703 FF (+229 %) — una ganancia de rendimiento muy superior, en proporción, al costo de recursos adicionales. Esta relación de compromiso es uno de los antecedentes que motivan, en este proyecto, descartar una extensión de ISA que modifique directamente la ALU del core (opción no contemplada en ningún nivel de este trabajo) y concentrar el esfuerzo de diseño en la interfaz PCPI (Nivel 1) y en el acelerador memory-mapped (Nivel 2), ambos externos al datapath del procesador.

## Rosso, Zerbini y Riva: instrucciones custom para filtrado SDR sobre NEORV32

El mismo grupo, en un trabajo relacionado sobre el core NEORV32 [5] (que expone una interfaz de extensión conceptualmente equivalente a la PCPI de PicoRV32, denominada _Custom Function Unit_), implementa dos instrucciones de multiplicación-acumulación en punto fijo (`fmac`/`fmac2`) para acelerar el filtrado adaptado (_Root Raised Cosine_) en receptores de radio definida por software, reportando speedups de 1,62× a 1,75× con un overhead de área mínimo (+100 LUT / +146 FF / +2 DSP sobre el baseline, según el propio trabajo). Este trabajo aporta, además, el patrón de diseño de acumulador interno a la unidad de extensión (en lugar del banco de registros del procesador) adoptado en este proyecto.

## Aceleradores de mayor escala: Gemmini y NVDLA

A nivel de proyectos internacionales de mayor escala, **Gemmini** (UC Berkeley, proyecto Chipyard) [6] es un generador de aceleradores sistólicos de multiplicación de matrices acoplado como acelerador RoCC a un core RISC-V Rocket de 64 bits, y **NVDLA** [7] (NVIDIA, open source) es un acelerador de inferencia de redes neuronales profundas configurable, integrado en trabajos posteriores a SoCs basados en RISC-V. Ambos proyectos comparten con este trabajo la idea de acoplar un acelerador de IA a un core RISC-V, pero operan con una complejidad de diseño, verificación e integración muy superior (cores de 64 bits con jerarquías de caché completas, herramientas de generación de hardware parametrizable), correspondiente a proyectos de investigación de mayor escala. Este proyecto prioriza, en cambio, un diseño íntegramente propio y didáctico sobre un core mínimo de 32 bits, donde cada componente (desde la cuantización hasta el RTL de la NPU) es construido y verificado por el autor.

## Generación automática de aceleradores: FINN y microTVM

**FINN** [8] (desarrollado originalmente por Xilinx Research Labs) es un compilador que genera aceleradores FPGA a partir de redes neuronales cuantizadas o binarizadas, y **microTVM** (subproyecto de Apache TVM) cumple un rol equivalente para microcontroladores de propósito general. Ambos son la inspiración declarada —no adoptada— para una eventual etapa de generación automática de firmware a partir del modelo cuantizado (ver capítulo siguiente): a diferencia de estas herramientas de producción, el generador contemplado en este proyecto sería una herramienta propia y acotada a la interfaz de la NPU descripta en este documento, con fines didácticos antes que de generalidad.

## Relevancia actual

Los cuatro antecedentes discutidos en este capítulo delimitan el espacio de diseño de este proyecto. Frente al trabajo de Rosso, Zerbini y Riva sobre PicoRV32 [4], este proyecto reutiliza la misma metodología de comparación de niveles y el mismo par núcleo/placa, pero la aplica a un dominio distinto —inferencia de redes neuronales cuantizadas— que introduce un problema ausente en la aceleración de CORDIC: la propagación de error numérico a través de múltiples capas, lo cual motiva la disciplina de verificación bit-exacta descripta en la Introducción. Frente a Gemmini y NVDLA [6, 7], la brecha no es de objetivo sino de escala: ambos operan sobre cores de 64 bits con jerarquías de caché y toolchains de generación de hardware parametrizable, un orden de complejidad de verificación muy superior al de un core de 32 bits sin caché como PicoRV32. Frente a FINN y microTVM [8], la diferencia es de alcance: este proyecto no busca construir un compilador general de redes a hardware, sino verificar a mano, sobre una arquitectura propia y acotada, los mismos principios —cuantización entera, generación de firmware desde el modelo— que esas herramientas automatizan a escala de producción; la eventual etapa de generación automática de firmware (ver «Alcance y Limitaciones del Proyecto») sería, en ese sentido, una versión mínima y didáctica de lo que FINN y microTVM resuelven de forma general.

La combinación de una ISA abierta (RISC-V), técnicas de cuantización estandarizadas en la industria (el esquema de Jacob et al. es la base de facto de TensorFlow Lite) y la disponibilidad de FPGAs de bajo costo hace que el patrón de diseño explorado en este proyecto sea, más allá de su carácter didáctico, directamente trasladable a escenarios de producción en dominios como IoT, edge computing y sistemas embebidos automotrices. No se tiene conocimiento, además, de un Proyecto Integrador previo en la Escuela de Ingeniería en Computación que combine estos tres elementos sobre el dominio específico de inferencia de redes neuronales.

# Planteo del Problema y Propuesta de Solución

## Definición del problema

El problema central de este proyecto es doble: (i) demostrar, de forma verificable y medible, que la cuantización entera preserva la exactitud de una red neuronal pequeña dentro de un margen aceptable, y (ii) cuantificar el costo/beneficio real de acelerar esa inferencia mediante hardware a medida, en dos mecanismos de extensión arquitectónicamente distintos (instrucción custom vs. periférico memory-mapped), sobre el mismo procesador y la misma FPGA. La dificultad práctica no reside en ninguno de los conceptos por separado, sino en sostener la disciplina de verificación bit-exacta a través de las sucesivas traducciones entre dominios: Python → C → RISC-V → RTL.

## Alcance y limitaciones del proyecto

El compromiso formal de este Proyecto Integrador comprende:

- Fundamentos y entrenamiento de la MLP 784→32→10 sobre MNIST, alcanzando un buen nivel de exactitud en punto flotante.
- Cuantización entera a mano, con golden reference en C validada bit a bit sobre el conjunto de test completo.
- SoC base en FPGA sobre el core PicoRV32, con medición del baseline en ciclos por inferencia.
- Aceleración por instrucción custom vía interfaz PCPI (**Nivel 1**: dot-product int8×4), con validación bit-exacta y medición de speedup.
- Acelerador **NPU** memory-mapped en el bus (**Nivel 2**), con arreglo paralelo de unidades MAC, validado de igual forma.
- Demostración funcional de clasificación en vivo, con el dígito ingresado mediante un panel táctil.

Quedan **fuera del alcance formal** de este PI, y se plantean como posible continuación del trabajo a intentar solo si el cronograma lo permite una vez cerrada la demostración funcional, la extensión del modelo a una red convolucional (CNN) y la generación automática de firmware a partir del modelo cuantizado. Esta decisión de alcance prioriza deliberadamente llegar a una demostración completa de punta a punta por sobre agregar sofisticación al modelo, dado el riesgo de que el cronograma se ajuste en un proyecto part-time de varios meses.

Por los motivos desarrollados en el capítulo «Trabajos Relacionados», se descarta explícitamente la adopción directa de generadores de hardware de producción (Gemmini, FINN, microTVM): se los toma como referencia conceptual, no como dependencia del proyecto.

## Modelo de verificación

Análogamente a como un proyecto de seguridad define un modelo de amenaza, este proyecto formaliza como **modelo de verificación** la disciplina motivada en la Introducción. La Figura 5.1 esquematiza la cadena completa de artefactos —referencia cuantizada en Python, golden reference en C puro, firmware en RISC-V, RTL simulado y RTL sintetizado en FPGA— donde cada eslabón debe reproducir exactamente, bit a bit y sobre el conjunto de test completo de MNIST (10.000 imágenes), la salida del artefacto anterior antes de avanzar a la siguiente etapa; en ese sentido, cada nivel es el oráculo del siguiente.

\begin{figure}[htbp]
\centering
\begin{tikzpicture}[
  font=\small,
  node distance=1.0cm,
  stage/.style={draw, rectangle, rounded corners, minimum width=8.4cm, minimum height=0.9cm, align=center},
  lbl/.style={font=\scriptsize}
]
\node[stage] (py) {Referencia cuantizada en Python\\(NumPy, post-training quantization)};
\node[stage, below=of py] (c) {Golden reference en C puro\\(int8 / int32, sin floats)};
\node[stage, below=of c] (fw) {Firmware en PicoRV32\\(mismo C, compilado con riscv-gcc)};
\node[stage, below=of fw] (rtl) {RTL simulado\\(PCPI / NPU en Verilator)};
\node[stage, below=of rtl] (fpga) {RTL sintetizado en FPGA\\(medición de ciclos y recursos)};

\draw[->] (py) -- (c) node[lbl, midway, right=2pt] {bit-exacto};
\draw[->] (c) -- (fw) node[lbl, midway, right=2pt] {bit-exacto};
\draw[->] (fw) -- (rtl) node[lbl, midway, right=2pt] {bit-exacto};
\draw[->] (rtl) -- (fpga) node[lbl, midway, right=2pt] {bit-exacto};
\end{tikzpicture}
\caption{Cadena de verificación bit-exacta del proyecto. Cada artefacto debe reproducir exactamente —bit a bit, sobre las 10.000 imágenes de test de MNIST— la salida del artefacto anterior antes de avanzar a la siguiente etapa.}
\label{fig:verificacion}
\end{figure}

El motivo por el que el checkpoint tiene que estar en cada eslabón de la Figura 5.1, y no solo al final, es que la visibilidad de depuración disponible cae en sentido inverso al costo de un error: en Python cualquier variable intermedia se inspecciona con un `print`; en el firmware corriendo sobre el core real, la única ventana disponible es lo que se decida sacar por UART; y en la FPGA sintetizada, ni siquiera eso es gratuito sin instrumentación adicional. Un ejemplo concreto: una diferencia tan chica como implementar el redondeo de la requantización como _round-half-up_ en lugar de _round-half-to-even_ desplaza en ±1 el valor cuantizado de un puñado de activaciones que caen justo en el límite de redondeo — invisible en la exactitud agregada sobre las 10.000 imágenes de test, pero indistinguible de un error de RTL si se detecta por primera vez en la FPGA. Validar bit a bit en cada transición es lo que permite ubicar ese tipo de error en la etapa donde todavía es barato encontrarlo.

La referencia final e inmutable de todo el proyecto es la golden reference en C puro construida en la etapa de cuantización: ningún nivel de aceleración se considera válido si no reproduce exactamente su salida.

## Propuesta técnica

### Arquitectura del sistema

El sistema se organiza en niveles sucesivos, cada uno construido sobre el anterior sin modificar los niveles previos: el mismo firmware en C, con solo el fragmento de cómputo reemplazado en cada nivel, corre sobre el baseline, el Nivel 1 y, a través de las llamadas a la NPU, el Nivel 2.

1. **Referencia software**: _forward pass_ en NumPy con pesos entrenados en PyTorch, y su versión cuantizada en C puro (entrada uint8, pesos int8, acumuladores int32, requantización de punto fijo) — el oráculo del proyecto.
2. **Baseline RISC-V**: la golden reference en C, portada sin modificaciones y ejecutada sobre PicoRV32 en FPGA, con medición de ciclos por inferencia mediante un contador de hardware.
3. **Nivel 1 — instrucción custom (PCPI)**: el mismo firmware, con el loop de multiplicación-acumulación reemplazado por invocaciones a la instrucción de dot-product int8×4, y el bloque PCPI correspondiente en RTL.
4. **Nivel 2 — NPU memory-mapped**: el mismo firmware, reemplazando ahora la capa completa por una invocación a la NPU (arreglo paralelo de unidades MAC, BRAMs dedicadas y FSM de control) mediante escrituras a un rango de registros mapeados en memoria (dimensiones, punteros, start/done).
5. **Demostración funcional**: un panel táctil resistivo por SPI, con lectura de posición calibrada, downscale por promediado a la resolución de entrada de la red (28×28) y centrado por centro de masa, alimenta al Nivel 2; el resultado de clasificación se reporta por UART.

### Metodología de evaluación

Cada nivel de aceleración se evalúa con el mismo conjunto de métricas (Tabla 5.1), sobre la misma red, el mismo core y la misma FPGA, para que la comparación entre niveles sea válida:

| Métrica | Descripción |
|---|---|
| Ciclos por inferencia | Medidos con contador de hardware, de punta a punta |
| Speedup | Relativo al baseline en software puro |
| LUT / FF | Recursos de lógica general consumidos |
| BRAM | Memoria de bloque consumida |
| DSP | Unidades de multiplicación-acumulación dedicadas consumidas |
| Exactitud | Sobre las 10.000 imágenes de test de MNIST, comparada bit a bit contra la golden reference |

: Métricas de evaluación comparadas entre niveles de aceleración.

### Justificación del periférico de entrada para la demostración

Se evaluó inicialmente una cámara (OV7670) como fuente de imagen para la demostración final, siguiendo un patrón de diseño estándar en este tipo de proyectos. Sin embargo, la mayor parte del pipeline de preprocesamiento que ese enfoque exige —recorte de región de interés, inversión de polaridad (una foto de papel es oscura sobre claro; MNIST es al revés), umbral adaptativo de contraste ante iluminación variable— existe únicamente para compensar los problemas inherentes a una fotografía real, no al problema de clasificación en sí. Se optó en cambio por un panel táctil resistivo, sobre el cual el dígito se dibuja directamente: la señal de entrada nace limpia (sin variabilidad de iluminación ni ruido óptico), lo que elimina la necesidad de la mayoría de esos pasos, a la vez que la interfaz de hardware (SPI) resulta sensiblemente más simple de implementar que la captura paralela de una cámara con su cruce de dominio de reloj asociado. Se mantiene, sí, el centrado por centro de masa, dado que la red utilizada (MLP) no posee ninguna invariancia arquitectónica a traslaciones del dígito dentro del cuadro.

# Estado del Anteproyecto y Plan de Trabajo

Este documento se presenta en etapa de anteproyecto: no existen todavía resultados experimentales de hardware, y este capítulo lista honestamente qué está decidido y qué permanece abierto, en lugar de anticipar mediciones que aún no se realizaron.

## Decisiones ya tomadas

- **Red y dataset**: MLP 784→32→10 sobre MNIST, con esquema de cuantización entera siguiendo Jacob et al. [1] (Tabla 3.1).
- **Core base**: PicoRV32.
- **Placa FPGA**: Arty A7-100T (Xilinx Artix-7), por margen de BRAM para pesos, activaciones y buffers de la NPU.
- **Nivel 1**: instrucción custom de dot-product int8×4 vía interfaz PCPI.
- **Nivel 2**: acelerador NPU memory-mapped con arreglo de 8 a 16 MACs.
- **Periférico de la demostración funcional**: panel táctil resistivo por SPI (tipo ILI9341+XPT2046), en lugar de cámara.
- **Priorización**: alcanzar la demostración funcional antes que extender el modelo a CNN (ver «Alcance y Limitaciones del Proyecto»).

## Decisiones pendientes

- Ancho definitivo del arreglo de MACs de la NPU (8 vs. 16) y organización del paralelismo (por neurona vs. por entrada), a definir con datos reales de consumo de recursos de la etapa de instrucción custom.
- Simulador principal (Verilator vs. CocoTB+Icarus) y lenguaje de descripción de hardware del acelerador.
- Polling vs. interrupciones para la sincronización CPU–NPU en la primera versión (se prioriza polling por simplicidad).
- Diseño concreto del periférico GPIO/SPI _bit-banged_ que expondrá el panel táctil al firmware.

## Plan de trabajo

El detalle de fases, dependencias entre etapas y cronograma tentativo se presenta en el Diagrama de Gantt adjunto a la Solicitud de Proyecto Integrador.

# Referencias {-}

1. Jacob, B.; Kligys, S.; Chen, B.; Zhu, M.; Tang, M.; Howard, A.; Adam, H.; Kalenichenko, D. (2018). "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference". *Proceedings of the IEEE CVPR*, pp. 2704–2713.

2. Wolf, C. "PicoRV32 – A Size-Optimized RISC-V CPU". github.com/YosysHQ/picorv32

3. Documentación de referencia del controlador táctil resistivo XPT2046 y calibración de coordenadas (fuentes de fabricante y comunidad de hardware abierto).

4. Rosso, D. A.; Zerbini, C. A.; Riva, G. G. (2025). "Exploring Hardware/Software Trade-offs for CORDIC Acceleration in a Customized Processor". *Congreso Argentino de Electrónica (CAE) 2025*, GInTEA, UTN Facultad Regional Córdoba.

5. Rosso, D. A.; Zerbini, C. A.; Riva, G. G. "Customizing NEORV32 for Matched Filtering in SDR Applications". GInTEA, UTN Facultad Regional Córdoba. github.com/ginteacomms-utn/2026_case_neorv32_rrc

6. Genc, H. et al. "Gemmini" – Berkeley systolic array generator para RISC-V (proyecto Chipyard). github.com/ucb-bar/gemmini

7. NVIDIA. "NVDLA – NVIDIA Deep Learning Accelerator" (open source). nvdla.org

8. Umuroglu, Y. et al. (2017). "FINN: A Framework for Fast, Scalable Binarized Neural Network Inference". *Proc. ACM/SIGDA Int. Symposium on FPGAs (FPGA '17)*.

9. Google. TensorFlow Lite – "Quantization specification" (documentación oficial). www.tensorflow.org/lite/performance/quantization_spec

10. IEEE. *IEEE Taxonomy*, January 2024, v1.03.
