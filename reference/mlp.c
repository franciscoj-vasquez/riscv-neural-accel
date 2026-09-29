/* Inferencia entera de la MLP 784→32→10: la referencia en C del proyecto.
 *
 * Replica bit a bit forward_int() de model/quantize.py: el checkpoint (check.c) lo verifica
 * sobre las 10.000 imágenes de test. No usa floats, malloc ni stdio, así que el mismo
 * archivo compila en la PC y en el PicoRV32. */
#include "mlp.h"
#include "mlp_int8.h"  /* los enteros de la red: model/artifacts/, generado por export_c.py */

/* Si se regenera mlp_int8.h con otra arquitectura, no compila hasta actualizar mlp.h. */
_Static_assert(sizeof mlp_w1 == MLP_HIDDEN * MLP_IN, "mlp_int8.h no coincide con mlp.h");
_Static_assert(sizeof mlp_w2 == MLP_OUT * MLP_HIDDEN, "mlp_int8.h no coincide con mlp.h");

/* int32 → uint8: round(acc · M0 / 2^N), saturado a 255.
 *
 * La ReLU va primero: todo acumulador negativo da 0, lo mismo que en Python, donde la
 * saturación lo lleva a 0. Así nunca se corre a la derecha un número negativo (en C, el
 * resultado de eso depende del compilador) y el producto se hace sin signo, en 64 bits
 * (acc < 2^31 y M0 < 2^31). Sumar 2^(N-1) antes de correr N bits redondea al más cercano. */
static uint8_t requantize(int32_t acc)
{
    uint64_t relu = acc > 0 ? (uint64_t)acc : 0;
    uint64_t r = (relu * MLP_M0 + ((uint64_t)1 << (MLP_N - 1))) >> MLP_N;
    return r > 255 ? 255 : (uint8_t)r;
}

void mlp_infer(const uint8_t x[MLP_IN], struct mlp_result *res)
{
    /* Capa oculta: 32 neuronas de 784 MACs (uint8 × int8), acumuladas en int32. */
    for (int j = 0; j < MLP_HIDDEN; j++) {
        int32_t acc = mlp_b1[j];
        for (int i = 0; i < MLP_IN; i++)
            acc += (int32_t)x[i] * mlp_w1[j][i];
        res->acc1[j] = acc;
        res->h[j] = requantize(acc);
    }

    /* Capa de salida: 10 neuronas de 32 MACs, sin requantizar. La predicción es el mayor
     * acumulador: comparar con > deja al primero ante un empate, igual que np.argmax. */
    int digit = 0;
    for (int k = 0; k < MLP_OUT; k++) {
        int32_t acc = mlp_b2[k];
        for (int j = 0; j < MLP_HIDDEN; j++)
            acc += (int32_t)res->h[j] * mlp_w2[k][j];
        res->acc2[k] = acc;
        if (acc > res->acc2[digit])
            digit = k;
    }
    res->digit = digit;
}
