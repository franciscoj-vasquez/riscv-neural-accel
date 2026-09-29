/* Inferencia entera de la MLP 784→32→10: la referencia en C del proyecto. */
#ifndef MLP_H
#define MLP_H

#include <stdint.h>

#define MLP_IN     784  /* píxeles de entrada: 28×28, fila por fila */
#define MLP_HIDDEN 32   /* neuronas de la capa oculta */
#define MLP_OUT    10   /* salidas: una por dígito */

/* Resultado de una inferencia: la predicción y todos los enteros intermedios, que el
 * checkpoint compara uno por uno contra Python. */
struct mlp_result {
    int32_t acc1[MLP_HIDDEN];  /* acumuladores de la capa oculta, con el bias */
    uint8_t h[MLP_HIDDEN];     /* capa oculta requantizada, con la ReLU aplicada */
    int32_t acc2[MLP_OUT];     /* acumuladores de salida, con el bias */
    int digit;                 /* argmax de acc2; ante empate, el primero */
};

/* x: 784 píxeles uint8, con trazo = 255 y fondo = 0, como MNIST. */
void mlp_infer(const uint8_t x[MLP_IN], struct mlp_result *res);

#endif /* MLP_H */
