/* Checkpoint de la Etapa 3, en la PC: mlp.c contra forward_int() de model/quantize.py.
 *
 * Uso (desde reference/):  make check
 *
 * build/vectors.bin lo genera model/export_c.py: las 10.000 imágenes de test, con su
 * etiqueta y todos los enteros que calculó Python. Para cada imagen se corre mlp_infer()
 * y se compara todo, valor por valor: acc1, h, acc2 y la predicción. Si algo difiere,
 * muestra los primeros casos y sale con código 1.
 *
 * Lee los int32 del archivo tal cual, así que supone una PC little-endian, como x86. En una
 * big-endian falla el chequeo del encabezado y no se compara nada. */
#include <stdio.h>
#include <string.h>

#include "mlp.h"

#define MAX_REPORTS 5  /* diferencias que se muestran en detalle */

struct record {  /* lo que calculó Python para una imagen */
    uint8_t x[MLP_IN];
    uint8_t label;
    int32_t acc1[MLP_HIDDEN];
    uint8_t h[MLP_HIDDEN];
    int32_t acc2[MLP_OUT];
    uint8_t digit;
};

/* Campo por campo: en el archivo no hay relleno entre campos, en el struct sí puede haber. */
static int read_record(FILE *f, struct record *r)
{
    return fread(r->x, 1, MLP_IN, f) == MLP_IN
        && fread(&r->label, 1, 1, f) == 1
        && fread(r->acc1, sizeof r->acc1[0], MLP_HIDDEN, f) == MLP_HIDDEN
        && fread(r->h, 1, MLP_HIDDEN, f) == MLP_HIDDEN
        && fread(r->acc2, sizeof r->acc2[0], MLP_OUT, f) == MLP_OUT
        && fread(&r->digit, 1, 1, f) == 1;
}

enum { ACC1, H, ACC2, DIGIT, N_KINDS };
static const char *const KIND_NAME[N_KINDS] = {"acc1", "h", "acc2", "predicción"};
static long diffs[N_KINDS];

static void compare(unsigned image, int kind, int index, long c, long python)
{
    if (c == python)
        return;
    if (diffs[ACC1] + diffs[H] + diffs[ACC2] + diffs[DIGIT] < MAX_REPORTS)
        printf("  imagen %u, %s[%d]: C %ld, Python %ld\n", image, KIND_NAME[kind], index, c,
               python);
    diffs[kind]++;
}

int main(int argc, char **argv)
{
    const char *path = argc > 1 ? argv[1] : "build/vectors.bin";
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "No se pudo abrir %s: se genera con model/export_c.py\n", path);
        return 1;
    }
    char magic[8];
    uint32_t dims[4];  /* imágenes, entradas, ocultas, salidas */
    if (fread(magic, 1, sizeof magic, f) != sizeof magic || memcmp(magic, "MLPVEC01", 8) != 0
        || fread(dims, sizeof dims[0], 4, f) != 4
        || dims[1] != MLP_IN || dims[2] != MLP_HIDDEN || dims[3] != MLP_OUT) {
        fprintf(stderr, "%s no tiene vectores para esta red (%d→%d→%d)\n", path, MLP_IN,
                MLP_HIDDEN, MLP_OUT);
        return 1;
    }

    printf("Checkpoint de la Etapa 3: mlp.c contra forward_int() de Python, %u imágenes\n",
           (unsigned)dims[0]);
    struct record r;
    struct mlp_result res;
    long correct = 0;
    for (unsigned n = 0; n < dims[0]; n++) {
        if (!read_record(f, &r)) {
            fprintf(stderr, "%s está truncado: faltan datos en la imagen %u\n", path, n);
            return 1;
        }
        mlp_infer(r.x, &res);
        for (int j = 0; j < MLP_HIDDEN; j++) {
            compare(n, ACC1, j, res.acc1[j], r.acc1[j]);
            compare(n, H, j, res.h[j], r.h[j]);
        }
        for (int k = 0; k < MLP_OUT; k++)
            compare(n, ACC2, k, res.acc2[k], r.acc2[k]);
        compare(n, DIGIT, 0, res.digit, r.digit);
        correct += res.digit == r.label;
    }
    if (fgetc(f) != EOF) {
        fprintf(stderr, "%s tiene más datos de los que dice su encabezado\n", path);
        return 1;
    }
    fclose(f);

    const long count[N_KINDS] = {(long)dims[0] * MLP_HIDDEN, (long)dims[0] * MLP_HIDDEN,
                                 (long)dims[0] * MLP_OUT, (long)dims[0]};
    long total = 0;
    for (int k = 0; k < N_KINDS; k++) {
        printf("  %s: %ld distintos de %ld\n", KIND_NAME[k], diffs[k], count[k]);
        total += diffs[k];
    }
    printf("  accuracy en C: %.2f%% (%ld de %u)\n", 100.0 * correct / dims[0], correct,
           (unsigned)dims[0]);
    printf("  → %s\n", total == 0 ? "OK: idéntico bit a bit" : "FALLA");
    return total == 0 ? 0 : 1;
}
