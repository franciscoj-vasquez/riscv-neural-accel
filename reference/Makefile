# Etapa 3: la referencia en C (mlp.c), verificada bit a bit contra Python, en la PC.
#
#   make check   genera los vectores con Python si hace falta, compila y corre el checkpoint
#   make clean   borra build/
#
# Los enteros de la red vienen de model/artifacts/mlp_int8.h, que genera model/export_c.py.
# El checkpoint se compila con los sanitizers de GCC: si el C tuviera un comportamiento
# indefinido (desborde de un entero con signo, corrimiento inválido, acceso fuera de un
# arreglo), la ejecución se corta con un error, en vez de pasar inadvertido hasta el PicoRV32.

ARTIFACTS = ../model/artifacts
CC        = gcc
CPPFLAGS  = -I$(ARTIFACTS)
CFLAGS    = -std=c11 -O2 -Wall -Wextra -Wpedantic -Werror
SAN       = -fsanitize=undefined,address -fno-sanitize-recover=all

.PHONY: check clean

# Primero los vectores: el script también reescribe mlp_int8.h, que el checkpoint compila.
check: build/vectors.bin build/check
	build/check build/vectors.bin

build/vectors.bin: $(ARTIFACTS)/mlp_int8.npz ../model/export_c.py ../model/quantize.py
	cd .. && .venv/bin/python model/export_c.py

build/check: check.c mlp.c mlp.h $(ARTIFACTS)/mlp_int8.h
	@mkdir -p build
	$(CC) $(CPPFLAGS) $(CFLAGS) $(SAN) -o $@ check.c mlp.c

clean:
	rm -rf build
