# MPI Tutorial 1 — Makefile
#
#   make          -> build the three demos into bin/
#   make clean    -> remove bin/
#
# Simple on purpose: this is an introductory tutorial.
# Set MPICC if mpicc is not on your PATH, e.g.:
#   make MPICC=/path/to/mpicc

MPICC ?= mpicc
CFLAGS ?= -O2 -Wall -Wextra

BIN := bin

all: $(BIN)/hello_mpi $(BIN)/send_recv $(BIN)/ping_pong

$(BIN):
	mkdir -p $(BIN)

$(BIN)/hello_mpi: src/hello_mpi.c | $(BIN)
	$(MPICC) $(CFLAGS) $< -o $@

$(BIN)/send_recv: src/send_recv.c | $(BIN)
	$(MPICC) $(CFLAGS) $< -o $@

$(BIN)/ping_pong: src/ping_pong.c | $(BIN)
	$(MPICC) $(CFLAGS) $< -o $@

clean:
	rm -rf $(BIN)

.PHONY: all clean
