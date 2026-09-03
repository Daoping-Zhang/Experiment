# MPI Tutorial 1 — Makefile
#
#   make          -> build the three demos into bin/
#   make clean    -> remove bin/
#
# Every step prints an explanatory line followed by the real command, so the
# compile command is easy to show and read aloud in class.
#
# Simple on purpose: this is an introductory tutorial.
# Set MPICC if mpicc is not on your PATH, e.g.:
#   make MPICC=/path/to/mpicc

MPICC ?= mpicc
CFLAGS ?= -O2 -Wall -Wextra

BIN := bin

all: $(BIN)/hello_mpi $(BIN)/send_recv $(BIN)/ping_pong

$(BIN):
	@echo "==> Creating $@/"
	mkdir -p $(BIN)

$(BIN)/hello_mpi: src/hello_mpi.c | $(BIN)
	@echo "==> Compiling $< -> $@"
	$(MPICC) $(CFLAGS) $< -o $@

$(BIN)/send_recv: src/send_recv.c | $(BIN)
	@echo "==> Compiling $< -> $@"
	$(MPICC) $(CFLAGS) $< -o $@

$(BIN)/ping_pong: src/ping_pong.c | $(BIN)
	@echo "==> Compiling $< -> $@"
	$(MPICC) $(CFLAGS) $< -o $@

clean:
	@echo "==> Removing $(BIN)/"
	rm -rf $(BIN)

.PHONY: all clean
