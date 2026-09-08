CC      = mpicc
CFLAGS  = -O2 -Wall
TARGETS = reduce_allreduce_demo allreduce_benchmark

all: $(TARGETS)

reduce_allreduce_demo: reduce_allreduce_demo.c
	$(CC) $(CFLAGS) -o $@ $<

allreduce_benchmark: allreduce_benchmark.c
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -f $(TARGETS)

.PHONY: all clean
