CC      = mpicc
CFLAGS  = -O2 -Wall -Wextra
TARGETS = bcast_example scatter_example gather_example \
          reduce_example allreduce_example barrier_example

all: $(TARGETS)

%_example: %_example.c
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -f $(TARGETS)

.PHONY: all clean
