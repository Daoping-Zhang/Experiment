# Expected Outputs

Reference for what each demo should print. Order between lines from
**different processes is NOT guaranteed** — only the set of lines matters.

## hello_mpi.c

`mpiexec -n P ./bin/hello_mpi` prints `P` lines of the form
`Hello from rank <r> of <P>`, one per process, in **any order**:

- `-n 1` → exactly one line: `Hello from rank 0 of 1`
- `-n 4` → four lines, each containing `rank <0..3>` and `of 4`

## send_recv.c

`mpiexec -n 2 ./bin/send_recv` prints:

```
Rank 1 received value 42 from rank 0
```

(Rank 0 itself prints nothing.)

`mpiexec -n 1 ./bin/send_recv` prints:

```
This example requires at least 2 MPI processes.
```

and exits normally (no crash, no hang).

## ping_pong.c

`mpiexec -n 2 ./bin/ping_pong` prints four lines; each rank prints after its
blocking send/recv, but stdout interleaving still means order is best-effort.
The invariant that verify.sh checks is that **43** appears (the value that made
the round trip):

```
Rank 0 sent 42 to rank 1
Rank 1 received 42
Rank 1 sent back 43
Rank 0 received 43
```
