# Real MPI — Native MPI Collective API Examples

> 中文版: [README.md](./README.md)

One C file per MPI collective, showing how the API is called in a real MPI
program. **No performance benchmark** here — how MPI implements a collective
internally is decided by the MPI library.

```text
MiniMPI (classroom_minimpi/)
→ visualize how a collective may be implemented

Real MPI Examples (this folder)
→ how native collective APIs are called in real C MPI programs
```

Key statement:

> MPI API defines what the collective does.
> The MPI library decides how it is implemented internally.

## Files (one API per file; each compiles/runs/reads on its own)

```text
bcast_example.c       MPI_Bcast     rank 0's value=42 -> every rank 42
scatter_example.c     MPI_Scatter   distribute [10,20,30,40] across ranks
gather_example.c      MPI_Gather    rank 0 collects 10 20 30 40
reduce_example.c      MPI_Reduce    SUM -> only on rank 0: 10
allreduce_example.c   MPI_Allreduce SUM -> every rank: 10
barrier_example.c     MPI_Barrier   a synchronization point (not data reduction)
```

This tutorial needs only these 6 core APIs; Alltoall/Scan/Reduce_scatter/
Scatterv/Gatherv get no code examples yet.

## Passing vectors? — no new type needed, use count

A contiguous int array = the same datatype + **count of elements**: change
`count=1` to `N` and point the buffer at the array (below: scalar vs.
N-element array, same shape otherwise):

```c
int local,  result;                 /* scalar           */
MPI_Allreduce(&local, &result, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

int local[N], result[N];            /* contiguous vector */
MPI_Allreduce(local, result, N, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
```

MPI has no special built-in "int vector" type. Derived datatypes
(`MPI_Type_vector` / `MPI_Type_indexed` / `MPI_Type_create_struct` ...,
with `MPI_Type_commit/free`) are only needed for NON-contiguous layouts
(strided access, sub-blocks, heterogeneous structs). GEMM rows / row
blocks are contiguous ints — `count` is enough.

## Run

```bash
make

mpirun -n 4 ./bcast_example
mpirun -n 4 ./scatter_example
mpirun -n 4 ./gather_example
mpirun -n 4 ./reduce_example
mpirun -n 4 ./allreduce_example
mpirun -n 4 ./barrier_example

# or all at once:
bash run_all_examples.sh 4
```

stdout order is not guaranteed (normal MPI behaviour).

## Environment

- any MPI (Open MPI / MPICH): `mpicc` / `mpirun` on PATH
- C: `make`, a C compiler
