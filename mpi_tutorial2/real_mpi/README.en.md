# Real MPI Server Demo — "Connect to Reality"

> 中文版: [README.md](./README.md)

Real MPI is not another tutorial. It answers two questions:

> **What does a real MPI program look like?**
> **How fast is a real `MPI_Allreduce`?**

Keep it to 10–15 minutes in class.

## 1. Relationship with MiniMPI

| | MiniMPI Benchmark | Real MPI Benchmark |
|---|---|---|
| Runtime | Python + TCP | native MPI library |
| Algorithm | controlled teaching algorithms (Ring / Recursive Doubling …) | real `MPI_Allreduce` (algorithm chosen by the MPI implementation) |
| Purpose | understand communication patterns | see the real performance of a production MPI API |

Never compare their **absolute milliseconds directly** (different runtimes,
different semantics).

**Core classroom message**:

> MiniMPI showed us how a collective may be implemented.
> In a real MPI program, programmers normally call MPI_Allreduce directly,
> and the MPI implementation chooses how to perform it.

Do not tell students "real MPI_Allreduce is Ring / Recursive Doubling by
default": the MPI **Standard defines semantics**, the MPI **implementation
chooses the algorithm**. In MiniMPI, Ring / Recursive Doubling are just
possible communication algorithms.

## 2. Demo 1 — Reduce / AllReduce API

```bash
cd real_mpi
make
mpirun -n 4 ./reduce_allreduce_demo
```

Expected (order may differ):

```text
REDUCE root result = 10

Rank 0 AllReduce result = 10
Rank 1 AllReduce result = 10
Rank 2 AllReduce result = 10
Rank 3 AllReduce result = 10
```

Teaching data: rank i holds `i+1` (1,2,3,4) → SUM = 10.

## 3. Demo 2 — a simple performance test

```bash
make
mpirun -n 4 ./allreduce_benchmark 4096      # 4096 int32 = 16 KB
```

- Each rank times its own `MPI_Allreduce` with `MPI_Wtime`;
- a collective's real performance is decided by the **slowest rank** → each
  repetition reduces max elapsed with
  `MPI_Reduce(…, MPI_MAX, root=0)`;
- `Warmup = 3` (not counted) + `Measured = 10`; rank 0 reports the **median**.

One-command sweep:

```bash
bash scripts/run_demo.sh 4
bash scripts/run_benchmark.sh            # default ranks 1 2 4 8 × 16B..16MB
```

`run_benchmark.sh` writes `results.csv`:

```text
ranks,bytes,time_ms
4,16,...
4,1024,...
...
```

> This demo does **not** force Open MPI MCA algorithm parameters
> (Ring/Recursive Doubling…). That is Open MPI internals and does not belong
> to a first MPI tutorial. The last slide may only say:
> *Advanced MPI implementations may select among algorithms such as
> recursive doubling, ring, pipelined algorithms, and others.*

## Language linkage

Assignment 1 and this demo share the same stack: **C + MPI (mpicc/mpirun)**.
Classroom chain: MiniMPI (Python, understand) → Real MPI Demo (C, connect
to reality) → Assignment 1 (C, implement it yourself). Python is only used
by course tools — never as a student submission language.

## 4. Environment needed

- any MPI (Open MPI / MPICH …): `mpicc` / `mpirun` on PATH
- C: `make`, a C compiler
