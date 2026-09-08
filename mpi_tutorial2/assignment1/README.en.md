# Assignment 1 — Parallel GEMM with MPI

> 中文版: [README.md](./README.md)

This round is only the **task definition**: implement parallel matrix
multiplication `C = A × B` in **C + MPI**. Grading rubric / hidden grading /
performance scores are **defined in a later round** (not part of this one).

## 1. Computational task

$$C = A \times B$$

| Matrix | Dimensions | Data type |
|---|---|---|
| A | M × K | int |
| B | K × N | int |
| C | M × N | int |

## 2. Parallelisation idea (teaching background: 1D row partitioning)

Simplest row-wise split:

```text
Matrix A
  rows 0..x   → Rank 0
  rows x..y   → Rank 1
  ...
```

Every rank:

```text
local_C = local_A × B
```

Finally combine the ranks' `local_C` into the full C.

Recommended (not required) collective flow:

```text
Rank 0 reads A and B
        ↓
MPI_Bcast   → share Matrix B (and M/K/N)
        ↓
MPI_Scatter → distribute rows of A
        ↓
Local Matrix Multiplication
        ↓
MPI_Gather  → collect local C
        ↓
Rank 0 obtains final C
```

## 3. Environment

```text
OS:             Linux (official grading server)
MPI Runtime:    Open MPI 4.1.2 (any compatible MPI is fine for local dev)
MPI Launcher:   mpirun
C Compiler:     mpicc
Language:       C
```

## 4. Input format

```text
M K N
<Matrix A: M rows, K ints each>
<Matrix B: K rows, N ints each>
```

Example (B = Identity, so C = A — easy to verify by hand):

```text
4 4 4
1 2 3 4
5 6 7 8
9 10 11 12
13 14 15 16
1 0 0 0
0 1 0 0
0 0 1 0
0 0 0 1
```

## 5. Output (only rank 0 prints the final C, straight to stdout)

```text
RESULT M N
<row 0>
...
<row M-1>
```

Example:

```text
RESULT 4 4
1 2 3 4
5 6 7 8
9 10 11 12
13 14 15 16
```

No TIME_MS / Speedup / Efficiency output is required this round.

## 6. Local self-test (manual; an automated checker comes later)

```bash
cd examples/c_template   # or your own submission directory
./build.sh
mpirun -n 4 ./run.sh ../../demo_input.txt
```

Compare with `demo_expected.txt`.

## 7. Starter

```text
assignment1/
├── README.md
├── demo_input.txt
├── demo_expected.txt
└── examples/c_template/
    ├── solution.c      # TODO only (1..6)
    ├── build.sh
    └── run.sh
```

The template `solution.c` only lists TODOs: MPI init → rank 0 reads
matrices → distribute data → local matrix multiply → collect the result →
print. No complete `MPI_Bcast/MPI_Scatter/MPI_Gather` answer is provided.

> You may find Bcast, Scatter and Gather useful.

## 8. Open items (a separate future round)

- grading rubric / score split
- hidden cases and input ranges
- whether performance / scaling is scored
- automated checker (check.py / grader)
