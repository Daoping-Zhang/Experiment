# Assignment 1 — MPI Reduce & AllReduce

> 中文版: [README.md](./README.md)

This is the first "Implement" step of the learning loop
**Understand → Run → Implement → Self-test → Auto-grade**.

Goal (very simple):

> Given the local vectors of P ranks, write an MPI program where every rank
> reads its own data, performs **Reduce (SUM → root = rank 0)** and
> **AllReduce (SUM → every rank)**, and prints the result in the unified
> Output Contract. Ring/Tree/Recursive Doubling/Chunk/performance tuning are
> NOT required; any internal implementation is allowed (direct
> `MPI_Reduce/MPI_Allreduce`, or your own `MPI_Send/MPI_Recv`).

## 1. Allowed languages

| Language | Submission contents |
|---|---|
| C / C++ + MPI | `solution.c` + `run.sh` (optional `build.sh`) |
| Python + mpi4py | `solution.py` + `run.sh` |

Uniform entry point — every language MUST provide `run.sh`, and
**`run.sh` must NOT call mpirun itself**:

```bash
mpirun -n 4 ./run.sh input.txt
```

`run.sh` example — Python:

```bash
#!/bin/bash
exec python3 solution.py "$@"
```

`run.sh` example — C (build first with `build.sh`):

```bash
#!/bin/bash
exec ./solution "$@"
```

## 2. Input format

```text
P N
vector_for_rank_0      # N integers
vector_for_rank_1
...
vector_for_rank_P-1
```

Example:

```text
4 8
1 2 3 4 5 6 7 8
2 3 4 5 6 7 8 9
3 4 5 6 7 8 9 10
4 5 6 7 8 9 10 11
```

Row r is **rank r's** local vector (length N). How you read the file is up
to you: every rank opens the file and reads its own row / rank 0 reads
everything and distributes with MPI / any other reasonable approach.

## 3. What to do

- **Reduce**: `SUM`, root = rank 0. Only rank 0 prints the final result.
  Example: `[1 2 3 4] [2 3 4 5] [3 4 5 6] [4 5 6 7]` → `[10 14 18 22]`.
- **AllReduce**: `SUM`; every rank ends up with `[10 14 18 22]` and prints it.

## 4. Output contract (strict)

Result lines must match exactly (any debug output is allowed, but each
result line may appear only once):

```text
REDUCE rank=0: 10 14 18 22

ALLREDUCE rank=0: 10 14 18 22
ALLREDUCE rank=1: 10 14 18 22
ALLREDUCE rank=2: 10 14 18 22
ALLREDUCE rank=3: 10 14 18 22
```

- MPI stdout order is not guaranteed — the checker parses by
  `operation + rank`, ignoring order.
- `REDUCE rank=0` must appear **exactly once**; each
  `ALLREDUCE rank=X` (X = 0..P-1) **exactly once**.
- Duplicates = malformed output = FAIL.

## 5. Local self-test (recommended)

Minimal submission layout (only `run.sh` + sources are required):

```text
my_assignment1/
├── run.sh
└── solution.py        # or solution.c (optional build.sh)
```

One-command local test (the checker reads the input and computes the
expected values itself; it does NOT depend on the expected file):

```bash
cd <course>/assignment1
python3 check_submission.py ./my_assignment1
```

Manual way:

```bash
mpirun -n 4 ./run.sh demo_input.txt > output.txt
python3 check.py demo_input.txt output.txt
```

Templates (starting points with TODO skeletons — no answers):

```text
examples/python_template/   # run.sh + solution.py (TODO)
examples/c_template/        # run.sh + build.sh + solution.c (TODO)
```

Classroom checker demo (no correct submission is published): run, inside
assignment1,

```bash
python3 check.py demo_input.txt examples/demo_output.txt
```

and you see `Overall: PASS`. When grading, the interface and checker rules
stay exactly the same — only the input cases change.

## 6. Grading principles (transparent)

- Grading uses **exactly the same output rules / parsing logic** as the
  public checker, with different INPUT cases.
- Hidden cases: different vector values (positives, negatives, zeros) and
  different N (e.g. 4/8/16/32/128); P stays 4.
- The concrete hidden values / seeds / full case set are **not published** —
  hardcoding cannot pass.
- Instructor side: run `mpirun -n P ./run.sh caseNN.txt` and check with the
  same rules; a timeout (e.g. 10 s) kills the whole mpirun process group
  and counts as TIMEOUT.

## 7. Submission

```bash
zip -r student_id_assignment1.zip my_assignment1/
```

Submit only the archive (`run.sh` + sources).
