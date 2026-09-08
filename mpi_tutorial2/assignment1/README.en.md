# Assignment 1 — MPI Reduce & AllReduce (C-only)

> 中文版: [README.md](./README.md)

This is the first "Implement" step of the learning loop
**Understand → Run → Implement → Self-test → Auto-grade**.

Goal (very simple):

> Given the local vectors of P ranks, write a program in **C + MPI** where
> every rank reads its own data, performs **Reduce (SUM → root = rank 0)**
> and **AllReduce (SUM → every rank)**, and prints the result in the unified
> Output Contract. Ring/Tree/Recursive Doubling/Chunk/performance tuning are
> NOT required; any internal implementation is allowed (direct
> `MPI_Reduce/MPI_Allreduce`, or your own `MPI_Send/MPI_Recv`).

## 1. Programming Language

Assignment 1 must be implemented in **C** using MPI.
Your program will be compiled using `mpicc` and executed using `mpirun` on
the official course server.

| Language | Submission contents |
|---|---|
| C + MPI (the only student language) | `solution.c` + `build.sh` + `run.sh` (`README.md` optional, not graded) |

> Python / C++ / mpi4py are no longer student submission languages.
> Python is only used by the instructor-provided tools:
> `check.py` / `check_submission.py` / the grader.

## 2. Official Grading Environment

```text
OS:             Linux
MPI Runtime:    Open MPI 4.1.2
MPI Launcher:   mpirun
C Compiler:     mpicc
Language:       C
```

You may develop locally with any compatible MPI (MPICH / Open MPI), but:

> Final grading is performed on the official course server.

There is no requirement to install Open MPI 4.1.2 locally.

## 3. Submission Contract (all three files required)

```text
student_submission/
├── solution.c     REQUIRED
├── build.sh       REQUIRED
└── run.sh         REQUIRED
```

- `build.sh`: only compiles with `mpicc` (never calls mpirun, never runs
  tests, never downloads dependencies, never modifies the system).
  Template:

  ```bash
  #!/bin/bash
  set -e
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  mpicc -O2 "$SCRIPT_DIR/solution.c" -o "$SCRIPT_DIR/solution"
  ```

- `run.sh`: only launches the local program instance for one MPI rank.
  Calling `mpirun`/`mpiexec` inside it is **forbidden** (the outer layer
  runs `mpirun -n P ./run.sh input.txt`). Template:

  ```bash
  #!/bin/bash
  set -e
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  exec "$SCRIPT_DIR/solution" "$@"
  ```

> If you later split into several `.c/.h` files, adjust `build.sh` yourself;
> the checker/grader fixes the **build.sh interface**, not "must be a
> single solution.c".

## 4. Input format

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
to you: every rank opens it and reads its own row / rank 0 reads everything
and distributes with MPI / any other reasonable approach.

## 5. What to do

- **Reduce**: `SUM`, root = rank 0. Only rank 0 prints the result.
  Example: `[1 2 3 4] [2 3 4 5] [3 4 5 6] [4 5 6 7]` → `[10 14 18 22]`.
- **AllReduce**: `SUM`; every rank ends with `[10 14 18 22]` and prints it.

## 6. Output contract (strict)

Use `printf(...)` to print to **stdout**. No output.txt / result.txt is
needed. Result lines must match exactly (debug output allowed, but each
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
- `REDUCE rank=0` exactly once; each `ALLREDUCE rank=X` exactly once.
- Duplicates = malformed output = FAIL.
- Saving stdout manually is optional (debug):
  `mpirun -n 4 ./run.sh input.txt > output.txt`.

## 7. Local self-test (Recommended)

### Step 1 — Clone

```bash
git clone https://github.com/Daoping-Zhang/Experiment.git
cd Experiment/mpi_tutorial2/assignment1
```

### Step 2 — Create your submission

```text
my_assignment1/
├── solution.c
├── build.sh
└── run.sh
```

### Step 3 — One-command self-test (recommended)

```bash
python3 check_submission.py ./my_assignment1
```

### Step 4 — Manual run (optional)

```bash
cd my_assignment1
./build.sh
mpirun -n 4 ./run.sh ../demo_input.txt
```

The checker reads the input and computes the expected values itself (it
never depends on an expected file), so any input works:

```bash
python3 check.py demo_input.txt output.txt
```

### Template (starting point — TODO skeleton only, no answer)

```text
examples/c_template/    # solution.c + build.sh + run.sh (TODO)
```

### Classroom checker demo (no correct submission is published)

```bash
python3 check.py demo_input.txt examples/demo_output.txt
```

→ `Overall: PASS`. Grading keeps the interface and checker rules identical;
only the input cases change.

## 8. Grading principles (transparent)

- Public and Hidden flows are **identical**:
  `input → build.sh → mpirun -n P ./run.sh → stdout → check.py`; the only
  difference is the input data.
- Hidden cases: P=4, N=4/8/16/32/128, values [-20,20] (positives, negatives,
  zeros), fixed seeds; hidden values / seeds / the full case set are **not
  published**.
- A timeout (e.g. 10 s) kills the whole mpirun process group and counts as
  TIMEOUT.

## 9. Submission

```bash
zip -r student_id_assignment1.zip my_assignment1/
```

Submit only the archive (`solution.c` + `build.sh` + `run.sh`).
