# MPI Tutorial 2 — Three Independent Parts

> 中文版: [README.md](./README.md)

This folder organises Tutorial 2 into three independent parts so nothing
gets mixed together:

```text
mpi_tutorial2/
├── classroom_minimpi/   ① Classroom interactive = MiniMPI (teaching MPI
│                           simulation runtime)
├── assignment1/         ② Assignment 1 (student implementation + public
│                           checker)
└── real_mpi/            ③ Real MPI Server Demo (server environment)
```

| Part | Purpose | How to use |
|---|---|---|
| `classroom_minimpi/` | **Understand**: the whole class runs Naive / Recursive Doubling / Ring etc. teaching collectives over real TCP and sees Send/Recv/Round/Barrier/Timing | `cd classroom_minimpi && python3 teacher.py --size 4`; students open `worker.py` |
| `assignment1/` | **Implement**: students write a real MPI program that does Reduce/AllReduce and self-test it with the public checker | `python3 check_submission.py <your-dir>`; grading uses hidden cases (the private grader is never pushed) |
| `real_mpi/` | **Connect to Reality**: see how real `MPI_Reduce`/`MPI_Allreduce` are written and how fast they really are (run on the server) | `make && mpirun -n 4 ./reduce_allreduce_demo`; `bash scripts/run_benchmark.sh` |

Learning loop:

```text
Understand (MiniMPI Classroom)
  → Run (classroom interactive)
  → Implement (Assignment 1)
  → Self-test (Public Checker)
  → Auto-grade (Hidden Cases, Instructor-Only)
  → Connect to Reality (Real MPI Server Demo)
```

See each sub-folder's `README.md` for details. MiniMPI
(`classroom_minimpi/`) itself is frozen; this layer is organisation only.

## Language & roles (so students are not confused)

```text
① classroom_minimpi/   Python-based teaching visualization (understand patterns)
② assignment1/         Student assignment = C + MPI (the only student language)
③ real_mpi/            Native C + MPI demos and benchmark
```

- MiniMPI is a Python teaching system used to VISUALIZE communication
  patterns;
- **Assignment 1 and Real MPI both use native C MPI APIs**;
- Python is used only by the course tools and the auto-grader
  (check.py / grader) — it is NOT a student submission language.

```text
Classroom MiniMPI  →  Understand
Real MPI Demo (C)  →  Connect to Reality
Assignment 1 (C)   →  Implement
Public Checker     →  Self-test
Private Grader     →  Grade
```
