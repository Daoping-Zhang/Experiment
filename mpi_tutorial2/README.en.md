# MPI Tutorial 2 — Three Independent Parts

> 中文版: [README.md](./README.md)

```text
mpi_tutorial2/
├── classroom_minimpi/   ① classroom interactive = MiniMPI (Python teaching
│                           visualization)
├── real_mpi/            ② Native MPI Collective API Examples (C)
└── assignment1/         ③ Assignment 1 = Parallel GEMM with MPI (task
                            definition, C)
```

Learning chain:

```text
classroom_minimpi/
Understand how collective algorithms work
        ↓
real_mpi/
Learn how native MPI collective APIs are called in C
        ↓
assignment1/
Use collectives to build a parallel GEMM
```

| Part | Language | Purpose | Quick use |
|---|---|---|---|
| `classroom_minimpi/` | Python | classroom interaction: visualise how Naive / Recursive Doubling / Ring etc. teaching collectives communicate | `cd classroom_minimpi && python3 teacher.py --size 4` |
| `real_mpi/` | C | Native collective API examples: Bcast/Scatter/Gather/Reduce/Allreduce/Barrier, one .c per API | `make && mpirun -n 4 ./allreduce_example` |
| `assignment1/` | C | Assignment: Parallel GEMM `C=A×B` (task definition only this round; grading later) | see `assignment1/README.md` |

Language roles (so students are not confused):

- **MiniMPI uses Python** to VISUALIZE communication patterns in class;
- **Assignment 1 and Real MPI both use native C MPI APIs**;
- Python is only used by course tools (e.g. earlier check.py/grader) — it is
  never a student submission language.

MiniMPI (`classroom_minimpi/`) itself is frozen; this layer is organisation
only.

> Student pre-class setup guide: [STUDENT_SETUP.en.md](./STUDENT_SETUP.en.md)
