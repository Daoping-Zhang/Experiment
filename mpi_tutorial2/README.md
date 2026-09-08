# MPI Tutorial 2 — 三个独立部分

> English version: [README.en.md](./README.en.md)

```text
mpi_tutorial2/
├── classroom_minimpi/   ① 课堂交互 = MiniMPI（Python 教学可视化）
├── real_mpi/            ② Native MPI Collective API Examples（C）
└── assignment1/         ③ Assignment 1 = Parallel GEMM with MPI（任务定义，C）
```

学习链：

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

| 部分 | 语言 | 作用 | 快速使用 |
|---|---|---|---|
| `classroom_minimpi/` | Python | 课堂交互：可视化 Naive/Recursive Doubling/Ring 等教学 collective 的通信过程 | `cd classroom_minimpi && python3 teacher.py --size 4` |
| `real_mpi/` | C | Native collective API 示例：Bcast/Scatter/Gather/Reduce/Allreduce/Barrier，每 API 一个 .c | `make && mpirun -n 4 ./allreduce_example` |
| `assignment1/` | C | 作业：Parallel GEMM `C=A×B`（本轮只有任务定义，评分后续再定） | 见 `assignment1/README.md` |

语言定位（避免困惑）：

- **MiniMPI 用 Python**，是为了课堂可视化通信 pattern；
- **Assignment 1 与 Real MPI 都用原生 C MPI API**；
- Python 仅用于课程教学工具（如之前的 check.py/grader），**不是**学生提交语言。

MiniMPI（classroom_minimpi/）本身已冻结；本层只做组织整理。
