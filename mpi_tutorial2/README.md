> English version: [README.en.md](./README.en.md)

# MPI Tutorial 2 — 三个独立部分

本文件夹把 Tutorial 2 拆成三个相互独立的部分，避免混杂：

```text
mpi_tutorial2/
├── classroom_minimpi/   ① 课堂交互 = MiniMPI（教学用 MPI 模拟 runtime）
├── assignment1/         ② 作业说明 = Assignment 1（学生实现 + 公共 checker）
└── real_mpi/            ③ 服务器真实运行 = Real MPI Server Demo
```

| 部分 | 作用 | 怎么用 |
|---|---|---|
| `classroom_minimpi/` | **Understand**：全班用真实 TCP 跑 Naive / Recursive Doubling / Ring 等教学 collective，看 Send/Recv/Round/Barrier/Timing | `cd classroom_minimpi && python3 teacher.py --size 4`，学生开 `worker.py` |
| `assignment1/` | **Implement**：学生写一个真实 MPI 程序完成 Reduce/AllReduce，本地用公共 checker 自测 | `python3 check_submission.py <你的目录>`；评分端换 hidden cases（私有 grader 不入库） |
| `real_mpi/` | **Connect to Reality**：看真实 `MPI_Reduce/MPI_Allreduce` 的写法和性能（服务器上跑） | `make && mpirun -n 4 ./reduce_allreduce_demo`；`bash scripts/run_benchmark.sh` |

学习闭环：

```text
看懂（MiniMPI Classroom）
  → 运行（课堂交互）
  → 实现（Assignment 1）
  → 自测（Public Checker）
  → 自动评分（Hidden Cases，Instructor-Only）
  → 对接现实（Real MPI Server Demo）
```

详细说明分别见各子目录的 `README.md`。MiniMPI（classroom_minimpi/）本身已冻结，
本层只做组织整理。
