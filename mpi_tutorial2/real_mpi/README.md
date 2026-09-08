# Real MPI — Native MPI Collective API Examples

> English version: [README.en.md](./README.en.md)

每个 MPI Collective 一个 C 文件，展示"真实 MPI 程序里这个 API 怎么调用"。
**不做性能 benchmark**；MPI 内部怎么实现由 MPI 库决定。

```text
MiniMPI（classroom_minimpi/）
→ 可视化 collective 可能怎么实现

Real MPI Examples（本目录）
→ 真实 C MPI 程序里 native collective API 怎么调用
```

核心口径：

> MPI API defines what the collective does.
> The MPI library decides how it is implemented internally.

## 文件（一个 API 一个文件，可独立编译/运行/阅读）

```text
bcast_example.c       MPI_Bcast     rank0 的 value=42 → 全体 42
scatter_example.c     MPI_Scatter   把 [10,20,30,40] 分给各 rank
gather_example.c      MPI_Gather    rank0 收集回 10 20 30 40
reduce_example.c      MPI_Reduce    SUM → 只在 rank0：10
allreduce_example.c   MPI_Allreduce SUM → 每个 rank：10
barrier_example.c     MPI_Barrier   同步点（不是数据归约）
```

本 Tutorial 只需要这 6 个核心 API；Alltoall/Scan/Reduce_scatter/Scatterv/
Gatherv 暂不写代码示例。

## 运行

```bash
make

mpirun -n 4 ./bcast_example
mpirun -n 4 ./scatter_example
mpirun -n 4 ./gather_example
mpirun -n 4 ./reduce_example
mpirun -n 4 ./allreduce_example
mpirun -n 4 ./barrier_example

# 或一把跑：
bash run_all_examples.sh 4
```

stdout 顺序不保证（MPI 正常行为）。

## 环境

- 任意 MPI（Open MPI / MPICH）：`mpicc` / `mpirun` 在 PATH
- C：`make`、C 编译器
