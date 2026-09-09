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

## 传 Vector（数组）怎么办？—— 不需要新类型，用 count

连续 int 数组 = 同一个 datatype + **元素个数**：把 `count=1` 换成 `N`、
buffer 指向数组即可（下面的两个调用做的事一样，一个是标量一个是 N 元素数组）：

```c
int local,  result;                 /* scalar           */
MPI_Allreduce(&local, &result, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

int local[N], result[N];            /* contiguous vector */
MPI_Allreduce(local, result, N, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
```

MPI 没有"int vector 专用内置类型"。只有当布局**不连续**（跨 stride、
矩阵子块、异构结构体）才需要 derived datatypes
（`MPI_Type_vector` / `MPI_Type_indexed` / `MPI_Type_create_struct` …，
需 `MPI_Type_commit/free`）。GEMM 的行/行分块是连续 int，用 count 即可。

## 矩阵 / 一行 / 行分块怎么做

row-major 下**一行是连续的**（第 r 行起点 = `&A[r*K]`，连续 K 个 int）；
推荐把矩阵**展平成一块连续 buffer**（`int A[M*K]`），分块行就是连续的一段：

```c
int rows = M / P;                     /* M % P == 0 时每人 rows 行 */
int *A      = malloc(sizeof(int) * M * K);     /* rank 0 读取后展平 */
int *local_A = malloc(sizeof(int) * rows * K);

MPI_Scatter(A,        rows * K, MPI_INT,   /* sendcount = 每人元素个数 */
            local_A,  rows * K, MPI_INT,
            0, MPI_COMM_WORLD);
```

结果同理：`MPI_Gather(local_C, rows*N, MPI_INT, C, rows*N, MPI_INT, 0, ...)`，
rank 0 拿到展平的整块 C（需要时可再按行做视图）。

跨 stride / 非连续才需要 derived types：取一列 → 复制或用
`MPI_Type_vector`；2D 子块 → `MPI_Type_create_subarray`；每 rank 行数不同
（M %% P != 0）→ `MPI_Scatterv / MPI_Gatherv`。

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
