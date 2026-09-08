# Assignment 1 — Parallel GEMM with MPI

> English version: [README.en.md](./README.en.md)

本轮为**任务定义**：用 **C + MPI** 实现并行矩阵乘 `C = A × B`。
评分细则 / hidden grading / 性能分数**后续再定**（本轮不包含）。

## 1. 计算任务

$$C = A \times B$$

| 矩阵 | 维数 | 数据类型 |
|---|---|---|
| A | M × K | int |
| B | K × N | int |
| C | M × N | int |

## 2. 并行化思路（教学背景：1D Row Partitioning）

最简单的按行划分：

```text
Matrix A
  rows 0..x   → Rank 0
  rows x..y   → Rank 1
  ...
```

每个 rank：

```text
local_C = local_A × B
```

最后把各 rank 的 `local_C` 组合成完整 C。

推荐（不强制）的 collective 流程：

```text
Rank 0 reads A and B
        ↓
MPI_Bcast   → share Matrix B (和 M/K/N)
        ↓
MPI_Scatter → distribute rows of A
        ↓
Local Matrix Multiplication
        ↓
MPI_Gather  → collect local C
        ↓
Rank 0 obtains final C
```

## 3. 环境

```text
OS:             Linux（官方评分服务器）
MPI Runtime:    Open MPI 4.1.2（兼容 MPI 均可本地开发）
MPI Launcher:   mpirun
C Compiler:     mpicc
Language:       C
```

## 4. Input Format

```text
M K N
<Matrix A: M 行，每行 K 个 int>
<Matrix B: K 行，每行 N 个 int>
```

例（B = Identity，C = A，便于人工验证）：

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

## 5. Output（只要求 Rank 0 输出最终 C，直接 stdout）

```text
RESULT M N
<row 0>
...
<row M-1>
```

例：

```text
RESULT 4 4
1 2 3 4
5 6 7 8
9 10 11 12
13 14 15 16
```

本轮**不要求** TIME_MS / Speedup / Efficiency 等输出。

## 6. 本地自测（手动，checker 后续提供）

```bash
cd examples/c_template   # 或你自己的提交目录
./build.sh
mpirun -n 4 ./run.sh ../../demo_input.txt
```

对比输出与 `demo_expected.txt`。

## 7. Starter

```text
assignment1/
├── README.md
├── demo_input.txt
├── demo_expected.txt
└── examples/c_template/
    ├── solution.c      # 只有 TODO（1..6）
    ├── build.sh
    └── run.sh
```

模板 `solution.c` 只列 TODO：MPI init → Rank0 读矩阵 → 分发数据 →
本地矩阵乘 → 收集结果 → 打印。不提供完整 `MPI_Bcast/MPI_Scatter/
MPI_Gather` 答案。

> You may find Bcast, Scatter and Gather useful.

## 8. 未定事项（后续单独开一轮）

- 评分 rubric / 分数分配
- hidden cases 与输入范围
- 是否引入 performance / scaling 分数
- 自动 checker（check.py / grader）
