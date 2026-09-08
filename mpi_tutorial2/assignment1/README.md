# Assignment 1 — MPI Reduce & AllReduce

**看懂 → 运行 → 实现 → 自测 → 自动评分** 的第一个"实现"环节。

目标（非常简单）：

> 给定 P 个 Rank 的 Local Vectors，编写一个 MPI 程序：每个 Rank 读到自己那一份
> 数据，做 **Reduce（SUM → Root=Rank 0）** 和 **AllReduce（SUM → 每 Rank 一份）**，
> 并按统一 Output Contract 输出。不考 Ring/Tree/Recursive Doubling/Chunk/
> 性能优化；内部用什么算法不限制（可直接调 `MPI_Reduce/MPI_Allreduce`，也可自己
> `MPI_Send/MPI_Recv` 实现）。

## 1. 允许的语言

| 语言 | 提交内容 |
|---|---|
| C / C++ + MPI | `solution.c` + `run.sh`（可加 `build.sh`） |
| Python + mpi4py | `solution.py` + `run.sh` |

统一入口（所有语言必须提供，且 **`run.sh` 内部不得再调用 mpirun**）：

```bash
mpirun -n 4 ./run.sh input.txt
```

`run.sh` 示例 —— Python：

```bash
#!/bin/bash
exec python3 solution.py "$@"
```

`run.sh` 示例 —— C（先用 `build.sh` 编译）：

```bash
#!/bin/bash
exec ./solution "$@"
```

## 2. Input Format

```text
P N
vector_for_rank_0      # N 个整数
vector_for_rank_1
...
vector_for_rank_P-1
```

例：

```text
4 8
1 2 3 4 5 6 7 8
2 3 4 5 6 7 8 9
3 4 5 6 7 8 9 10
4 5 6 7 8 9 10 11
```

第 r 行就是 **Rank r** 的 local vector（长度 N）。学生**怎么读文件都可以**：
每个 Rank 自己打开文件读自己那行 / Rank 0 读完后 MPI 分发 / 其他合理方式，都接受。

## 3. 要做的事

- **Reduce**：`SUM`，root = Rank 0。只有 Rank 0 输出最终结果。
  例：`[1 2 3 4] [2 3 4 5] [3 4 5 6] [4 5 6 7]` → `[10 14 18 22]`。
- **AllReduce**：`SUM`，每个 Rank 最终都有 `[10 14 18 22]` 并各自输出。

## 4. Output Contract（统一、严格）

结果行必须严格以下格式（允许任意 debug 输出，但结果行只能出现一次）：

```text
REDUCE rank=0: 10 14 18 22

ALLREDUCE rank=0: 10 14 18 22
ALLREDUCE rank=1: 10 14 18 22
ALLREDUCE rank=2: 10 14 18 22
ALLREDUCE rank=3: 10 14 18 22
```

- MPI stdout 顺序不保证 —— checker 按 `operation + rank` 解析，不看顺序。
- `REDUCE rank=0` 必须**恰好 1 次**；每个 `ALLREDUCE rank=X`（X=0..P-1）**恰好 1 次**。
- 重复 = malformed output = FAIL。

## 5. 本地自测（推荐）

学生提交目录结构（最低要求只有 `run.sh` + 源码）：

```text
my_assignment1/
├── run.sh
└── solution.py        # 或 solution.c（可加 build.sh）
```

一键本地测试（checker 自己读 input 计算 expected，不依赖 expected 文件）：

```bash
cd <course>/assignment1
python3 check_submission.py ./my_assignment1
```

人工方式：

```bash
mpirun -n 4 ./run.sh demo_input.txt > output.txt
python3 check.py demo_input.txt output.txt
```

模板（起点，含 TODO 骨架；不含答案）：

```text
examples/python_template/   # run.sh + solution.py（TODO）
examples/c_template/        # run.sh + build.sh + solution.c（TODO）
```

课堂演示 checker（不需要公开任何正确提交）：老师/同学在 assignment1 目录运行

```bash
python3 check.py demo_input.txt examples/demo_output.txt
```

看到 `Overall: PASS`；**评分时老师不会改接口与 checker 规则，只换 input cases**。

## 6. 评分原则（透明）

- 评分与公共 checker **完全同一套 Output 规则/解析逻辑**，只是**更换 Input cases**。
- 隐藏 case：不同的 vector 值（含正/负/0）与不同 N（如 4/8/16/32/128），P 固定 4。
- 具体 hidden values / seed / 完整 case 集**不公布**——不会因为 hardcode 过关。
- 课堂/老师端：`mpirun -n P ./run.sh caseNN.txt` 跑你的程序，再按相同规则核对输出；
  超时（如 10 s）会杀掉整个 mpirun 进程组并按 TIMEOUT 计。

## 7. 提交

```bash
zip -r student_id_assignment1.zip my_assignment1/
```

只交压缩包（`run.sh` + 源码即可）。
