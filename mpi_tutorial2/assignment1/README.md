# Assignment 1 — MPI Reduce & AllReduce（C-only）

> English version: [README.en.md](./README.en.md)

**看懂 → 运行 → 实现 → 自测 → 自动评分** 的第一个"实现"环节。

目标（非常简单）：

> 给定 P 个 Rank 的 Local Vectors，使用 **C + MPI** 编写程序：每个 Rank 读到自己
> 那一份数据，做 **Reduce（SUM → Root=Rank 0）** 和 **AllReduce（SUM → 每 Rank 一份）**，
> 并按统一 Output Contract 输出。不考 Ring/Tree/Recursive Doubling/Chunk/
> 性能优化；内部实现方式不限（可直接 `MPI_Reduce/MPI_Allreduce`，也可自己
> `MPI_Send/MPI_Recv` 实现）。

## 1. Programming Language

Assignment 1 must be implemented in **C** using MPI.
Your program will be compiled using `mpicc` and executed using `mpirun` on
the official course server.

| 语言 | 提交内容 |
|---|---|
| C + MPI（唯一学生语言） | `solution.c` + `build.sh` + `run.sh`（`README.md` 可选，不作为评分） |

> Python / C++ / mpi4py 不再作为学生提交语言。
> Python 只用于教师提供的工具：`check.py` / `check_submission.py` / grader。

## 2. Official Grading Environment

```text
OS:             Linux
MPI Runtime:    Open MPI 4.1.2
MPI Launcher:   mpirun
C Compiler:     mpicc
Language:       C
```

学生可以本地用其它兼容 MPI（MPICH / Open MPI 均可）开发，但：

> 最终评分以官方服务器环境为准。

不需要在本地强制安装 Open MPI 4.1.2。

## 3. Submission Contract（三个文件都必须提供）

```text
student_submission/
├── solution.c     REQUIRED
├── build.sh       REQUIRED
└── run.sh         REQUIRED
```

- `build.sh`：只负责用 `mpicc` 编译（不调用 mpirun、不执行测试、不下载依赖、
  不修改系统环境）。模板：

  ```bash
  #!/bin/bash
  set -e
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  mpicc -O2 "$SCRIPT_DIR/solution.c" -o "$SCRIPT_DIR/solution"
  ```

- `run.sh`：只负责启动本 MPI Rank 对应的本地程序实例。**禁止**在其中调用
  `mpirun`/`mpiexec`（外层由 grader 执行 `mpirun -n P ./run.sh input.txt`）。
  模板：

  ```bash
  #!/bin/bash
  set -e
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  exec "$SCRIPT_DIR/solution" "$@"
  ```

> 未来若拆成多个 `.c/.h`，只需在 `build.sh` 里自行调整编译命令；
> grader/checker 固定的是 **build.sh 接口**，不是"必须只有一个 solution.c"。

## 4. Input Format

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

第 r 行就是 **Rank r** 的 local vector（长度 N）。怎么读文件不限：
每个 Rank 自己打开读自己那行 / Rank 0 读完后 MPI 分发 / 其他合理方式，都接受。

## 5. 要做的事

- **Reduce**：`SUM`，root = Rank 0。只有 Rank 0 输出结果。
  例：`[1 2 3 4] [2 3 4 5] [3 4 5 6] [4 5 6 7]` → `[10 14 18 22]`。
- **AllReduce**：`SUM`；每个 Rank 最终都是 `[10 14 18 22]` 并各自输出。

## 6. Output Contract（统一、严格）

用 `printf(...)` 直接输出到 **stdout**。**不需要**生成 output.txt/result.txt。
结果行严格如下（允许任意 debug 输出，但结果行只能出现一次）：

```text
REDUCE rank=0: 10 14 18 22

ALLREDUCE rank=0: 10 14 18 22
ALLREDUCE rank=1: 10 14 18 22
ALLREDUCE rank=2: 10 14 18 22
ALLREDUCE rank=3: 10 14 18 22
```

- MPI stdout 顺序不保证 —— checker 按 `operation + rank` 解析，不看顺序。
- `REDUCE rank=0` 必须**恰好 1 次**；每个 `ALLREDUCE rank=X` **恰好 1 次**。
- 重复 = malformed output = FAIL。
- 手动保存 stdout 可选（debug）：`mpirun -n 4 ./run.sh input.txt > output.txt`。

## 7. 本地自测（Recommended）

### Step 1 — Clone

```bash
git clone https://github.com/Daoping-Zhang/Experiment.git
cd Experiment/mpi_tutorial2/assignment1
```

### Step 2 — 创建提交目录

```text
my_assignment1/
├── solution.c
├── build.sh
└── run.sh
```

### Step 3 — 一键自测（推荐）

```bash
python3 check_submission.py ./my_assignment1
```

### Step 4 — 手动运行（可选）

```bash
cd my_assignment1
./build.sh
mpirun -n 4 ./run.sh ../demo_input.txt
```

checker 自己读 input 计算 expected，不依赖 expected 文件，因此可对任意
input 使用：

```bash
python3 check.py demo_input.txt output.txt
```

### 模板（起点，只有 TODO 骨架、不含答案）

```text
examples/c_template/    # solution.c + build.sh + run.sh（TODO）
```

### 课堂演示 checker（不需要公开任何正确提交）

```bash
python3 check.py demo_input.txt examples/demo_output.txt
```

看到 `Overall: PASS`。评分时老师不会改接口与 checker 规则，只换 input cases。

## 8. 评分原则（透明）

- Public / Hidden 流程**完全一致**：`input → build.sh → mpirun -n P ./run.sh →
  stdout → check.py`；唯一区别是输入数据。
- 隐藏 case：P=4，N=4/8/16/32/128，值 [-20,20]（正/负/0），固定 seed；
  hidden values / seeds / 完整 case 集**不公布**。
- 超时（如 10 s）会杀掉整个 mpirun 进程组，按 TIMEOUT 计。

## 9. 提交

```bash
zip -r student_id_assignment1.zip my_assignment1/
```

只交压缩包（`solution.c` + `build.sh` + `run.sh`）。
