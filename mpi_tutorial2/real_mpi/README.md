# Real MPI Server Demo — "Connect to Reality"

真实 MPI 不是另一套教程，它只回答两个问题：

> **真实 MPI 程序到底怎么写？**
> **真实 `MPI_Allreduce` 的性能是什么样？**

课堂建议 10–15 分钟。

## 1. 它和 MiniMPI 的关系

| | MiniMPI Benchmark | Real MPI Benchmark |
|---|---|---|
| 运行时 | Python + TCP | native MPI library |
| 算法 | 受控的教学算法（Ring / Recursive Doubling …） | 真实 `MPI_Allreduce`（由 MPI 实现选择算法） |
| 用途 | 理解 communication pattern | 看生产 MPI API 的真实执行性能 |

两者的**绝对毫秒数不要直接比较**（不同运行时/不同语义）。

**核心课堂话术**：

> MiniMPI showed us how a collective may be implemented.
> In a real MPI program, programmers normally call MPI_Allreduce directly,
> and the MPI implementation chooses how to perform it.

不要对学生说"真实 MPI_Allreduce 默认就是 Ring / Recursive Doubling"：
MPI **Standard 定义语义**，MPI **implementation 选择算法**。MiniMPI 里的
Ring / Recursive Doubling 只是 possible communication algorithms。

## 2. Demo 1 — Reduce / AllReduce API

```bash
cd real_mpi
make
mpirun -n 4 ./reduce_allreduce_demo
```

期望（顺序允许不同）：

```text
REDUCE root result = 10

Rank 0 AllReduce result = 10
Rank 1 AllReduce result = 10
Rank 2 AllReduce result = 10
Rank 3 AllReduce result = 10
```

教学数据：Rank i 持有 `i+1`（1,2,3,4）→ SUM = 10。

## 3. Demo 2 — 简单 Performance Test

```bash
make
mpirun -n 4 ./allreduce_benchmark 4096      # 4096 个 int32 = 16 KB
```

- 每个 rank 用 `MPI_Wtime` 测自己的 `MPI_Allreduce`；
- 一次 collective 的真实性能由**最慢 rank** 决定 → 每轮用
  `MPI_Reduce(…, MPI_MAX, root=0)` 汇总 max elapsed；
- `Warmup = 3`（不计入）＋ `Measured = 10`，Rank 0 报 **median**。

一键 sweep：

```bash
bash scripts/run_demo.sh 4
bash scripts/run_benchmark.sh            # 默认 ranks 1 2 4 8 × 16B..16MB
```

`run_benchmark.sh` 输出 `results.csv`：

```text
ranks,bytes,time_ms
4,16,...
4,1024,...
...
```

> 本 Demo **不**强制 Open MPI MCA 算法参数（Ring/Recursive Doubling…）。
> 那属于 Open MPI internals，不属于第一次 MPI Tutorial。课堂 PPT 最后一页最多写：
> *Advanced MPI implementations may select among algorithms such as
> recursive doubling, ring, pipelined algorithms, and others.*

## 4. 需要什么环境

- 任意 MPI（Open MPI / MPICH…）：`mpicc` / `mpirun` 在 PATH 中
- C：`make`、C 编译器
