# Lecture 01 Tutorial — CPU vs GPU 对照实验

一套**可重复执行**的 CPU/GPU 对照实验，用**真实测量数据**回答三个问题：

> **总命题：CPU = 少量强线程；GPU = 大量轻线程。**

| 实验 | 研究的问题 |
| --- | --- |
| Experiment 1 | 为什么一个 CPU thread 通常比一个 GPU thread 强？（单线程：依赖链 / ILP / 分支预测） |
| Experiment 2 | 数学工作完全相同、线程数固定，只改变数据/工作映射，为什么性能损失不同？（Cache/Coalescing/Load-Balance/Warp-Divergence） |
| Experiment 3 | 随着线程数量增加，CPU 与 GPU 的吞吐扩展为什么不同？（scaling curve） |
| AI GEMM | 用 `C = A × B` 验证为什么现代 AI workload 更适合 GPU。 |

主线与 Lecture 01 中 CPU 的乱序执行（OoO）、超标量（Superscalar）、分支预测，以及 GPU 的 Warp/SIMT 与大规模数据并行逻辑一致。

---

## 0. 教学主线（不可偏离）

```text
Experiment 1 (threads = 1)
├─ CPU worker 强：能利用 ILP (OoO + Superscalar)、Branch Prediction
└─ GPU worker 弱：单线程延迟高、无强分支预测

Experiment 2 (thread count 固定, total work 固定)
├─ CPU: Cache Locality / Hardware Prefetch / Load Balance
└─ GPU: Coalesced Access / Memory Transaction / Warp Divergence
      ==> Same Math ≠ Same Hardware Cost

Experiment 3 (mapping 规则, workload 固定, 不断增加 threads)
├─ CPU: Few Strong Threads, 很快达到硬件线程上限
└─ GPU: Massive Lightweight Threads, 靠数量提升吞吐

AI GEMM ==> Massive Regular Parallelism ==> GPU-centric AI Computing
```

---

## 1. 环境要求

| 组件 | 用途 | 是否必需 |
| --- | --- | --- |
| Linux / Ubuntu 22.04+ 或 macOS | 运行 | 推荐 Linux |
| `g++`（C++17） | 编译 CPU 程序 | **必需** |
| `nvcc` + NVIDIA GPU | 编译/运行 GPU 程序 | 可选（无则 GPU 部分跳过） |
| `perf`（Linux） | CPU 硬件计数器（IPC/cache/branch） | 可选（无则硬件列填 `NA`） |
| NVIDIA Nsight Compute (`ncu`) | GPU 硬件指标（transaction/occupancy） | 可选 |
| Python 3 + `pandas` + `matplotlib` | 画图与生成报告 | **必需** |
| OpenBLAS / CBLAS（Linux）或 Apple Accelerate（macOS） | CPU GEMM | 可选（无则用内置 fallback） |

```bash
# Ubuntu 一次性安装（示例）
sudo apt-get update
sudo apt-get install -y build-essential linux-tools-common linux-tools-generic \
     libopenblas-dev python3 python3-pip
pip3 install pandas matplotlib

# macOS
pip3 install pandas matplotlib   # Accelerate 框架系统自带
```

所有硬件信息在运行开始时自动记录到 `results/system_info.txt`（CPU 型号/核心数/频率、GPU 型号/CUDA capability/SM 数、CUDA/driver 版本等）。

---

## 2. 目录结构

```text
lecture01_tutorial/
├── README.md
├── Makefile
├── CMakeLists.txt
├── common/
│   ├── common.h          # 共享：计时/统计/CSV/CLI/正确性
│   └── cuda_utils.h      # 共享：CUDA 错误检查/事件计时/permutation
├── exp1_single_thread/   # cpu.cpp, gpu.cu, common.h
├── exp2_mapping/         # cpu_memory/gpu_memory/cpu_control/gpu_control
├── exp3_scaling/         # cpu.cpp, gpu.cu, common.h
├── ai_gemm/              # cpu_gemm.cpp, gpu_gemm.cu, common.h
├── scripts/
│   ├── common.sh         # 共享 shell 工具（perf 检测 / run_cpu）
│   ├── perf_merge.py     # 把 perf 计数器合并进 CSV
│   ├── detect_blas.sh    # 检测 CBLAS
│   ├── collect_system_info.sh
│   ├── run_exp1.sh / run_exp2.sh / run_exp3.sh / run_gemm.sh
│   ├── run_all.sh        # 一键全跑
│   ├── plot_results.py   # 生成全部图
│   └── make_summary.py   # 生成 results/summary.md
├── results/              # 运行输出（CSV / system_info.txt / summary.md）
└── figures/              # 生成的图
```

---

## 3. 快速开始

```bash
# 1. 编译（CPU 总会编译；检测到 nvcc 才会编译 GPU 部分）
make                 # 或  cmake -B build && cmake --build build

# 2. 一键运行全部实验 + 画图 + 生成报告
bash scripts/run_all.sh

# 3. 查看结果
ls results/          # CSV、system_info.txt、summary.md
ls figures/          # 8+ 张图
```

也可以分步运行：

```bash
bash scripts/collect_system_info.sh
bash scripts/run_exp1.sh
bash scripts/run_exp2.sh
bash scripts/run_exp3.sh
bash scripts/run_gemm.sh
python3 scripts/plot_results.py
python3 scripts/make_summary.py
```

> CMake 构建会把二进制放到 `build/bin`。运行脚本默认在项目根 `bin/` 下找二进制，因此使用 CMake 时请用
> `cmake -B build -DCMAKE_RUNTIME_OUTPUT_DIRECTORY="$PWD/bin" && cmake --build build`，
> 或把 `build/bin` 复制/软链到根目录 `bin/`。

---

## 3.5 课堂逐步演示（demo-friendly）

每个 binary 独立可运行、每个 case 独立可观察、每个 profiler 独立可调用，方便老师「看代码 → 问预测 → 手动跑一个 case → 看数据 → 换 case → 对比 → 开 profiler → 下结论」。

**人类可读输出**（所有 binary 支持 `--format human`，默认 `csv` 供脚本采集）：
```bash
./bin/exp1_cpu --case dependent --iterations 100000000 --format human
./bin/exp3_gpu --threads 262144 --size 1000000 --compute-iterations 500 --format human
```

**看任务映射 / 启动配置（不跑 benchmark）**：
```bash
./bin/exp2_cpu_control --distribution grouped --threads 8 --tasks 32 --dry-run
./bin/exp2_cpu_control --distribution mixed   --threads 8 --tasks 32 --dry-run
./bin/exp2_gpu_control --distribution grouped --threads 64 --tasks 64 --dry-run   # 按 Warp 显示
./bin/exp3_gpu       --threads 262144 --show-launch                               # threads/block/grid/warp
```

**逐线程耗时**（CPU 负载不均）：
```bash
./bin/exp2_cpu_control --distribution grouped --threads 8 --tasks 1048576 --thread-times
```

**并排对比 / profiler 单点调用**：
```bash
python3 scripts/compare_exp1.py dependent    # 或 independent / branch
bash scripts/perf_exp1_cpu.sh independent 8  # IPC
bash scripts/perf_exp1_cpu.sh branch random  # branch-miss
bash scripts/profile_exp2_cpu_memory.sh cyclic 8
bash scripts/profile_exp2_gpu_memory.sh 32       # 抽取 memory transaction/sector
bash scripts/profile_exp2_gpu_control.sh mixed   # 抽取 divergence/lane
bash scripts/profile_exp3_gpu.sh 262144          # 抽取 occupancy/SM utilization
```

**单张图独立生成**：
```bash
python3 scripts/plot_exp2_gpu_stride.py
python3 scripts/plot_exp3_scaling.py
python3 scripts/plot_gemm.py
```

这些是**独立工具**，不会被 `run_all.sh` 一键滚屏替代——老师可以随时停下、换参数、重跑。

---

## 4. 统一实验原则（已在代码中落实）

1. **CPU/GPU 数学工作一致**：两边执行相同数量的 FLOP（`FLOPs = 2N`），仅平台不同。
2. **防死代码消除**：所有结果都写入 `volatile` sink 或输出 buffer，并做 correctness check；基准循环的结果被使用，compiler 无法删除。
3. **Warm-up ≥ 3 次，正式运行 ≥ 10 次**（Experiment 3 的 sweep 默认 10 次，可用 `--iters` 调整），记录 median / mean / stddev，图表优先使用 **median**。
4. **GPU 区分两种延迟**：
   - `kernel_latency_ms`：仅 kernel 在 GPU 上的执行时间（CUDA Events）。
   - `end_to_end_ms`：launch + 数据传输 + kernel + synchronize。
   - 这样能区分「GPU 单线程本身慢」和「GPU offload 有额外 overhead」。

---

## 5. 各实验设计

### 5.1 Experiment 1 — 强 CPU 线程 vs 轻量 GPU 线程

**核心约束**：CPU = 恰好 1 个软件线程；GPU = 恰好 1 个 CUDA 线程（`kernel<<<1,1>>>`），禁止多线程/多 block。

#### 1A 依赖链（Dependent Chain）

```cpp
x = x * A + B;   // 执行 N 次，存在真实数据依赖
```

- 默认 `A=0.5, B=1.0`（收敛到 2.0，便于 correctness 检测 DCE）。
- 指标：latency / iterations-per-second / GFLOPS（`GFLOPS = 2N / (T·1e9)`）。
- CPU 额外通过 `perf` 收集 `cycles, instructions` → `IPC = instructions / cycles`。

#### 1B 独立链（Independent Chains，ILP）

- 保持总计算量 ≈ 1A，测 `chains = 1 / 2 / 4 / 8`：

```cpp
for (i = 0; i < N/4; ++i) {
    x0 = x0 * A + B;  x1 = x1 * A + B;
    x2 = x2 * A + B;  x3 = x3 * A + B;
}
```

- GPU 仍是 `<<<1,1>>>`，一个线程内执行相同的 independent chains（通过模板保证 compile-time 展开）。
- 计算 `Gain(C) = Throughput(C) / Throughput(1)`。

#### 1C 分支预测（Branch Prediction）

```cpp
if (data[i] > 0) x += f(data[i]); else x += g(data[i]);
```

- `f`/`g` 计算量几乎相同且 `noinline`（保证真实条件分支而非 cmov）。
- **predictable**：全部 `data[i] > 0`；**random**：50/50（固定 seed=42，可复现）。
- CPU：`perf stat -e cycles,instructions,branches,branch-misses` → `BranchMissRate = branch_misses / branches`。
- 损失：`Loss = 1 - Throughput(random)/Throughput(predictable)`。
- GPU 单线程这里**不研究 Warp Divergence**（只有 1 个线程），留给 Experiment 2。

---

### 5.2 Experiment 2 — 相同数学，不同映射（线程数固定）

任务：`C[i] = A[i] + B[i]`（算术极简单，突出 memory behavior）。

#### 2A 内存映射

- **CPU**（`T = min(8, physical cores)`，可 `--threads` 覆盖）：
  - `block`：`[t·N/T, (t+1)·N/T)` 连续分区（局部性好）。
  - `cyclic`：`i = tid; i < N; i += T`（跨大步长，局部性差）。
  - 指标：latency / elements/s / **Effective Bandwidth = 12N / T**（读 A 4B + 读 B 4B + 写 C 4B）。
  - 硬件：`cycles, instructions, cache-references, cache-misses, LLC-loads, LLC-load-misses`。
- **GPU**（默认 65536 逻辑线程，block=256）：
  - `stride 1`：coalesced grid-stride loop。
  - `stride s`（2/4/8/16/32）：通过 **permutation 数组**（真正的双射，每个元素恰好处理一次、不遗漏不重复、总 FLOPs/读写量相同），让 warp 内相邻线程访问相隔 `s` 的地址 → 不合并。
  - 指标：kernel latency / elements/s / Effective Bandwidth。
  - profiler：`ncu` 采集 DRAM throughput / memory transaction 相关指标（raw report 存 `results/exp2/ncu/`）。

#### 2B 工作/控制映射

- 两种任务：`heavy = 200` 次内层 FMA，`light = 20` 次；总数中始终 **50% heavy + 50% light**（总工作量固定）。
- **grouped**：前一半 heavy、后一半 light；**mixed**：H L H L 交替。
- CPU：block 分区 → grouped 导致线程间负载不均；记录 `max/min/avg` 线程时间与 `LoadImbalanceRatio = T_max / T_avg`。
- GPU：grouped 让每个 warp 内部统一（无发散）；mixed 让 warp 内 H/L 交替 → **Warp Divergence**。
- profiler：`ncu` 采集 branch efficiency / active lane / warp execution efficiency。

---

### 5.3 Experiment 3 — 少量强 CPU 线程 vs 大量轻 GPU 线程

- 只改变 **thread count**；使用最规则的映射（block 分区 / grid-stride），排除 Experiment 2 的 mapping 问题。
- Workload：完全独立的 compute-heavy element kernel

```cpp
out[i] = F(in[i]);   // F: x = x*a + b 循环 K 次（默认 K=500）
```

- **CPU sweep**：`1, 2, 4, ..., logical_hw_threads`（记录 physical/logical core 数，便于区分 SMT 区间）。
  - 指标：Latency / Throughput / GFLOPS / Speedup(`T(1)/T(p)`) / Parallel Efficiency(`Speedup/p`)。
- **GPU sweep**：`1, 2, 4, ..., min(N, 1M)`，grid-stride loop 使逻辑线程数可自由变化：
  - `G==1 → <<<1,1>>>`（与 Experiment 1 单线程点交叉验证）；`1<G≤1024 → <<<1,G>>>`；`G>1024 → <<<ceil(G/256),256>>>` + early-exit guard。
  - profiler：`ncu` 采集 occupancy / SM utilization / compute & DRAM throughput。
- **核心图** `exp3_thread_scaling_throughput.png`：横轴线程数（log），纵轴吞吐（log），CPU/GPU 同图；CPU 只画到真实硬件线程数，不补 0。
- 另出 **per-worker throughput** 图（`TotalThroughput / Threads`）帮助理解「CPU worker 单个强、GPU worker 单个轻、GPU 靠数量」。

---

### 5.4 AI GEMM 验证

- `C = A × B`（float32），**CPU 用优化 BLAS**（macOS Accelerate / Linux OpenBLAS），**GPU 用 cuBLAS**。
- 矩阵尺寸默认 `128, 256, 512, 1024, 2048, 4096`（显存足够可加 8192）。
- 指标：latency / GFLOPS（`2·n³ / T`）/ CPU-vs-GPU speedup。
- 正确性用 **随机投影校验**（比较 `C·v` 与 `A·(B·v)`，O(n²)，对所有尺寸都便宜）。
- cuBLAS 是列主序，求行主序 `C = A·B` 用「操作数交换 + OP_N」技巧（见 `ai_gemm/gpu_gemm.cu` 内注释）。

---

## 6. 输出文件

### 6.1 CSV

| 文件 | 关键列 |
| --- | --- |
| `results/exp1/dependent.csv` / `independent.csv` / `branch.csv` | platform, workload, variant, threads, iterations, latency_ms, throughput_ops_s, gflops, cycles, instructions, ipc, branches, branch_misses, branch_miss_rate, kernel_latency_ms, end_to_end_ms |
| `results/exp2/cpu_memory_mapping.csv` | mapping, threads, size, latency_ms, elements_per_s, effective_bandwidth_gbs, cycles, cache_references, cache_misses, llc_loads, llc_load_misses |
| `results/exp2/gpu_memory_mapping.csv` | stride, threads, size, latency_ms, elements_per_s, effective_bandwidth_gbs |
| `results/exp2/cpu_work_mapping.csv` | distribution, threads, tasks, latency_ms, throughput_tasks_s, max/min/avg_thread_ms, load_imbalance_ratio |
| `results/exp2/gpu_work_mapping.csv` | distribution, threads, tasks, latency_ms, throughput_tasks_s |
| `results/exp3/cpu_scaling.csv` / `gpu_scaling.csv` | threads, size, k, latency_ms, throughput_elem_s, gflops |
| `results/exp3/*_scaling_with_speedup.csv` | 上面 + speedup + parallel_efficiency |
| `results/gemm/cpu_gemm.csv` / `gpu_gemm.csv` | matrix_size, latency_ms, gflops |

GPU 不适用的硬件字段填 `NA`。

### 6.2 图（至少 8 张，实际生成 9 张）

1. `exp1_cpu_ipc_vs_independent_chains.png` — CPU 单线程 IPC 随独立链数变化
2. `exp1_cpu_gpu_single_thread_throughput.png` — 单 worker 计算能力 CPU vs GPU
3. `exp1_branch_performance_loss.png` — 分支预测性能损失
4. `exp2_cpu_mapping_bandwidth.png` — block vs cyclic 有效带宽
5. `exp2_gpu_stride_bandwidth.png` — stride 与有效带宽
6. `exp2_cpu_gpu_loss_comparison.png` — **实验二总结图**（Memory/Work 两种不规则 × CPU/GPU 损失）
7. `exp3_thread_scaling_throughput.png` — **核心图**（线程数 vs 吞吐，log-log）
8. `ai_gemm_gflops.png` — GEMM 矩阵尺寸 vs GFLOPS
9. `exp3_per_worker_throughput.png` — 每 worker 吞吐（教学观察）

### 6.3 报告

- `results/system_info.txt` — 硬件/软件环境。
- `results/summary.md` — 自动回答 Q1~Q6（CPU/GPU 延迟比、ILP 变化、分支损失、映射损失、scaling 饱和点、GEMM crossover）。

---

## 7. 正确性检查

所有实验先验证正确性再记时：

- **Vector Add**：全量检查 `|C[i] - (A[i]+B[i])| < ε`。
- **链式/分支/元素 kernel**：与 double 精度 reference 比较（相对误差 < 容差）。
- **GEMM**：随机投影校验 `C·v == A·(B·v)`。
- 任意 correctness failure 会**立即退出并标记 FAILED**，错误结果不会写入 performance CSV。

---

## 8. 硬件指标说明

- **CPU**：运行脚本检测到 `perf` 且可用时，用 `perf stat` 采集 `cycles/instructions/branches/branch-misses/cache*/LLC*`，由 `perf_merge.py` 合并进 CSV 并计算 `ipc`、`branch_miss_rate`。无 `perf`（如 macOS）时这些列填 `NA`。
- **GPU**：`ncu`（Nsight Compute）best-effort 采集，raw report 存 `results/exp2/ncu/`、`results/exp3/ncu/`。不同 ncu 版本 metric 名不同，因此**不硬编码 metric 名**，只保存原始 report，由用户按版本查看对应字段。
- 每个结论都尽量同时给出 **Application-level evidence** 与 **Hardware-level evidence**，不写「因为 coalescing 不好所以慢」这类只有结论的话。

---

## 9. 特别约束（实现者必须遵守）

> **不要预设实验结果**：CPU/GPU 的绝对性能、下降比例、cache miss、branch miss、profiler 数值都必须来自真实测量。
>
> **不要为了「漂亮结论」修改数据**。
>
> 对每个结论分别给出 application-level 与 hardware-level 证据；无数据就写 `NA`，GEMM 无 crossover 就写「No crossover observed in tested range.」，**不能伪造**。

---

## 10. CLI 参数参考

```bash
# Experiment 1
./bin/exp1_cpu  --case dependent --iterations 100000000
./bin/exp1_cpu  --case independent --chains 4 --iterations 100000000
./bin/exp1_cpu  --case branch --data random --iterations 10000000
./bin/exp1_gpu  --case independent --chains 4 --iterations 100000000

# Experiment 2
./bin/exp2_cpu_memory  --mapping block --threads 8 --size 100000000
./bin/exp2_gpu_memory  --stride 32 --threads 65536 --size 16777216
./bin/exp2_cpu_control --distribution grouped --threads 8 --tasks 1048576
./bin/exp2_gpu_control --distribution mixed --threads 65536 --tasks 1048576

# Experiment 3
./bin/exp3_cpu --threads 8 --size 1000000 --compute-iterations 500
./bin/exp3_gpu --threads 65536 --size 1000000 --compute-iterations 500

# GEMM
./bin/ai_gemm_cpu --sizes 128,256,512,1024,2048,4096
./bin/ai_gemm_gpu --sizes 128,256,512,1024,2048,4096
```

通用参数：`--warmup 3`、`--iters 10`、`--csv-file path`（追加写一行，首次自动写表头）。

---

## 11. 常见问题

- **macOS 上跑**：没有 `perf`、没有 `nvcc`，所以硬件计数器列和 GPU 行是 `NA`/缺失，但 CPU 的应用级指标（latency/throughput/GFLOPS）全部可测，图会优雅地只画 CPU 部分。
- **Linux 上 perf 权限不足**：`kernel.perf_event_paranoid` 限制时脚本自动检测并退化为 `NA`。
- **nvcc `-arch=native` 报错**：`make gpu CUDA_ARCH=-arch=sm_86`（换成你的 GPU 架构）。
- **GEMM 用了 fallback**：说明没装 OpenBLAS/CBLAS，正式结果请 `sudo apt-get install libopenblas-dev`。
- **图里 CPU 曲线被压扁**：核心 scaling 图用 log-log 是为了同时看清 CPU（~1e8）与 GPU（~1e11）的吞吐；这是教学上的有意选择。

---

## 12. 最终教学逻辑（README 主线）

```text
Experiment 1 → CPU worker 强 / GPU worker 弱
Experiment 2 → Same Math ≠ Same Hardware Cost
Experiment 3 → CPU 少量强线程 vs GPU 大量轻线程
AI GEMM       → Massive Regular Parallelism → GPU-centric AI Computing
```
