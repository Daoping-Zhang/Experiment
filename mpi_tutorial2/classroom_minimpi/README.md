> English version: [README.en.md](./README.en.md)

# MPI Tutorial 2 — Classroom Collective Communication Runtime

MiniMPI：一个**纯 Python 标准库**的教学 runtime。目标不是复刻生产 MPI，而是让全班学生真实参与 collective communication：每个学生 = 一个独立 worker/rank，worker 之间用**真实 peer-to-peer TCP** 交换数据，collective 全部建立在 `send/recv` 原语之上。

```
TCP transport
      ↓
  send / recv
      ↓
Naive / Tree / Ring collectives
      ↓
   Teaching Mode = 正常算法 + 数据面 allreduce-of-1 barrier
   Performance Mode = 正常算法，无人工同步
```

## 1. 为什么值得看这个实现

- **两层分离**：Control Plane（学生 ↔ teacher：join/rank/peer 表/算法选择/指标收集）与 Data Plane（worker ↔ worker 的 TCP 帧）互不混淆；collective payload 从不经过 coordinator。
- **全部 collective 只用 `send/recv`**：`naive_reduce/allreduce`、`tree_reduce`（二项树）、`recursive_doubling_allreduce`（log2(P) 轮双向交换）、`ring_allreduce`（reduce-scatter + allgather）。
- **Teaching / Performance 用同一套代码**，唯一差别是每轮结束后的同步点：
  - Teaching：每轮结束人人参与一个**数据面 Barrier = AllReduce-of-1**：每 rank 贡献 `[1]`，rank 0 求和得到 `world_size` 再广播回去——每个 rank 拿到 `world_size` 即"全班到齐"；rank 0 收齐后打印全局视图、按 ENTER 才放行下一轮。每次 RUN 前的 Start Barrier（`comm.Barrier()`）同理。
  - Performance：无 barrier，rank 连续执行，round 只是日志标签。

### 参数 ↔ 平面（一句话教学映射）
```c
MPI_Send(&value, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);   /* Tutorial 1 视角 */
                        // dest=1  : 发给谁（数据面去向）
                        // tag=0   : 数据面点对点 payload（Tutorial 1 同款 tag=0）
                        // comm    : 哪个组 —— MPI_COMM_WORLD = teacher 注册时建立的成员表
```
MiniMPI 数据面 tag 分区（见 `minimpi/protocol.py`）：每个 collective 模块占用
一个专属小 tag 通道（ping_pong=0，naive/tree/ring=101/201/301/401/501）；teaching
同步 / Start Barrier 从 `BARRIER_TAG_BASE=7000` 起（实际 tag = 7000+rnd）。两区
不相交、永不误配；课堂控制消息（RUN/DONE/SHUTDOWN）走独立控制信道，不是 MPI
消息、没有 tag。

## 1.5 学生只需看三处核心代码

**① 初始化 / 生命周期**（真实 MPI 习惯，`minimpi/mpi.py` + `collectives_dispatch.run`）：
```python
MPI.Init()                      # MPI session 开始（worker 内部完成 connect/join/rank/size）
comm = MPI.COMM_WORLD           # 得到本进程的 communicator
rank = comm.Get_rank(); size = comm.Get_size()
# … 多次 collective run（每次输入一个整数 → [n]×Data Size）…
total = comm.allreduce(value, op=MPI.SUM)
MPI.Finalize()                  # MPI session 结束（进程只各一次）
```

**② 同步点**（`comm.sync_round(r)`，在 `collectives/*.py` 每轮结束处可见）：
Teaching 模式 = 人人进入一个数据面 allreduce-of-1 barrier，rank0 打印该轮并等 ENTER；
Performance 模式 = 空操作。算法本身从不分支。

**③ 集合通信算法**：`collectives/naive_reduce.py` 等，每个文件就是整套算法，
只出现 `comm.send / comm.recv / combine`。

### MPI ↔ MiniMPI API 对照
| 标准 MPI | MiniMPI（本仓库） |
|---|---|
| `MPI_Init` / `MPI_Finalize` | `MPI.Init()` / `MPI.Finalize()` |
| `MPI_COMM_WORLD` | `MPI.COMM_WORLD` |
| `MPI_Comm_rank/size` | `comm.Get_rank()/Get_size()` |
| `MPI_Send(buf,cnt,type,dest,tag,comm)` | `comm.send(value, dest, tag)` |
| `MPI_Recv(...,source,tag,comm,&st)` | `value = comm.recv(source, tag)` |
| `MPI_Reduce(...,MPI_SUM,root,comm)` | `comm.reduce(value, op, root)` |
| `MPI_Allreduce(...,MPI_SUM,comm)` | `comm.allreduce(value, op)` |
| 教学扩展 | `comm.naive_*/tree_*/ring_*` |

## 1.6 课堂阅读路径

学生主要阅读：

```text
collectives/          # 每个 collective 只看 comm.send / comm.recv
```

基础设施（一般不在课堂展开）：

```text
teacher.py  worker.py  minimpi/  scripts/  tests/
```

学生终端看到的是 **MPI 通信视角**（Round / SEND / RECV / Peer / Payload），
不会暴露 socket / protocol / thread / transport 细节。

## 1.7 Round、Start Barrier 与 Timing（Round-Based Refactor）

**RUN spec（只含三个核心字段）**
```python
run = {"algorithm": "recursive_doubling_allreduce", "data_size": 1024, "mode": "teaching"}
```

**每次 RUN 的统一流程**
```text
输入一个整数 → [value] × data_size
        ↓
comm.Barrier()          # Start Barrier：两种 Mode 都有（公平起点/统一起点）
        ↓
collective（同一份算法文件）
```

**Data Size 语义（全项目统一）**：`Data Size = N` → 每 rank 的**本地数据** = `[value] × N`
（int32，即 Local Data per Rank = N×4 B）。它不是单条消息大小——Ring 每条只发一个
chunk（N/P 元素 = N/P×4 B），真实单条消息 `payload_bytes` 由通信事件展示。

**两种 Mode 用完全相同的算法**：`collectives/*.py` 只有一份实现；差异只在
`comm.sync_round(rnd)` —— Teaching=round barrier（数据面 allreduce-of-1，属于
MiniMPI 教学实现，不代表真实 MPI barrier 算法），Performance=no-op。

**Tag 分离**：算法消息用各模块专属小 tag（ping_pong=0，naive/tree/ring=
101/201/301/401/501），全部 < `ALGO_TAG_BASE=1000`；teaching 同步消息从
`BARRIER_TAG_BASE=7000` 起（=7000+rnd）。两区不相交，算法消息永不误配到
barrier 接收、反之亦然；classroom 控制（RUN/DONE/SHUTDOWN）走独立 control
channel，不是 MPI message、没有 tag。

**Start Barrier（每次 RUN 前，两种 Mode 都有）**：`comm.Barrier()` 内部是
AllReduce-of-1（`minimpi/barrier.py`）：每 rank 发 `[1]`，rank 0 求和 = `world_size`
再广播。rank 0 在"全班到齐（gather 完成）之后、广播 release 之前"打时间戳，所以
Round 1 / Collective Time 的起点不会因为 release 的顺序而偏晚。

**Timing（两个时钟域，各回答各的问题）**
- **Student — LOCAL TIMELINE（本 rank 自己的时钟，ms 相对自己的 Round Start）**：显示本轮真实发生的事件完成时刻——`Send / Receive / SUM / COPY Completed`，全部是**累计偏移**（`+0.19 ms` = 该事件在 Round Start 后 0.19 ms 完成），**不是 duration、绝不相加**；本 rank 本轮唯一的本地总工作时间只有一行 `Local Work Total`。时刻都在**真实操作处**记录（`send()/recv()` 返回、collective 中真实 `combine()`/copy 之后的一行 `comm.note_operation_complete("sum"/"copy")`、`sync_round()` 进入 barrier 前 = Local Work Total）。UI 打印与事件上传在 barrier arrival **之后**进行，永远不进 timing boundary。
- **Teacher — SYNCHRONIZATION — Rank 0 Observation（只有 Rank 0 一只时钟）**：每轮 barrier 每个 rank 的 token 到达 Rank 0 的时刻（transport 收到时盖 Rank 0 时钟）+ Rank 0 自己完成本地工作的时刻；全部以第一个到达者归零：

  ```
  SYNCHRONIZATION — Rank 0 Observation
  Rank 1 arrived: +0.00 ms | waited 3.38 ms
  Rank 0 arrived: +0.09 ms | waited 3.29 ms
  ...
  Synchronization Window: 3.38 ms
  ```

  `arrived` 的定义是"Rank 0 观察到该 rank 到达本轮同步点"（含极小的 token 传输），**不是**"精确完成于某时刻"。`Synchronization Window = 最后到达 − 最先到达`，只回答"各 rank 完成得多不整齐"，**不代表整轮算法耗时**（已删除 `Whole Round Finished`）。测试 N/O 验证：慢 rank 人为延迟 1s → window ≈1s、其它人 waited ≈1s。
- 跨时钟**不做差值**：不计算 `TeacherArrival − StudentLocalWork`。
- 无 Send/无 op 的轮次对应行不显示（不伪造）；真实性能只由 Performance Benchmark 测量。

**性能实验**：Performance Mode 不打印/不上传每轮教学事件。Benchmark 会话：setup 时每
个 rank 只输入一次整数，之后 54 个 timed case 自动连续执行并复用该值——每个 rank 的值
经 `make_benchmark_payload(value, bytes)` 编码成自己的 raw payload（`!i` pattern 重复，
字节级公平、xor combine），rank0 同构；`Collective Time` 从 Start Barrier 到齐到全体
完成；同一（算法×尺寸）跑 3 次取 median。另有 `Session wall time`（含输入/控制/UI），
只说明课堂节奏、不是算法性能。数据不预设谁快（Python/TCP/拓扑/机器相关，结果来自
真实测量）。

## 2. 目录

```
classroom_minimpi/
├── teacher.py            # Coordinator + Rank 0（控制面 + 菜单）
├── worker.py             # 学生端：join + 专用读线程 + 执行 collective
├── minimpi/
│   ├── protocol.py       # 数据帧/控制消息、TAG/平面常量
│   ├── transport.py      # 唯一允许碰 socket 的层：帧协议 + 匹配队列
│   ├── communicator.py   # send/recv + 值编解码 + combine 内核
│   ├── barrier.py        # teaching 同步：数据面 allreduce-of-1
│   ├── collectives_dispatch.py# RUN → World + Start Barrier + 算法分发
│   ├── teaching.py       # Teaching View 语义层（phase/chunk/preview/local view）
│   ├── classroom_worker.py# 学生控制面读线程 + run 队列（教学基础设施）
│   ├── metrics.py        # CommunicationEvent / EventLog
│   └── runtime.py        # rank 端身份、事件、round 同步
├── collectives/          # 每个算法只调 comm.send/comm.recv
│   ├── ping_pong.py      # Demo 0：点对点复习（tag=0，数据面）
│   ├── naive_reduce.py   # all-to-one -> root
│   ├── naive_allreduce.py# all-to-one + one-to-all（baseline）
│   ├── tree_reduce.py    # 二项树 reduce（power-of-two）
│   ├── recursive_doubling_allreduce.py # tree reduce + 反向广播
│   └── ring_allreduce.py # reduce-scatter + allgather
├── scripts/
│   ├── check_env.py      # Python/环境检查
│   ├── local_demo.py     # 单机跑 teacher+workers
│   ├── verify.py         # 自动验收（含 timeout，hang 即 FAIL）
│   ├── run_teaching_review.py  # 一键生成 Teaching View 审核包（tar.gz）
│   └── _proc.py          # 子进程助手
└── tests/                # 教学单元测试
```

## 3. 快速开始

```bash
python3 scripts/check_env.py          # 环境自检（零第三方依赖）

# 课堂：一台机器上开 teacher（交互式）
python3 teacher.py --size 4 --host 0.0.0.0 --port 9000
# 学生/多终端：每个 worker（MPI 身份只有 Rank，没有名字/初始值参数；
# 每次 RUN 时各自在终端输入一个整数）
python3 worker.py --server <teacher-ip>:9000

# Teacher 主菜单选择 Algorithm 后依次设置：
#   Data Size（每个 rank 的向量长度；该 rank 的向量 = [学生初值] * Data Size）
#   Mode（1 Teaching / 2 Performance）
# 可连续换算法 / 换 Data Size / 换模式，Worker 无需重启。
# Rank 0 在每次 RUN 时也输入自己的整数（菜单提示，默认 1）；
# 学生 Worker 收到 RUN 后提示 "Input one integer:"，输入 n，
# 实际向量 = [n] * Data Size。非交互(管道/自动化)默认 = rank+1。

# 自动模式（验收 / 本地测试，不受课堂 UI 影响）
python3 teacher.py --size 4 --demo naive_allreduce --mode teaching --auto
python3 teacher.py --size 4 --demo recursive_doubling_allreduce --mode performance --data-size 16

# 单机一把跑（真实子进程 teacher + 3 workers）
python3 scripts/local_demo.py --size 4 --demo recursive_doubling_allreduce --mode teaching
python3 scripts/verify.py            # 自动验收
python3 scripts/run_teaching_review.py   # 一键生成 Teaching View 审核包
```

课堂演示数据约定：`Data Size = N` 表示每 rank 的**本地数据 = [学生初值] × N**（N 个 int32，
即 **Local Data per Rank = N × 4 B**）。它**不是**单条消息的大小：每条消息携带多少
由算法决定（Ring 每条只发一个 chunk = N/P 元素 = N/P × 4 B），真实消息大小由
通信事件视图按 `payload_bytes` 展示。大小显示自动转 KB/MB（如 64 B / 1 KB / 4 MB）。
最终结果只显示**一个数**（向量每个元素相等，它就是归约/AllReduce 的结果），
不在终端打印整条大向量。barrier 同步消息不计入通信视图。

## 4. 课堂演示主线

```
Point-to-Point (Demo 0, tag=0 数据面)
   ↓
All-to-One            -> Naive Reduce（root 热点）
   ↓
All-to-One + One-to-All -> Naive AllReduce（baseline）
   ↓
Tree Reduce（log(P) 轮，减热点）
   ↓
Recursive Doubling AllReduce（log2(P) 轮，每轮双向 Exchange + Reduce）
   ↓
大消息 → Ring AllReduce（reduce-scatter + allgather，每轮只收发 N/P）
   ↓
Message-size benchmark（8B..4MB × 三种 allreduce）
```

**Teaching Mode（Teaching View 最终形态）**：每轮结束全班在数据面 barrier 汇合。
每次 RUN 前 Start Barrier 可见（学生："Local data ready / Entering MPI Barrier..."，
teacher 打印 Start Barrier 块、ENTER 后开跑）。每轮固定结构：

- Teacher 四段视图：① 轮 Header（Algorithm / Phase / Round x/y）② `GLOBAL
  COMMUNICATION`（只显示 `谁→谁 + 数据量`，Ring 带 `Chunk k`，Recursive
  Doubling 用 `Rank a ⇄ Rank b ... each direction`；不显示其它 rank 的完整
  数据；`OPERATIONS` 汇总每 rank 的运算 Exchange+SUM / +SUM / +COPY）③
  `RANK 0 LOCAL VIEW`（teacher = 真实 rank 0，真实 Before/Send/Receive/
  Operation/After + 自己的 `LOCAL TIMELINE`）④ `SYNCHRONIZATION — Rank 0
  Observation`（arrived/window）+ Teacher Control（ENTER 控制下一轮）。
- 学生本地视图：Algorithm / Phase / Round x/y / Rank / `My Role`（Sender /
  Receiver / +Reduce / +Copy / 明确 `Idle`；Recursive Doubling 每轮都是
  `Sender + Receiver + Reduce`）/ BEFORE / SEND / RECEIVE / OPERATION /
  AFTER / `LOCAL TIMELINE` + `Local Work Total` / `Waiting at round
  synchronization...`。payload 都是**真实收发数据**，vector 预览 ≤8 元素。
- collectives 允许一行教学 annotation：真实 `combine()`/copy 之后紧跟
  `comm.note_operation_complete("sum"/"copy")`（只加观测、不改算法、不改变
  Send/Recv/Round/Partner/数据），学生读起来反而更清楚操作何时完成。
- Phase 语义由 presentation 层给出：Recursive Doubling AllReduce P=4 → 只 2 轮，
  每轮 `Phase: Exchange + Reduce`（双向交换 + SUM，无 Idle/无 Broadcast）；Ring
  P=4 → 6 轮：3×`Reduce-Scatter`（SUM）+ 3×`AllGather`（COPY，不显示 SUM）。
- 结束时 `Collective Complete`：AllReduce → `Result: 10` + "All ranks received
  the same reduced result."；Reduce → 只 root 拥有结果。

**Performance Mode** 无人工同步、不打印任何教学 state，跑完给出 `Collective Time`
（Start Barrier 到齐 → 全体完成）与 `Session wall time`。

## 5. 协议速览

数据面帧：`[4B header_len][JSON header][raw payload]`。header 含
`src,dst,tag,fmt,plen,rnd,phase,algo`；payload 始终 raw bytes（benchmark 不用 JSON/base64 污染数据）。控制面：逐行 JSON。

## 6. 限制与边界（README 声明的教学边界）

- **Tree Reduce / Recursive Doubling AllReduce 需要 power-of-two world size**（代码会明确报错；RD 为 log2(P) 轮双向交换）。
- **Ring AllReduce 需要 payload 长度能被 world size 整除**（本仓库默认向量长度=size；README 明示）。
- payload `op` 默认 `sum`（int 向量逐元素和）；`raw` 大消息用 `xor`（大整数按位，纯 stdlib 也快）。
- **这不是生产 MPI**：它只为教学复现"通信模型 / 热点 / 步数 / transfer time / effective bandwidth"，不要声称性能等同真实 MPI；校园网噪声大，benchmark 不设硬性 pass/fail 阈值。
- 环境变量 `MINIMPI_TEACH_PAUSE`（auto 模式模拟 ENTER）、`MINIMPI_INPUT_DELAY`（模拟慢输入）、`MINIMPI_LOCAL_VIEW_DELAY`（模拟慢终端）、`MINIMPI_LOCAL_WORK_DELAY`（模拟慢 rank 本地工作）、`MINIMPI_SHOW_WORK`（debug）均为 **test-only / 自动化钩子**，课堂交互不使用它们。
- 依赖：**Python ≥ 3.8，仅标准库**（socket/threading/struct/json/time…）。macOS/Windows/Linux 均可。

## 7. 术语对应

- `transfer time`：观测到的 send/recv 完成耗时（含网络+协议+runtime），非纯硬件延迟；用 `time.perf_counter_ns()`。
- `effective bandwidth = payload_bytes / transfer_time`；小消息不强调带宽。
- `rank/size/MPI_COMM_WORLD/source/destination/tag`：与 Tutorial 1 相同 mental model。
