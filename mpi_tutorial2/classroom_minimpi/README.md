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
个 rank 只输入一次整数，之后所有 timed case 自动连续执行并复用该值——每个 rank 的值
经 `make_benchmark_payload(value, bytes)` 编码成自己的 raw payload（`!i` pattern 重复，
字节级公平、xor combine），rank0 同构；`Collective Time` 从 Start Barrier 到齐到全体
完成；同一（算法×尺寸）跑 3 次取 median。另有 `Session wall time`（含输入/控制/UI），
只说明课堂节奏、不是算法性能。数据不预设谁快（Python/TCP/拓扑/机器相关，结果来自
真实测量）。结果表会**广播到每个 student rank 终端**（每人看到同一张
median 表）。默认尺寸到 4 MB（16 MB 已移除，避免真实局域网课堂演示过久）；可用
`MINIMPI_BENCH_SIZES=16,1024,16384` 这类环境变量临时改尺寸。

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
│   ├── local_benchmark.py# 单机跑完整 Performance Benchmark（输入 --size rank 数；看 loopback baseline）
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
#   --size 是**容量**（最多几个 rank），不是"必须到齐几个"。
#   交互课堂只要 ≥2 个 rank 且不再有人加入（静默 3 秒）就开始上课；
#   迟到的同学随时还能加入，中途退出的同学也不会让课堂卡住。
#   每个学生加入/退出/重排都会打印：
#   [JOIN] 192.168.1.45:51234 -> Rank 1  (data plane ...)  [2/4 ranks ready - capacity]
#   [LEAVE] Rank 2 disconnected
#   [ROSTER] rank compaction: 3 -> 2
#   [ROSTER] Rank 2 re-welcomed (world size is now 2)
#   失败会打印 REJECTED 原因（版本不一致 / world 已满 / 正在跑）
# 学生一键启动（推荐）：Mac 双击 scripts/start_worker_mac.command；
#   Windows 双击 scripts/start_worker_windows.bat —— 输入老师 IP:Port 即连接。
# 手动方式（等价；每个 worker 的 MPI 身份只有 Rank，每次 RUN 各自输入一个整数）：
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
python3 scripts/local_benchmark.py --size 4   # 单机跑完整 benchmark（loopback baseline）
python3 scripts/run_teaching_review.py   # 一键生成 Teaching View 审核包
```

课堂演示数据约定：`Data Size = N` 表示每 rank 的**本地数据 = [学生初值] × N**（N 个 int32，
即 **Local Data per Rank = N × 4 B**）。它**不是**单条消息的大小：每条消息携带多少
由算法决定（Ring 每条只发一个 chunk = N/P 元素 = N/P × 4 B），真实消息大小由
通信事件视图按 `payload_bytes` 展示。大小显示自动转 KB/MB（如 64 B / 1 KB / 4 MB）。
最终结果只显示**一个数**（向量每个元素相等，它就是归约/AllReduce 的结果），
不在终端打印整条大向量。barrier 同步消息不计入通信视图。

### 版本一致性（务必全班同版本）

Teacher 启动会打印：

```text
Version: 2.2.0 (protocol 2)
```

Worker 启动也会打印自己的版本，并在 join 时把 `version/protocol` 一起上报。若与学生机上的副本
**版本不一致**，worker 会被当场拒绝，双方都会给出明确提示：

```text
teacher: [VERSION] rejected a worker: version mismatch: worker=0.0.0 (protocol 1),
         teacher=2.2.0 (protocol 2) — please UPDATE the student copy ...
worker : [VERSION MISMATCH] ... Please update this student copy (git pull /
         re-download) and start the worker again.
```

**处理方式**：在学生机 `git pull`（或重新下载 `classroom_minimpi`）后，重新双击
`scripts/start_worker_*` 即可；teacher 侧不需要改任何参数。

### 动态成员与鲁棒性（真实课堂的关键）

课堂不是批处理：有人迟到、有人合上笔记本、有人中途重启。MiniMPI 的规则是
**容量（capacity）≠ 当前规模（active size）**：

| 事件 | Teacher 行为 | 学生看到 |
| --- | --- | --- |
| 加入（含迟到） | 分配**最小空闲 rank**，world size +1，打印 `[JOIN]`；给其它 rank 重发 `C_WELCOME` 更新 peers | 新同学打印 `[ROSTER] You are now Rank r / n` |
| 空闲时退出 | 打印 `[LEAVE]`；**rank 压缩成连续的 1..N-1**，打印 `[ROSTER] rank compaction` 并重新 welcome 每个 rank | 剩余同学看到自己 rank 变了（数据面 TCP 连接会重建，不需要重启） |
| 正在跑时退出 | 打印 `[LEAVE] Rank r disconnected (during a RUN)`，立刻 `[ABORT]`：给剩余 rank 发 `C_ABORT`，**取消所有正在阻塞的 recv** | 该 RUN 打印 `[ABORT] this RUN ended early: ...`，随后回到等待下一次 RUN（进程不退出） |
| 某个 rank 卡死（进程还活着但完全不动） | **停车看门狗**：超过 `MINIMPI_RUN_TIMEOUT`（默认 600 s，按"无任何进展"计）无进展就 abort，并**点名 + 剔除**那个不再发心跳的 rank（`[ROSTER] excluding Rank r`），课堂用剩下的人继续 | 该 RUN abort；被剔除的同学唤醒后会看到控制连接断开并退出，重启 worker 即可重新加入 |
| 学生机器"消失"（合盖/拔网线，不发 FIN） | 数据面 + 控制面都开了 **TCP keepalive**（idle 8 s + 2 s×3 次探测），于是它会像正常掉线一样被 `[LEAVE]` 发现 | 同上（该 rank 被正常移除） |
| 正在跑时 teacher 消失 | —— | worker 控制连接断开 → 取消阻塞 recv → 打印原因后正常退出，不会永久卡死 |
| 讲课停在 `[ENTER]` 时发生 abort | 暂停会被取消（`(pause cancelled: this RUN was aborted)`），不会吞掉下一条菜单输入 | 该 RUN abort，菜单可继续用 |

要点：

- 每次 RUN 都用**当前**的 active size（`Running...  (World Size n)`），所以"3 个人就按 3 个 rank 跑"。
- Collective 算法代码**没有被改**：abort 是在 transport 层取消阻塞的 recv（`transport.abort_pending()`），
  算法照旧抛异常结束，教学语义（轮次/视图/timing）保持冻结。
- 每次成员变化后数据面 peers 会重建（按 rank 缓存的出站连接必须失效），否则压缩后的
  rank 会复用到别人的连接——这是最容易出错的地方，`PeerTransport.reset_peers()` 专门处理它。
- 这些场景都有自动化验收：`python3 tests/test_robustness.py`（**54 项检查，全部真实进程**）。

#### 逐流程审计：worker 在哪个时刻退出会怎样

| Worker 退出的时刻 | 行为 | 课堂是否卡住 |
| --- | --- | --- |
| 刚连上、还没读完 welcome 就断（"点完就合盖"） | teacher 按连接（不是按 rank）识别退出，打印 `[LEAVE]` 并压缩 rank | 不会 |
| 等待同学到齐（`--demo/--benchmark` 要求满员） | 打印 `[ROSTER] a rank left before the world was ready`，并每 20 s 说明还缺哪些 rank | 不会（但 demo 仍要等到满员，这是约定） |
| 网络自检（Network Check）中途退出 | 检测到成员变化就提前结束检查，未知边标 `?` 并注明 "N edges unknown" | 不会 |
| 空闲（菜单停在 `Select:`） | `[LEAVE]` + rank 压缩 + 重新 welcome | 不会 |
| RUN 进行中（collective 阻塞中） | 立即 `[ABORT]`：给其他人 `C_ABORT`，取消所有阻塞中的 recv | 不会 |
| 讲课停在 `[ENTER]` 时退出 | 同上，且暂停被取消（`pause cancelled`），不会吞掉下一条菜单输入 | 不会 |
| 两个人几乎同时退出 | 两次退出都被识别（按连接查找，rank 变了也认得），压缩后仍然一致 | 不会 |
| 进程冻住（`SIGSTOP`/睡眠，不发 FIN） | 心跳停止 → 看门狗 abort + 点名 + 剔除 | 不会 |
| 机器消失（合盖/断网，不发 FIN） | TCP keepalive → 当成普通掉线 | 不会 |
| 性能 Benchmark 进行中退出 | 立刻取消整个 benchmark session（不发一张全是 0.00/n/a 的表），回到菜单 | 不会 |
| 全部同学都退出（World Size 变成 1） | 菜单提示"现在跑就是 Rank 0 一个人"；此时 RUN 仍能正常完成 | 不会 |
| RUN 进行中有人**想加入** | 明确拒绝（`code=busy`），worker 端不再是 `VERSION MISMATCH` 而是 `[JOIN REFUSED]`，并且**自动每 3 秒重试最多 60 次**，RUN 一结束就作为迟到者加入 | 不会 |
| 世界人数变了（3 人班） | Benchmark 里 RD（需要 2 的幂）与"载荷不能被 3 整除"的 Ring 会打印 `[skip]`，表里显示 `n/a`，不再记录无意义的数字 | 不会 |

**并发正确性（踩过的坑，已修）**：加入/退出/RUN 分别由不同线程处理，所以控制面发送被
**串行化**（`Coordinator.ctrl_lock`）——否则两次并发的 `welcome` 会在同一个 socket 上交错，
让不同 rank 拿到**不同的 world size**，下一次 collective 就会配对错位而永久卡住
（这正是"所有人都输入了数字却跑不起来"的根因）。同一把锁也保证"RUN 已发出"和
"roster 已更新"不会互相插队：RUN 期间有人加入会被明确拒绝，而不是偷偷改掉正在跑的世界。
测试 F（burst join）专门盯这个：三个同学同时加入时，每个人的 world size 必须一致。

```bash
python3 tests/test_robustness.py        # 加入/退出/中断/看门狗 鲁棒性验收
MINIMPI_RUN_TIMEOUT=120 python3 teacher.py --size 8   # 想更快触发看门狗
```

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

控制面消息（课堂基础设施，不是 MPI API）：`join / welcome / run / round_done / done /
check / check_report / heartbeat / abort / shutdown`。其中 `welcome` 会在**任何成员变化后重发**用于更新
rank/size/peers；`abort` 只用于"这个 RUN 不可能完成了"（有人退出 / 看门狗）。

## 6. 限制与边界（README 声明的教学边界）

- **Tree Reduce / Recursive Doubling AllReduce 需要 power-of-two world size**（代码会明确报错；RD 为 log2(P) 轮双向交换）。
- **Ring AllReduce 需要 payload 长度能被 world size 整除**（本仓库默认向量长度=size；README 明示）。
- payload `op` 默认 `sum`（int 向量逐元素和）；`raw` 大消息用 `xor`（大整数按位，纯 stdlib 也快）。
- **这不是生产 MPI**：它只为教学复现"通信模型 / 热点 / 步数 / transfer time / effective bandwidth"，不要声称性能等同真实 MPI；校园网噪声大，benchmark 不设硬性 pass/fail 阈值。
- 环境变量 `MINIMPI_TEACH_PAUSE`（auto 模式模拟 ENTER）、`MINIMPI_INPUT_DELAY`（模拟慢输入）、`MINIMPI_LOCAL_VIEW_DELAY`（模拟慢终端）、`MINIMPI_LOCAL_WORK_DELAY`（模拟慢 rank 本地工作）、`MINIMPI_SHOW_WORK`（debug）均为 **test-only / 自动化钩子**，课堂交互不使用它们。
- 心跳：worker 每 `HEARTBEAT_S = 2.0 s` 发一条 `heartbeat`，teacher 超过 `HEARTBEAT_STALE_S = 6.0 s` 没收到就认为这个 rank "不再说话"。心跳只证明"活着"，不算"有进展"——否则看门狗永远不会为一个冻住的 rank 触发。
- 课堂鲁棒性相关：`MINIMPI_RUN_TIMEOUT`（默认 600 s）是**停车看门狗**——按"多久没有任何 rank 上报进展"计，
  不是总时长；所以老师在每轮之间讲解很久没问题，但真要卡死会被 abort。若某轮讲解会超过这个时间，
  把它调大（`MINIMPI_RUN_TIMEOUT=1800`）。
- 动态成员的边界（诚实声明）：world size 变化只发生在 RUN 之间；**RUN 进行中有人加入会被拒绝**
  （提示"a collective run is in progress"），不会让正在跑的 collective 中途换 size。中途退出的 RUN
  会被 abort 而不是"缩容继续"——真实 MPI 同样不允许；这里只是让课堂不卡死。
- 依赖：**Python ≥ 3.8，仅标准库**（socket/threading/struct/json/time…）。macOS/Windows/Linux 均可。

## 7. 术语对应

- `transfer time`：观测到的 send/recv 完成耗时（含网络+协议+runtime），非纯硬件延迟；用 `time.perf_counter_ns()`。
- `effective bandwidth = payload_bytes / transfer_time`；小消息不强调带宽。
- `rank/size/MPI_COMM_WORLD/source/destination/tag`：与 Tutorial 1 相同 mental model。
