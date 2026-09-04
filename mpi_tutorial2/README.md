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
- **全部 collective 只用 `send/recv`**：`naive_reduce/allreduce`、`tree_reduce/allreduce`（二项树）、`ring_allreduce`（reduce-scatter + allgather）。
- **Teaching / Performance 用同一套代码**，唯一差别是每轮结束后的同步点：
  - Teaching：每轮结束人人参与一个**数据面 barrier = allreduce 传 1**（类似 NCCL 把 barrier 做成 dummy allreduce）；rank 0 收齐即"全班完成本轮"，打印全局视图、按 ENTER 才放行下一轮。
  - Performance：无 barrier，rank 连续执行，round 只是日志标签。

### 参数 ↔ 平面（一句话教学映射）
```c
MPI_Send(&value, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);   /* Tutorial 1 视角 */
                        // dest=1  : 发给谁（数据面去向）
                        // tag=0   : 走哪条信道：0=数据面(TAG_DATA)，1=控制/同步面(TAG_CONTROL)
                        // comm    : 哪个组 —— 由 teacher 的 CONTROL plane 注册时建立的成员表
```
（collective 匹配用独立 tag；teaching 同步消息用 `CTRL_TAG_BASE` 起，见 `minimpi/protocol.py`。）

## 2. 目录

```
mpi_tutorial2/
├── teacher.py            # Coordinator + Rank 0（控制面 + 菜单）
├── worker.py             # 学生端：join + 专用读线程 + 执行 collective
├── minimpi/
│   ├── protocol.py       # 数据帧/控制消息、TAG/平面常量
│   ├── transport.py      # 唯一允许碰 socket 的层：帧协议 + 匹配队列
│   ├── communicator.py   # send/recv + 值编解码 + combine 内核
│   ├── barrier.py        # teaching 同步：数据面 allreduce-of-1
│   ├── synchronization.py# (deprecated) 旧控制面握手，已由 barrier 取代
│   ├── metrics.py        # CommunicationEvent / EventLog
│   └── runtime.py        # rank 端身份、事件、round 同步
├── collectives/          # 每个算法只调 comm.send/comm.recv
│   ├── ping_pong.py      # Demo 0：点对点复习（tag=0，数据面）
│   ├── naive_reduce.py   # all-to-one -> root
│   ├── naive_allreduce.py# all-to-one + one-to-all（baseline）
│   ├── tree_reduce.py    # 二项树 reduce（power-of-two）
│   ├── tree_allreduce.py # tree reduce + 反向广播
│   └── ring_allreduce.py # reduce-scatter + allgather
├── scripts/
│   ├── check_env.py      # Python/环境检查
│   ├── local_demo.py     # 单机跑 teacher+workers
│   ├── verify.py         # 自动验收（含 timeout，hang 即 FAIL）
│   └── _proc.py          # 子进程助手
└── tests/                # 教学单元测试
```

## 3. 快速开始

```bash
python3 scripts/check_env.py          # 环境自检（零第三方依赖）

# 课堂：一台机器上开 teacher
python3 teacher.py --size 4 --host 0.0.0.0 --port 9000
# 学生/多终端：每个 worker
python3 worker.py --server <teacher-ip>:9000 --name Alice

# 或单机一把跑（真实子进程 teacher + 3 workers）
python3 scripts/local_demo.py --size 4 --demo tree_allreduce --mode teaching
python3 scripts/local_demo.py --size 4 --demo ring_allreduce --mode performance

# 验收
python3 scripts/verify.py
```

## 4. 课堂演示主线

```
Point-to-Point (Demo 0, tag=0 数据面)
   ↓
All-to-One            -> Naive Reduce（root 热点）
   ↓
All-to-One + One-to-All -> Naive AllReduce（baseline）
   ↓
Tree Reduce / Tree AllReduce（log(P) 轮，减热点）
   ↓
大消息 → Ring AllReduce（reduce-scatter + allgather，每轮只收发 N/P）
   ↓
Message-size benchmark（8B..4MB × 三种 allreduce）
```

**Teaching Mode** 每轮结束：全班在数据面 barrier 汇合 → teacher 打印该轮全局通信视图 → 按 ENTER → 放行下一轮。学生终端各自显示自己的 local view（收发、payload、transfer time、本地值变化）。**Performance Mode** 无人工同步，跑完给出汇总时长。

## 5. 协议速览

数据面帧：`[4B header_len][JSON header][raw payload]`。header 含
`src,dst,tag,fmt,plen,rnd,phase,algo`；payload 始终 raw bytes（benchmark 不用 JSON/base64 污染数据）。控制面：逐行 JSON。

## 6. 限制与边界（README 声明的教学边界）

- **Tree Reduce / Tree AllReduce 需要 power-of-two world size**（代码会明确报错）。
- **Ring AllReduce 需要 payload 长度能被 world size 整除**（本仓库默认向量长度=size；README 明示）。
- payload `op` 默认 `sum`（int 向量逐元素和）；`raw` 大消息用 `xor`（大整数按位，纯 stdlib 也快）。
- **这不是生产 MPI**：它只为教学复现"通信模型 / 热点 / 步数 / transfer time / effective bandwidth"，不要声称性能等同真实 MPI；校园网噪声大，benchmark 不设硬性 pass/fail 阈值。
- 依赖：**Python ≥ 3.8，仅标准库**（socket/threading/struct/json/time…）。macOS/Windows/Linux 均可。

## 7. 术语对应

- `transfer time`：观测到的 send/recv 完成耗时（含网络+协议+runtime），非纯硬件延迟；用 `time.perf_counter_ns()`。
- `effective bandwidth = payload_bytes / transfer_time`；小消息不强调带宽。
- `rank/size/MPI_COMM_WORLD/source/destination/tag`：与 Tutorial 1 相同 mental model。
