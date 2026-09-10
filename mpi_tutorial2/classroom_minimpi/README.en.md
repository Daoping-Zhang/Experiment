# MPI Tutorial 2 — Classroom Collective Communication Runtime (MiniMPI)

> 中文版: [README.md](./README.md)

MiniMPI is a **pure-Python standard-library** teaching runtime. It does not
try to replicate production MPI — its goal is to let the whole class
genuinely take part in collective communication: every student = one
independent worker/rank, ranks exchange data over **real peer-to-peer TCP**,
and every collective is built on the `send/recv` primitives.

```
TCP transport
      ↓
  send / recv
      ↓
Naive / Tree / Ring collectives
      ↓
   Teaching Mode = the real algorithm + a data-plane allreduce-of-1 barrier
   Performance Mode = the real algorithm, no manual synchronization
```

## 1. Why this implementation is worth reading

- **Two planes kept separate**: Control Plane (student ↔ teacher: join/rank/
  peer table/algorithm choice/metric collection) and Data Plane
  (worker ↔ worker TCP frames) never mix; collective payload never passes
  through the coordinator.
- **Every collective uses only `send/recv`**: `naive_reduce/allreduce`,
  `tree_reduce` (binomial tree), `recursive_doubling_allreduce` (log2(P)
  pairwise-exchange rounds), `ring_allreduce` (reduce-scatter + allgather).
- **Teaching / Performance share the same code**: the only difference is the
  synchronization point at the end of each round:
  - Teaching: at the end of every round everyone joins a **data-plane
    Barrier = AllReduce-of-1**: every rank contributes `[1]`, rank 0 sums
    them to `world_size` and broadcasts it back — receiving `world_size`
    means "the whole class arrived"; after rank 0 collected everything it
    prints the global view and waits for ENTER before releasing the next
    round. The Start Barrier before each RUN (`comm.Barrier()`) works the
    same way.
  - Performance: no barrier; ranks run continuously and a round is just a
    log label.

### One-line teaching mapping: arguments ↔ planes
```c
MPI_Send(&value, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);   /* Tutorial 1 view */
                        // dest=1  : who receives (data-plane destination)
                        // tag=0   : plain data-plane point-to-point payload
                        //           (same tag=0 as Tutorial 1)
                        // comm    : which group — MPI_COMM_WORLD = the
                        //           membership table built at registration
```
MiniMPI data-plane tag ranges (see `minimpi/protocol.py`): each collective
module owns one small dedicated tag channel (ping_pong=0,
naive/tree/ring=101/201/301/401/501); teaching sync / Start Barrier starts
at `BARRIER_TAG_BASE=7000` (actual tag = 7000+rnd). The two ranges never
overlap and can never be matched by mistake; classroom control messages
(RUN/DONE/SHUTDOWN) travel on a separate control channel — they are not MPI
messages and have no tag.

## 1.5 The three places students need to read

**① Initialization / lifecycle** (real-MPI habits, `minimpi/mpi.py` +
`collectives_dispatch.run`):
```python
MPI.Init()                      # MPI session starts (worker does connect/join/rank/size)
comm = MPI.COMM_WORLD           # this process's communicator
rank = comm.Get_rank(); size = comm.Get_size()
# … multiple collective runs (type one integer per run → [n]×Data Size) …
total = comm.allreduce(value, op=MPI.SUM)
MPI.Finalize()                  # MPI session ends (once per process)
```

**② Synchronization point** (`comm.sync_round(r)`, visible at the end of
each round in `collectives/*.py`): in Teaching mode everyone enters a
data-plane allreduce-of-1 barrier, rank 0 prints that round and waits for
ENTER; in Performance mode it is a no-op. The algorithm itself never
branches.

**③ Collective algorithms**: `collectives/naive_reduce.py` etc. — each file
is one complete algorithm and only ever uses
`comm.send / comm.recv / combine`.

### MPI ↔ MiniMPI API map
| Standard MPI | MiniMPI (this repo) |
|---|---|
| `MPI_Init` / `MPI_Finalize` | `MPI.Init()` / `MPI.Finalize()` |
| `MPI_COMM_WORLD` | `MPI.COMM_WORLD` |
| `MPI_Comm_rank/size` | `comm.Get_rank()/Get_size()` |
| `MPI_Send(buf,cnt,type,dest,tag,comm)` | `comm.send(value, dest, tag)` |
| `MPI_Recv(...,source,tag,comm,&st)` | `value = comm.recv(source, tag)` |
| `MPI_Reduce(...,MPI_SUM,root,comm)` | `comm.reduce(value, op, root)` |
| `MPI_Allreduce(...,MPI_SUM,comm)` | `comm.allreduce(value, op)` |
| teaching extras | `comm.naive_*/tree_*/ring_*` |

## 1.6 Classroom reading path

Students mainly read:

```text
collectives/          # for every collective: only comm.send / comm.recv
```

Infrastructure (usually not expanded in class):

```text
teacher.py  worker.py  minimpi/  scripts/  tests/
```

What a student's terminal shows is the **MPI communication view**
(Round / SEND / RECV / Peer / Payload) — no socket / protocol / thread /
transport details leak through.

## 1.7 Round, Start Barrier and Timing (round-based refactor)

**RUN spec (only the three core fields)**
```python
run = {"algorithm": "recursive_doubling_allreduce", "data_size": 1024, "mode": "teaching"}
```

**Uniform flow of every RUN**
```text
type one integer → [value] × data_size
        ↓
comm.Barrier()          # Start Barrier: present in BOTH modes (fair / common start)
        ↓
collective (the same algorithm file)
```

**Data Size semantics (project-wide)**: `Data Size = N` → every rank's
**local data** = `[value] × N` (int32, i.e. Local Data per Rank = N×4 B).
It is NOT the size of one message — Ring sends one chunk per message
(N/P elements = N/P×4 B); the real per-message `payload_bytes` is shown by
the communication events.

**Both modes use the exact same algorithm**: `collectives/*.py` has one
implementation; the only difference is `comm.sync_round(rnd)` —
Teaching = round barrier (data-plane allreduce-of-1, a MiniMPI teaching
implementation, NOT a claim about the real MPI barrier algorithm),
Performance = no-op.

**Tag separation**: algorithm messages use per-module small tags
(ping_pong=0, naive/tree/ring=101/201/301/401/501), all below
`ALGO_TAG_BASE=1000`; teaching sync messages start at
`BARRIER_TAG_BASE=7000` (=7000+rnd). The ranges never overlap, so an
algorithm message can never satisfy a barrier receive and vice versa;
classroom control (RUN/DONE/SHUTDOWN) travels on a separate control
channel — not an MPI message, no tag.

**Start Barrier (before every RUN, both modes)**: `comm.Barrier()` is
internally an AllReduce-of-1 (`minimpi/barrier.py`): every rank sends
`[1]`, rank 0 sums to `world_size` and broadcasts it back. Rank 0 stamps
the time *after the whole class arrived (gather complete) and before it
broadcasts the release*, so the Round 1 / Collective Time start point is
never skewed by the order of the releases.

**Timing (two clock domains — each answers its own question)**
- **Student — LOCAL TIMELINE (this rank's own clock, ms from its own Round
  Start)**: shows the completion instants of the events that really
  happened this round — `Send / Receive / SUM / COPY Completed`. All values
  are **cumulative offsets** (`+0.19 ms` = that event finished 0.19 ms
  after Round Start); they are **NOT durations and must never be added**;
  the only additive figure for this rank's local work this round is the
  single line `Local Work Total`. Instants are recorded at the REAL
  operation sites (`send()`/`recv()` return; one
  `comm.note_operation_complete("sum"/"copy")` after the real
  `combine()`/copy in a collective; `sync_round()` entering the barrier =
  Local Work Total). UI printing and event upload happen AFTER the barrier
  arrival and never enter the timing boundary.
- **Teacher — SYNCHRONIZATION — Rank 0 Observation (only rank 0's single
  clock)**: per round, when each rank's barrier token reached rank 0 (time
  stamped on rank 0's clock by the transport) plus rank 0's own
  local-work-done instant, all re-zeroed against the first arrival:

  ```
  SYNCHRONIZATION — Rank 0 Observation
  Rank 1 arrived: +0.00 ms | waited 3.38 ms
  Rank 0 arrived: +0.09 ms | waited 3.29 ms
  ...
  Synchronization Window: 3.38 ms
  ```

  `arrived` means "rank 0 observed this rank reaching the round
  synchronization point" (includes a tiny token transmission) — NOT
  "finished exactly at". `Synchronization Window = last arrival − first
  arrival` only answers "how unevenly the ranks finished", and does NOT
  claim the whole algorithm round took that long (`Whole Round Finished`
  was removed). Tests N/O: an artificially delayed slow rank (~1 s) makes
  the window ≈1 s and the other ranks' waited ≈1 s.
- Cross-clock: never subtract — do not compute
  `TeacherArrival − StudentLocalWork`.
- Rounds without Send/without an op simply omit those lines (nothing is
  fabricated); real performance is measured only by the Performance
  Benchmark.

**Performance experiment**: Performance mode prints/uploads no per-round
teaching events. Benchmark session: at setup every rank types ONE integer;
then all timed cases run automatically in sequence, reusing that value —
each rank's value is encoded by `make_benchmark_payload(value, bytes)`
into its own raw payload (repeated `!i` pattern, byte-fair, xor combine);
rank 0 builds its own the same way. `Collective Time` runs from Start
Barrier all-ready to every rank's completion; the same (algorithm × size)
is run 3 times and the median is reported. A `Session wall time`
(incl. input / control / UI) is also printed but only describes classroom
pacing, not algorithm performance. No outcome is pre-decided (Python/TCP/
topology/machine dependent — numbers come from real measurements). The results table is
  broadcast to EVERY student rank (everyone sees the same median table).
  Default sizes stop at 4 MB (16 MB was removed so a real-LAN classroom
  demo stays short); override with MINIMPI_BENCH_SIZES="16,1024,16384" if
  needed.

## 2. Directory

```
classroom_minimpi/
├── teacher.py            # Coordinator + Rank 0 (control plane + menu)
├── worker.py             # student side: join + dedicated reader thread + run loop
├── minimpi/
│   ├── protocol.py       # frame/control message, TAG/plane constants
│   ├── transport.py      # the ONLY layer allowed to touch sockets
│   ├── communicator.py   # send/recv + value codec + combine kernels
│   ├── barrier.py        # teaching sync: data-plane allreduce-of-1
│   ├── collectives_dispatch.py# RUN → World + Start Barrier + dispatch
│   ├── teaching.py       # Teaching View semantics (phase/chunk/preview/local view)
│   ├── classroom_worker.py# student control reader + run queue (infrastructure)
│   ├── metrics.py        # CommunicationEvent / EventLog
│   └── runtime.py        # per-rank identity, events, round sync
├── collectives/          # every algorithm only calls comm.send/comm.recv
│   ├── ping_pong.py      # Demo 0: point-to-point refresh (tag=0, data plane)
│   ├── naive_reduce.py   # all-to-one -> root
│   ├── naive_allreduce.py# all-to-one + one-to-all (baseline)
│   ├── tree_reduce.py    # binomial-tree reduce (power-of-two)
│   ├── recursive_doubling_allreduce.py # log2(P) pairwise-exchange allreduce
│   └── ring_allreduce.py # reduce-scatter + allgather
├── scripts/
│   ├── check_env.py      # Python/environment check
│   ├── local_demo.py     # run teacher+workers on one machine
│   ├── verify.py         # automated acceptance (timeouts; hang = FAIL)
│   ├── run_teaching_review.py  # generate the Teaching View review tarball
│   └── _proc.py          # subprocess helpers
└── tests/                # teaching unit tests
```

## 3. Quick start

```bash
python3 scripts/check_env.py          # environment self-check (zero deps)

# Classroom: open a teacher on one machine (interactive)
python3 teacher.py --size 4 --host 0.0.0.0 --port 9000
#   Every join is logged, e.g.
#   [JOIN] 192.168.1.45:51234 -> Rank 1  (data plane ...)  [2/4 ranks ready]
#   failures print REJECTED (version mismatch / world full); leaving prints [LEAVE] Rank x
# One-click student launcher (recommended): double-click
#   scripts/start_worker_mac.command  (macOS)
#   scripts/start_worker_windows.bat  (Windows)
# and type the teacher's IP:Port. Manual equivalent (each worker has only a
# rank; type one integer per RUN):
python3 worker.py --server <teacher-ip>:9000

# Teacher menu: pick Algorithm, then set
#   Data Size (this rank's vector length; vector = [student value] * Data Size)
#   Mode (1 Teaching / 2 Performance)
# You can switch algorithm / Data Size / mode without restarting workers.
# Rank 0 also types its own integer per RUN (menu, default 1).
# A student worker prints "Input one integer:" and types n;
# the real vector = [n] * Data Size. Non-interactive (pipe/automation) = rank+1.

# Automated modes (acceptance / local tests, independent of the classroom UI)
python3 teacher.py --size 4 --demo naive_allreduce --mode teaching --auto
python3 teacher.py --size 4 --demo recursive_doubling_allreduce --mode performance --data-size 16

# One-machine full run (real subprocesses: teacher + 3 workers)
python3 scripts/local_demo.py --size 4 --demo recursive_doubling_allreduce --mode teaching
python3 scripts/verify.py            # automated acceptance
python3 scripts/run_teaching_review.py   # one-click Teaching View review package
```

Classroom data convention: `Data Size = N` means every rank's **local data =
[student initial value] × N** (N int32, i.e. **Local Data per Rank = N × 4 B**).
It is **not** the size of one message: how many bytes a message carries is
decided by the algorithm (Ring sends one chunk per message = N/P elements =
N/P × 4 B); the real message size is shown by the communication events via
`payload_bytes`. Sizes render automatically as B/KB/MB (e.g. 64 B / 1 KB /
4 MB). The final result shows **one number only** (every vector element is
equal — it is the reduced/AllReduced result); we never print a whole big
vector. Barrier sync messages never appear in the communication view.

### Version consistency (the whole class must match)

The teacher prints on startup:

```text
Version: 2.1.0 (protocol 1)
```

A worker also prints its own version and reports `version/protocol` when it
joins. If a student copy differs, the worker is refused immediately and both
sides say exactly what to do:

```text
teacher: [VERSION] rejected a worker: version mismatch: worker=0.0.0 ...
worker : [VERSION MISMATCH] ... Please update this student copy (git pull /
         re-download) and start the worker again.
```

**Fix**: `git pull` (or re-download `classroom_minimpi`) on the student
machine, then double-click `scripts/start_worker_*` again. Nothing to change
on the teacher side.

## 4. Classroom demo main line

```
Point-to-Point (Demo 0, tag=0 data plane)
   ↓
All-to-One            -> Naive Reduce (root hotspot)
   ↓
All-to-One + One-to-All -> Naive AllReduce (baseline)
   ↓
Tree Reduce (log(P) rounds, less hotspot)
   ↓
Recursive Doubling AllReduce (log2(P) rounds, bidirectional Exchange + Reduce each round)
   ↓
big messages → Ring AllReduce (reduce-scatter + allgather, N/P per round)
   ↓
Message-size benchmark (8B..4MB × three allreduces)
```

**Teaching Mode (the final Teaching View shape)**: after every round the
whole class joins on a data-plane barrier. Before every RUN the Start
Barrier is visible (student: "Local data ready / Entering MPI Barrier...";
teacher prints the Start Barrier block and runs after ENTER). Every round
has a fixed structure:

- Teacher four-part view: ① round header (Algorithm / Phase / Round x/y)
  ② `GLOBAL COMMUNICATION` (only `who → whom + payload size`; Ring shows
  `Chunk k`; Recursive Doubling shows `Rank a ⇄ Rank b ... each
  direction`; never the other ranks' full buffers; `OPERATIONS` summarises
  each rank's op: Exchange+SUM / +SUM / +COPY) ③ `RANK 0 LOCAL VIEW`
  (the teacher is a real rank 0: real Before/Send/Receive/Operation/After
  plus its own `LOCAL TIMELINE`) ④ `SYNCHRONIZATION — Rank 0 Observation`
  (arrived/window) + Teacher Control (ENTER paces the next round).
- Student local view: Algorithm / Phase / Round x/y / Rank / `My Role`
  (Sender / Receiver / +Reduce / +Copy / explicit `Idle`; Recursive
  Doubling is `Sender + Receiver + Reduce` every round) / BEFORE / SEND /
  RECEIVE / OPERATION / AFTER / `LOCAL TIMELINE` + `Local Work Total` /
  `Waiting at round synchronization...`. Payloads are real sent/received
  data; vectors preview ≤ 8 elements.
- Collectives may contain one-line teaching annotations: right after the
  real `combine()`/copy,
  `comm.note_operation_complete("sum"/"copy")` — observation only, it never
  changes the algorithm, Send/Recv/Round/Partner or data (and it makes the
  code easier to read).
- Phase semantics come from the presentation layer: Recursive Doubling
  AllReduce P=4 → only 2 rounds, each `Phase: Exchange + Reduce`
  (bidirectional exchange + SUM; no Idle/no Broadcast); Ring P=4 → 6
  rounds: 3×`Reduce-Scatter` (SUM) + 3×`AllGather` (COPY — SUM is never
  shown there).
- At the end: `Collective Complete` — AllReduce → `Result: 10` + "All
  ranks received the same reduced result."; Reduce → only the root owns the
  result.

**Performance Mode** has no manual synchronization and prints no teaching
state; after it finishes it reports `Collective Time` (Start Barrier all
ready → everyone done) and `Session wall time`.

## 5. Protocol at a glance

Data-plane frame: `[4B header_len][JSON header][raw payload]`. The header
carries `src,dst,tag,fmt,plen,rnd,phase,algo`; the payload is always raw
bytes (benchmarks never pollute data with JSON/base64). Control plane:
newline-delimited JSON.

## 6. Limits and boundaries (the teaching boundary this README declares)

- **Tree Reduce / Recursive Doubling AllReduce need a power-of-two world
  size** (the code raises a clear error; RD = log2(P) exchange rounds).
- **Ring AllReduce needs the payload length divisible by the world size**
  (this repo's default vector length = size; stated here).
- payload `op` defaults to `sum` (element-wise for int vectors); big `raw`
  messages use `xor` (bit-wise over big ints — fast in pure stdlib).
- **This is not production MPI**: it exists to teach communication models /
  hotspots / steps / transfer time / effective bandwidth. Do not claim its
  performance equals real MPI; classroom networks are noisy, so the
  benchmark has no hard pass/fail threshold.
- Env variables `MINIMPI_TEACH_PAUSE` (simulate ENTER in auto mode),
  `MINIMPI_INPUT_DELAY` (simulate slow input), `MINIMPI_LOCAL_VIEW_DELAY`
  (simulate a slow terminal), `MINIMPI_LOCAL_WORK_DELAY` (simulate a slow
  rank's local work), `MINIMPI_SHOW_WORK` (debug) are **test-only /
  automation hooks** — classroom interaction never uses them.
- Dependencies: **Python ≥ 3.8, standard library only**
  (socket/threading/struct/json/time…). macOS/Windows/Linux all work.

## 7. Terminology

- `transfer time`: the observed send/recv completion time (network +
  protocol + runtime), not pure hardware latency; measured with
  `time.perf_counter_ns()`.
- `effective bandwidth = payload_bytes / transfer_time`; bandwidth is not
  emphasised for tiny messages.
- `rank/size/MPI_COMM_WORLD/source/destination/tag`: same mental model as
  Tutorial 1.
