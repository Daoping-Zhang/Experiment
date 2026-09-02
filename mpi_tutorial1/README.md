# MPI Tutorial 1

入门、hands-on 的 MPI 教程。重点不是覆盖完整 MPI API，而是让你亲眼看到：

> **一份可执行文件，怎么被 `mpiexec -n P` 启动成 P 个独立进程，两个进程又怎么用 `MPI_Send` / `MPI_Recv` 显式交换一条消息。**

Each MPI process can be viewed as a worker responsible for a different portion of a larger computation（AI training 里每个 worker 负责一块数据做 local computation——真正的 gradient aggregation 留给 Tutorial 2）。

> 本 Tutorial **只讲点对点通信**，不讲 collective（Reduce / Allreduce / Bcast / Scatter / Gather 属于 Tutorial 2）。代码 CPU-only，C / MPI。

## 1. Learning Goals

- 用 `mpiexec -n P` 把同一个 executable 启动成多个 MPI processes
- 理解 `process` / `rank` / `size` / `MPI_COMM_WORLD`
- 理解 MPI 程序的 lifecycle：`MPI_Init` → … → `MPI_Finalize`
- 用 `MPI_Send` / `MPI_Recv` 在两个进程间显式交换数据
- 能自己 compile / run / modify 程序

## 2. Environment Check

课前先确认 MPI 环境：

```bash
./scripts/check_env.sh
```

期望看到全部 `[PASS]`（mpicc、mpiexec/mpirun、编译与运行测试）。

## 3. Build

```bash
make
```

生成：

```text
bin/hello_mpi
bin/send_recv
bin/ping_pong
```

（`make clean` 删除 `bin/`。若 `mpicc` 不在 PATH：`make MPICC=/path/to/mpicc`。）

## 4. Demo 1 — Hello MPI

```bash
mpiexec -n 1 ./bin/hello_mpi
mpiexec -n 4 ./bin/hello_mpi
```

同一个 `bin/hello_mpi`，`-n P` 让 runtime 启动 P 个独立进程；每个进程用 `MPI_Comm_rank` 拿到自己的 `rank`、用 `MPI_Comm_size` 拿到总数 `size`。

```
Hello from rank 0 of 4
Hello from rank 1 of 4
...
```

**提醒：不要依赖输出顺序。** 进程独立执行，谁先拿到 stdout 不保证，rank 顺序不一定是 0,1,2,3。验证脚本也只检查「每个 rank 都出现」，不检查顺序。

## 5. Demo 2 — Send / Recv

```bash
mpiexec -n 2 ./bin/send_recv
```

Rank 0 把自己局部的 `value = 42` 通过 `MPI_Send` 发给 Rank 1；Rank 1 用 `MPI_Recv` 接收并打印：

```
Rank 1 received value 42 from rank 0
```

`src/send_recv.c` 里把 `MPI_Send` / `MPI_Recv` 的每个参数单独写在一行——课堂目的就是直接看原始 API：**buffer / count / datatype / source / destination / tag / communicator**。

## 6. Exercise（动手做）

先自己想，再对照 `src/ping_pong.c`。

**Exercise 1：改进程数**
```bash
mpiexec -n 1 ./bin/hello_mpi
mpiexec -n 2 ./bin/hello_mpi
mpiexec -n 4 ./bin/hello_mpi
```
观察 `rank` / `size` / 输出顺序。

**Exercise 2：改数据**
把 `send_recv.c` 里的 `value = 42` 改成别的数字，重新 `make` 再跑，确认 Rank 1 收到新值。

**Exercise 3：让消息往返**
让 Rank 1 收到后**修改 value 再发回 Rank 0**（例如 `value += 1`），Rank 0 打印最终结果。参考实现 `src/ping_pong.c`：

```
Rank 0 sent 42 to rank 1
Rank 1 received 42
Rank 1 sent back 43
Rank 0 received 43
```

两个进程各有一份自己的 `value`（**没有共享内存**）——这正是"进程"与"线程"的区别之一。

## 7. Verification

```bash
./scripts/verify.sh
```

自动检查：build、hello（n=1/2/4）、send_recv（42）、ping_pong（43）、进程数不足时的友好提示。全部通过显示 `ALL TESTS PASSED`。

## 8. Next Tutorial

Tutorial 2 will introduce collective communication, including Reduce and AllReduce.
