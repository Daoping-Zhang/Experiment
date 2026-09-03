# MPI Tutorial 1 — 课堂演示脚本（上课照此执行）

> 用途：课堂上逐步照着跑的命令清单。每条命令可直接复制粘贴。
> 节奏：**先看代码/提问 → 预测 → 运行 → 解释**。别一次跑完，跑一步讲一步。

---

## 0. 课前准备（进教室前自己先跑一遍）

```bash
cd ~/Experiment/mpi_tutorial1
git pull                 # 拿到最新代码
make clean && make       # 编译（会显示完整编译命令）
./scripts/check_env.sh   # 环境自检，全部 [PASS] 才开课

# 如果你以 root 运行（OpenMPI 默认禁止 root），先设两个环境变量：
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

# 小规格主机（物理核数 < 演示进程数）会有 "not enough slots" 限制：
# 这是 OpenMPI 把物理核数当 slot。下面这条环境变量等价于每次加
# --oversubscribe，设置后裸 mpirun -n 4 也能直接跑：
export OMPI_MCA_rmaps_base_oversubscribe=1

# 三个 export 都可写进 ~/.bashrc 免重复设置
```

> 两种运行方式的区别：
> - **脚本**（`./scripts/run_all.sh`、`verify.sh`）自动处理 root（`--allow-run-as-root`）与小主机（`--oversubscribe`）问题，不用你操心。
> - **手动敲命令**：把上面三个 export 放进 `~/.bashrc` 后，直接 `mpirun -n 4 ./bin/hello_mpi` 即可；不想设环境变量就临时加 flag：`mpirun --allow-run-as-root --oversubscribe -n 4 ./bin/hello_mpi`。

课堂开头对全班说一句：
> 一个 executable + `mpiexec -n P` → P 个独立进程。今天看两件事：怎么造多进程、两个进程怎么说话。

---

## 1. Demo 1 — Hello MPI（对应 PPT: Why MPI / Execution Model / Lifecycle）

**① 看代码（只给看核心，别展开整个文件）**
```bash
head -18 src/hello_mpi.c
```
指着讲 4 行：
```c
MPI_Init(&argc, &argv);               // 每个 MPI 程序的入口
MPI_Comm_rank(MPI_COMM_WORLD, &rank); // 我是谁（0..size-1）
MPI_Comm_size(MPI_COMM_WORLD, &size); // 一共几个
MPI_Finalize();                       // 出口
```

**② 提问**：同一个文件怎么变成 4 份同时跑？（等学生说 `-n`）

**③ 跑 1 个进程**
```bash
mpiexec -n 1 ./bin/hello_mpi
```

**④ 跑 4 个进程**
```bash
mpiexec -n 4 ./bin/hello_mpi
```

**⑤ 关键教学点——连跑两次证明顺序不保证**
```bash
mpiexec -n 4 ./bin/hello_mpi
mpiexec -n 4 ./bin/hello_mpi
```
> 台词："输出顺序不保证，谁先抢到 stdout 不一定。写代码别假设 rank 0 先打印。"

板书：
```text
mpiexec -n 4 hello_mpi
   ↓
rank0  rank1  rank2  rank3    （4 个独立进程，各有各的内存）
```

---

## 2. Demo 2 — MPI_Send / MPI_Recv（对应 PPT: How Processes Communicate）

**① 提问（最重要）**："刚才 4 个进程能看见对方的变量吗？" → 不能，没有共享内存。

**② 看代码**（只看收发两段）
```bash
sed -n '36,62p' src/send_recv.c
```
带全班读参数：
```c
MPI_Send(&value, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
         ↑buffer ↑count ↑datatype ↑dest ↑tag ↑communicator
```

**③ 先问预测**："Rank 1 会打印出多少？" 再跑：
```bash
mpiexec -n 2 ./bin/send_recv
```
预期：`Rank 1 received value 42 from rank 0`

**④ 进程数不足的兜底**（问：只开 1 个进程会怎样？）
```bash
mpiexec -n 1 ./bin/send_recv
```
预期：`This example requires at least 2 MPI processes.`（不 crash、不挂死）

---

## 3. Exercise（学生动手改，约 15 分钟）

### Exercise 1 — 换进程数
```bash
mpiexec -n 1 ./bin/hello_mpi
mpiexec -n 2 ./bin/hello_mpi
mpiexec -n 4 ./bin/hello_mpi
```
观察 rank / size / 顺序。

### Exercise 2 — 改数据
把 `src/send_recv.c` 里的 `value = 42` 改成别的数：
```bash
sed -i 's/value = 42;/value = 100;/' src/send_recv.c   # 例子：改成 100
make && mpiexec -n 2 ./bin/send_recv                   # Rank 1 应收到 100
git checkout src/send_recv.c                           # 跑完恢复原样
```

### Exercise 3 — 让消息往返（先让学生自己写，再给答案）
参考答案：
```bash
sed -n '1,60p' src/ping_pong.c
```
跑通：
```bash
make && mpiexec -n 2 ./bin/ping_pong
```
预期四行：`sent 42` / `received 42` / `sent back 43` / `received 43`
> 台词："`value += 1` 只改了 Rank 1 的本地副本——没有共享内存；通信双向，全靠消息一来一回。"

---

## 4. 收尾

```bash
./scripts/verify.sh     # 应显示 ALL TESTS PASSED
```

最终板书：
```text
① 一个 executable → mpiexec -n P → P 个进程（rank / size / MPI_COMM_WORLD）
② 进程无共享内存 → MPI_Send / MPI_Recv 显式传消息
③ Tutorial 2：把各 worker 的 g0+g1+g2+g3 合起来 → collective（Reduce / AllReduce）
```

---

## 附：上课常用速查

| 想做什么 | 命令 |
|---|---|
| 重新编译 | `make`（无改动时显示 Nothing to be done；强制重编先 `make clean`） |
| 只看命令不执行 | `make -n` |
| 单跑某个 demo | `mpiexec -n 4 ./bin/hello_mpi` 等 |
| 恢复被我改过的源码 | `git checkout src/send_recv.c` |
| 重新验收 | `./scripts/verify.sh` |
| 换 MPI 启动器 | 脚本自动优先 `mpiexec`、回退 `mpirun`；OpenMPI+root 自动加 `--allow-run-as-root` |

> 边界提醒：Tutorial 1 全程只用 `MPI_Init / Comm_rank / Comm_size / Send / Recv / Finalize`；
> 不要演示 collective / Barrier / 非阻塞（留给 Tutorial 2），不要引入 GPU / PyTorch。
