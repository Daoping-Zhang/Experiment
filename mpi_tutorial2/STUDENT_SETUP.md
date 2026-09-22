# 学生课前准备说明 — Tutorial 2 课堂交互（MiniMPI）

> English version: [STUDENT_SETUP.en.md](./STUDENT_SETUP.en.md)

## 0. 一句话

一台 Mac 或 Windows 电脑 + **Python（≥3.8）** 即可；**不需要安装 MPI**，也不需要
`pip install` 任何包（MiniMPI 是纯 Python 标准库）。

## 1. 下载代码（二选一）

**方式 A：有 git（推荐，便于更新）**

```bash
git clone https://github.com/Daoping-Zhang/Experiment.git
cd Experiment/mpi_tutorial2/classroom_minimpi
```

**方式 B：没有 git**

打开 <https://github.com/Daoping-Zhang/Experiment> → **Code** → **Download ZIP**
→ 解压 → 进入 `Experiment/mpi_tutorial2/classroom_minimpi`。

> 更新（重要，见第 5 节）：在 `Experiment` 目录执行 `git pull`，或重新下载 ZIP 覆盖。

## 2. 检查环境

```bash
python3 --version            # macOS
python  --version            # Windows（用 python，不要用 python3）
```

然后在 `classroom_minimpi` 目录运行环境自检：

```bash
python3 scripts/check_env.py   # macOS
python  scripts/check_env.py   # Windows
```

各项通过即可。若报错，把输出发给老师。

> **Windows 提醒**：如果敲 `python3` 后没有任何输出、窗口一闪就结束，那是
> Microsoft Store 的 `python3` 占位程序；请改用 `python` 或 `py -3`。

## 3. 课堂怎么运行

### 老师端

1. 查本机局域网 IP：macOS `ipconfig getifaddr en0`；Windows `ipconfig`（看 IPv4）。
2. 启动 teacher（把 IP 换成上面的局域网 IP）：

```bash
python3 teacher.py --size 4 --host 0.0.0.0 --port 9000 --advertise 192.168.1.23
```

3. 看到 `Version: 2.2.0 (protocol 2)` 与 `Waiting for ranks (n joined, capacity 4)...`；
   人来得差不多（≥2 人且不再有人加入）老师就可以开始，
   `MPI World Ready: n ranks (n-1 student workers)` 就是**现在实际参加的人数**。
   `--size 4` 是容量上限：迟到的同学之后仍可加入（会打印 `[JOIN] ... -> Rank r`），
   中途退出的同学老师端会打印 `[LEAVE]` 并把 rank 重新排好（`[ROSTER] rank compaction`），
   课堂不会因此卡住。

### 学生端（最简：双击启动脚本）

```text
macOS   : 双击 scripts/start_worker_mac.command   （首次需右键→打开，或 chmod +x）
Windows : 双击 scripts/start_worker_windows.bat
```

提示 `Teacher IP:Port [192.168.1.100:9000]:` 时输入老师公布的 IP（只填 IP 会自动补
`:9000`）。成功会看到：

```text
MiniMPI Worker version 2.2.0 (protocol 2)
MiniMPI Worker
Rank: 2 / 4

Waiting for Rank 0...
```

手动等价命令：

```bash
python3 worker.py --server 192.168.1.23:9000    # macOS
python worker.py --server 192.168.1.23:9000     # Windows
```

## 4. 课堂上你会看到什么

```text
Input one integer:        ← 每个 RUN 输入一个整数
        ↓
Entering MPI Barrier...   ← 等全班到齐
        ↓
Round 视图（Send / Receive / SUM / LOCAL TIMELINE ...）
        ↓
Collective Complete → Result
        ↓
等待老师下一次 RUN
```

Performance Benchmark 时**只输入一次**，之后自动跑完；结果表会显示在**每个 rank**
的终端上。

### 我的 rank 是什么？为什么会变？

- **rank 是你这一次 collective 里的位置**（0..N-1），不是你的学号；你的"身份"是你这台
  机器（老师名单里显示 `ip:port`，那个不会变）。
- 有人加入 / 退出 / 被踢时，课堂会**重建一个 communicator**——相当于真实 MPI 里的
  `MPI_Comm_split`：只为在场的人建一个，rank 重新排成 0..N-1。所以你的号可能变化，
  终端会明确告诉你：
  ```text
  [ROSTER] world size is now 3, you are Rank 2  (you were Rank 3)
  [ROSTER] a student left and the ranks were renumbered
  ```
- **为什么必须这样**：MPI 的 collective **必须全体参与**，"人不齐"不是少几个人跑，而是
  永远等下去；真实 MPI 里进程死了通常是整个作业失败。课堂不能因为一个人合盖就停课，
  所以我们选择"重建 communicator 继续"——这是课堂版唯一有意偏离标准 MPI 的地方。
- 每次 RUN 开始前老师会打印 `Running...  (World Size n)`：那个 n 就是这一次的参与人数，
  和你终端上的 `Rank: r / n` 一致。

### 掉线了怎么办？（终端会打印什么）

| 你会看到 | 含义 | 你要做什么 |
|---|---|---|
| `[ROSTER] ... (you were Rank 3)` | 有人加入/退出，你的号变了 | **什么都不用做**，继续等下一次 RUN |
| `[ABORT] this RUN ended early: ...` | 这一轮作废（有人掉线 / 老师中止） | **什么都不用做**，等老师下一次 RUN |
| `[KICKED] the teacher removed this rank` | 老师把你移出课堂（例如你的机器卡住了） | 重新双击启动脚本即可再加入（作为新同学，rank 可能不同） |
| `[CONTROL LOST] ...` | 你和老师的连接断了（网络抖动 / 老师重启） | 重新双击启动脚本 |
| `[JOIN REFUSED] ... run is in progress` | 老师正在跑一轮，你暂时进不来 | **什么都不用做**：worker 每 3 秒自动重试，RUN 一结束就自动加入 |
| `[JOIN REFUSED] ... cannot reach the teacher` | 老师还没启动，或 IP/端口填错 | 核对老师公布的 `IP:端口` 后重新双击启动脚本 |
| 一直停在 `Waiting for Rank 0...` | 同上 | 同上 |

> 注意：**不要因为自己 rank 变了就重启 worker**。rank 变化是正常的课堂行为；
> 只有上面标了"重新双击启动脚本"的几种情况才需要你动手。

## 5. 版本一致性（务必注意）

teacher 与 worker 必须同版本，否则 join 会被拒绝：

```text
[VERSION MISMATCH] version mismatch: worker=0.0.0 (protocol 1),
teacher=2.2.0 (protocol 2) — please UPDATE the student copy ...

Please update this student copy (git pull / re-download) and start the worker again.
```

处理：`git pull`（或重新下载 ZIP）后，重新双击启动脚本。

## 6. 常见问题

| 现象 | 处理 |
|---|---|
| 敲 `python3` 无输出直接结束 | Windows：改用 `python` 或 `py -3` |
| worker 停在 `Waiting for Rank 0...` | 老师未启动，或 IP/端口不对 |
| Connectivity 有 FAIL | 不在同一局域网；老师需用局域网 IP（不要 127.0.0.1）；允许防火墙入站 |
| 中文/符号乱码 | 启动脚本已设 `PYTHONUTF8=1`；手动运行可先 `chcp 65001` |
| Mac 双击没反应 | 右键→打开，或 `chmod +x scripts/start_worker_mac.command` |

## 7. 课前 Checklist

- [ ] 已下载/克隆代码并进入 `classroom_minimpi`
- [ ] Python ≥3.8，`scripts/check_env.py` 通过
- [ ] 能打开一键启动脚本（Mac 已 `chmod +x`）
- [ ] 知道老师公布的 IP
- [ ] 知道更新方式（`git pull` / 重新下载）
- [ ] 无需安装 MPI、mpi4py 或任何 pip 包

## 8. 后续内容（本次课前不用准备）

Real MPI Demo 与 Assignment 1（Parallel GEMM）使用 **C + MPI**（`mpicc` / `mpirun`），
到时会另发服务器环境说明；Python 只作为教师工具语言。
