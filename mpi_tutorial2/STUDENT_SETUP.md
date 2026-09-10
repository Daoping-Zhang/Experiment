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

3. 看到 `Version: 2.1.0 (protocol 1)` 与 `Waiting for ranks (1/4)...`；
   学生到齐后变成 `MPI World Ready: 4 / 4 ranks`。

### 学生端（最简：双击启动脚本）

```text
macOS   : 双击 scripts/start_worker_mac.command   （首次需右键→打开，或 chmod +x）
Windows : 双击 scripts/start_worker_windows.bat
```

提示 `Teacher IP:Port [192.168.1.100:9000]:` 时输入老师公布的 IP（只填 IP 会自动补
`:9000`）。成功会看到：

```text
MiniMPI Worker version 2.1.0 (protocol 1)
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

## 5. 版本一致性（务必注意）

teacher 与 worker 必须同版本，否则 join 会被拒绝：

```text
[VERSION MISMATCH] version mismatch: worker=0.0.0 (protocol 1),
teacher=2.1.0 (protocol 1) — please UPDATE the student copy ...

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
