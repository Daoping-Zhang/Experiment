# Student Pre-Class Setup — Tutorial 2 Classroom (MiniMPI)

> 中文版: [STUDENT_SETUP.md](./STUDENT_SETUP.md)

## 0. In one sentence

A Mac or Windows laptop with **Python (≥ 3.8)** is all you need. You do **not**
need MPI, and you do **not** need to `pip install` anything (MiniMPI is pure
Python standard library).

## 1. Get the code (choose one)

**Option A — with git (recommended; easy to update)**

```bash
git clone https://github.com/Daoping-Zhang/Experiment.git
cd Experiment/mpi_tutorial2/classroom_minimpi
```

**Option B — without git**

Open <https://github.com/Daoping-Zhang/Experiment> → **Code** → **Download ZIP**
→ unzip → enter `Experiment/mpi_tutorial2/classroom_minimpi`.

> Updating (important, see §5): run `git pull` inside `Experiment`, or
> re-download the ZIP and overwrite.

## 2. Check your environment

```bash
python3 --version            # macOS
python  --version            # Windows (use python, not python3)
```

Then, inside `classroom_minimpi`:

```bash
python3 scripts/check_env.py   # macOS
python  scripts/check_env.py   # Windows
```

Everything should pass. If not, send the output to the instructor.

> **Windows note**: if `python3` prints nothing and the window just closes
> instantly, that is the Microsoft Store `python3` placeholder — use `python`
> or `py -3` instead.

## 3. How to run it in class

### Instructor side

1. Find your LAN IP: macOS `ipconfig getifaddr en0`; Windows `ipconfig` (IPv4).
2. Start the teacher (replace the IP with that LAN address):

```bash
python3 teacher.py --size 4 --host 0.0.0.0 --port 9000 --advertise 192.168.1.23
```

3. You should see `Version: 2.2.0 (protocol 2)` and
   `Waiting for ranks (n joined, capacity 4)...`; the teacher can start once
   a couple of students are in and nobody is still joining, and
   `MPI World Ready: n ranks (n-1 student workers)` names the **actual**
   number of participants.
   `--size 4` is a capacity, not a quota: a late student can still join
   (the teacher logs `[JOIN] ... -> Rank r`), and if somebody leaves the
   teacher logs `[LEAVE]` and renumbers the ranks (`[ROSTER] rank
   compaction`) — the class never wedges because of it.

### Student side (simplest: double-click the launcher)

```text
macOS   : double-click scripts/start_worker_mac.command
          (first time: right-click -> Open, or chmod +x)
Windows : double-click scripts/start_worker_windows.bat
```

At the prompt `Teacher IP:Port [192.168.1.100:9000]:` type the teacher's IP
(an IP without a port gets `:9000` appended automatically). On success:

```text
MiniMPI Worker version 2.2.0 (protocol 2)
MiniMPI Worker
Rank: 2 / 4

Waiting for Rank 0...
```

Manual equivalent:

```bash
python3 worker.py --server 192.168.1.23:9000    # macOS
python worker.py --server 192.168.1.23:9000     # Windows
```

## 4. What you will see in class

```text
Input one integer:        <- one integer per RUN
        ↓
Entering MPI Barrier...   <- wait until the whole class arrives
        ↓
Round view (Send / Receive / SUM / LOCAL TIMELINE ...)
        ↓
Collective Complete → Result
        ↓
wait for the teacher's next RUN
```

In the Performance Benchmark you type your integer **once**; all cases then
run automatically, and the results table is shown on **every rank**.

### What is my rank, and why does it change?

- **A rank is your position in THIS collective** (0..N-1), not your student ID.
  Your identity is your machine (the teacher's roster shows its `ip:port`,
  which never changes).
- When somebody joins / leaves / is kicked, the class **creates a new
  communicator** — the equivalent of real MPI's `MPI_Comm_split`: a fresh one
  for the students who are present, with ranks renumbered 0..N-1. Your number
  may therefore change, and your terminal says so:
  ```text
  [ROSTER] world size is now 3, you are Rank 2  (you were Rank 3)
  [ROSTER] a student left and the ranks were renumbered
  ```
- **Why it has to work that way**: an MPI collective needs **every** member.
  "Some people missing" is not "run with fewer ranks" — it waits forever, and
  in real MPI a dead process normally fails the whole job. A classroom cannot
  stop because one laptop closed, so we rebuild the communicator and carry on.
  This is the one deliberate deviation from standard MPI in the classroom
  version.
- Before each RUN the teacher prints `Running...  (World Size n)`: that n is the
  number of participants this time, and it matches the `Rank: r / n` on your
  terminal.

### What if I get disconnected? (what the terminal prints)

| You see | Meaning | What you do |
|---|---|---|
| `[ROSTER] ... (you were Rank 3)` | Somebody joined/left, your number changed | **Nothing** — wait for the next RUN |
| `[ABORT] this RUN ended early: ...` | This RUN is void (someone left / teacher stopped it) | **Nothing** — wait for the next RUN |
| `[KICKED] the teacher removed this rank` | The teacher removed you (e.g. your machine froze) | Double-click the launcher again to rejoin (as a new student, possibly a new rank) |
| `[CONTROL LOST] ...` | Your link to the teacher broke (network blip / teacher restarted) | Double-click the launcher again |
| `[JOIN REFUSED] ... run is in progress` | A RUN is in flight, you cannot join yet | **Nothing** — the worker retries every 3 s and joins automatically when the RUN ends |
| `[JOIN REFUSED] ... cannot reach the teacher` | The teacher is not running, or the IP/port is wrong | Check the teacher's `IP:port` and launch again |
| Stuck at `Waiting for Rank 0...` | Same as above | Same as above |

> Do **not** restart your worker just because your rank changed — that is normal
> classroom behaviour. Only the rows marked "double-click the launcher again"
> need action from you.

## 5. Version consistency (important)

The teacher and every worker must be the same version, otherwise the join is
refused:

```text
[VERSION MISMATCH] version mismatch: worker=0.0.0 (protocol 1),
teacher=2.2.0 (protocol 2) — please UPDATE the student copy ...

Please update this student copy (git pull / re-download) and start the worker again.
```

Fix: `git pull` (or re-download the ZIP), then double-click the launcher again.

## 6. Troubleshooting

| Symptom | Fix |
|---|---|
| `python3` prints nothing and exits (Windows) | use `python` or `py -3` |
| worker stuck at `Waiting for Rank 0...` | teacher not started, or wrong IP/port |
| Connectivity shows FAIL | not on the same LAN; teacher must use the LAN IP (not 127.0.0.1); allow Python through the firewall |
| Garbled Chinese/arrow characters | launchers already set `PYTHONUTF8=1`; manually: `chcp 65001` |
| macOS double-click does nothing | right-click → Open, or `chmod +x scripts/start_worker_mac.command` |

## 7. Pre-class checklist

- [ ] Code downloaded/cloned; you are in `classroom_minimpi`
- [ ] Python ≥ 3.8; `scripts/check_env.py` passes
- [ ] The one-click launcher opens (macOS: `chmod +x` done)
- [ ] You know the teacher's IP (announced in class)
- [ ] You know how to update (`git pull` / re-download)
- [ ] No MPI, mpi4py or pip packages needed

## 8. What comes later (nothing to prepare now)

The Real MPI demo and Assignment 1 (Parallel GEMM) use **C + MPI**
(`mpicc` / `mpirun`); a separate server-environment guide will be provided.
Python is only the course-tool language.
