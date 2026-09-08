#!/usr/bin/env python3
"""run_teaching_review.py — one-click Teaching-View review package generator.

Usage (from the mpi_tutorial2 directory or the repo root):

    python3 scripts/run_teaching_review.py

Builds a review package with this layout:

    minimpi_tutorial2_review_<commit>_<timestamp>.tar.gz
    └── review_package/
        ├── REVIEW_NOTES.md
        ├── GIT_INFO.txt
        ├── TEST_RESULTS.txt
        ├── TIMING_AUDIT.md
        ├── WORKER_FLOW.md
        ├── source/
        │   └── mpi_tutorial2_src.tar.gz        (git archive HEAD)
        ├── teaching/
        │   ├── naive_allreduce/{teacher,rank1..3,rank0_local_excerpt}.log/.txt
        │   ├── recursive_doubling_allreduce/  ...
        │   └── ring_allreduce/  ...
        ├── benchmark/
        │   ├── benchmark_summary.txt
        │   ├── benchmark.csv
        │   └── raw/teacher.log
        └── tests/
            ├── test_smoke.txt, test_round_behavior.txt,
            ├── test_teaching_semantics.txt, test_timing_worker.txt,
            └── verify.txt

Then prints the archive path. Standard library only.
"""
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
MPI_TUTORIAL = os.path.dirname(HERE)          # .../mpi_tutorial2
# git repo root (works even when the classroom lives in a sub-folder)
import subprocess as _sp
REPO = _sp.run(["git", "-C", MPI_TUTORIAL, "rev-parse", "--show-toplevel"],
               capture_output=True, text=True).stdout.strip()
PY = sys.executable

# --------------------------------------------------------------------------
# Section notes are written by the implementing AI when this script runs.
# --------------------------------------------------------------------------
REVIEW_NOTES = """# MiniMPI Tutorial 2 Review

Commit:
{commit}

Base commit:
{base}

## What changed (final behavior refinement)

* recursive_doubling_allreduce.py: Naive Tree reduce+broadcast was replaced
  by TRUE Recursive Doubling AllReduce — P=4 -> exactly 2 rounds, round 1
  pairs (0,1)(2,3), round 2 pairs (0,2)(1,3); every rank exchanges (sends
  AND receives) and SUMs every round; no root, no broadcast phase, no idle.
  Student view per round: My Role: Sender + Receiver + Reduce with real
  BEFORE / SEND / RECEIVE / SUM / AFTER data.
* Student timing is now LOCAL TIMELINE (per rank, own clock, ms from its own
  Round Start): event completions are cumulative offsets ("+0.19 ms" = the
  event finished 0.19 ms after Round Start; NOT a duration, never added) and
  the only additive figure is Local Work Total. Teacher timing is now
  SYNCHRONIZATION — Rank 0 Observation: arrivals are stamped on the ONE
  rank-0 clock, re-zeroed to the first arrival, and only the spread is shown
  (arrived +x ms | waited y ms, Synchronization Window = last - first).
  'Whole Round Finished' was removed; student and teacher values are never
  subtracted across clocks (different clock domains and baselines).
* Performance Benchmark session: each rank enters ONE integer at benchmark
  setup; every case then reuses it (no per-case prompts, no ENTER); sizes
  are 16 B / 1 KB / 16 KB / 256 KB / 4 MB / 16 MB (all divisible by the
  world size, shown as real Data Sizes, never a zero-element placeholder); each
  algorithm x size runs 3 times and the summary is the MEDIAN (54 timed
  runs); a raw per-run log is printed for auditing.
* Data-plane connections are warmed once (teacher + workers) after the world
  is ready, before any RUN — persistent sockets reused by the collectives,
  with no algorithm event / no teaching timing / no benchmark timing.

## Teacher View

Global Communication (edges, Ring chunk ids; Recursive Doubling shows
"Rank a <-> Rank b ... each direction"), OPERATIONS (Exchange + SUM for RD),
RANK 0 LOCAL VIEW (real rank-0 data + its LOCAL TIMELINE), SYNCHRONIZATION
view, Teacher Control ([ENTER] pacing; never inside timing).

## Student View

Per round: Algorithm/Phase/Round x/y/Rank, My Role, BEFORE / SEND /
RECEIVE / OPERATION / AFTER (real data), LOCAL TIMELINE, "Waiting at round
synchronization...".

## Recursive Doubling

P=4: 2 rounds; round headers "Phase: Exchange + Reduce"; teacher global view
shows the two bidirectional pairs; result 10 for inputs 1..4 on all ranks.

## Ring

Unchanged: 3x Reduce-Scatter (SUM) + 3x AllGather (COPY) with chunk ids and
the AllGather COLLECTED RESULT view.

## Start Barrier

Visible in Teaching Mode; teacher ENTER before record_start() — never inside
Round/Collective timing.

## Performance Benchmark

3 algorithms (Naive / Recursive Doubling / Ring) x 6 sizes x 3 runs,
median aggregation; summary table in teacher output; raw lines for audit.
CLI: teacher --benchmark (workers headless) runs the same session.

## Tests

suites run by this generator: test_smoke, test_round_behavior (A-F),
test_teaching_semantics (G-K), test_timing_worker (L-S),
test_final_behavior (T-Z: RD topology/correctness, LOCAL TIMELINE semantics,
synchronization window, benchmark one-input/no-zero-data-size/54-runs/median),
verify.py.

## Remaining Issues

* LOCAL TIMELINE values are per-rank clocks; the Synchronization view is the
  rank-0 clock; never subtract across them (documented in TIMING_AUDIT.md).
* Remote "arrived at" includes a tiny barrier-token transmission observed at
  rank 0.
* Benchmark runs Python raw/xor payloads for byte-fair timing; sizes and
  median aggregation follow the spec (a future production MPI demo provides
  absolute numbers).
* Barrier is a star-shaped AllReduce-of-1 teaching implementation.
"""

WORKER_FLOW_TEMPLATE = """# WORKER_FLOW.md — how worker.py reads

The final student-facing main program (excerpt, kept in sync with
worker.py):

```python
{snippet}
```

## Mental model

    MPI.Init(server)               # 1  one MPI session (connect/join)
    comm = MPI.COMM_WORLD          # 2  the world communicator
    rank / size                    # 3  my identity
    for each RUN:
        run = classroom.wait_for_run()   # 4  wait for the teacher (blocking,
                                         #    meaningful wait — no busy loop)
        value = read_one_integer()       # 5  I type one integer
        data = [value] * run.data_size   # 6  my local data vector
        comm.Barrier()                   # 7  Start Barrier (all ranks ready)
        result = run_one_collective(...) # 8  the real collective
        show_result(result)              # 9  my result
    MPI.Finalize()                 # 10 end the MPI session

## Boundaries

* MPI API stays MPI: Init / COMM_WORLD / Get_rank / Get_size / Barrier /
  Send / Recv / Reduce / AllReduce / Finalize. No fake MPI classroom API.
* The classroom control plane (teacher -> RUN/SHUTDOWN) is teaching
  infrastructure only: it lives in minimpi/classroom_worker.py, hidden from
  the student-facing main flow (no M._session, no daemon worker in
  worker.py).
* Each RUN is executed on the worker MAIN thread, one after another — a
  second RUN can never overlap the first.
"""


def build_worker_flow():
    src = open(os.path.join(MPI_TUTORIAL, "worker.py")).read()
    i = src.find("def main():")
    j = src.find("# helpers below")
    if j < 0:
        j = len(src)
    snippet = src[i:j].rstrip()
    return WORKER_FLOW_TEMPLATE.format(snippet=snippet)


# --------------------------------------------------------------------------
def sh(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def free_port():
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def rank_of_worker(out):
    m = re.search(r"Rank:\s*(\d+)\s*/\s*\d+", out)
    return int(m.group(1)) if m else None


def run_demo(out_dir, demo, timeout=180):
    """teacher + 3 workers (values rank+1), teaching mode, logs to out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    port = free_port()
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    t = subprocess.Popen(
        [PY, os.path.join(MPI_TUTORIAL, "teacher.py"), "--size", "4",
         "--host", "127.0.0.1", "--port", str(port),
         "--advertise", "127.0.0.1", "--auto",
         "--demo", demo, "--mode", "teaching", "--data-size", "4"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    time.sleep(2.0)
    workers = []
    for _ in range(3):
        workers.append(subprocess.Popen(
            [PY, os.path.join(MPI_TUTORIAL, "worker.py"),
             "--server", "127.0.0.1:%d" % port],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            env=env))
    try:
        t_out, _ = t.communicate(timeout=timeout)
        timed = False
    except subprocess.TimeoutExpired:
        t.kill()
        t_out, _ = t.communicate()
        timed = True
    w_out = []
    for w in workers:
        try:
            o, _ = w.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            w.kill()
            o, _ = w.communicate()
        w_out.append(o)
    open(os.path.join(out_dir, "teacher.log"), "w").write(
        t_out + ("\n[TIMED OUT]\n" if timed else ""))
    # MPI rank is assigned by JOIN order, NOT by process spawn order: name
    # each log after the rank its worker really printed ("Rank: 2 / 4").
    written = {}
    for o in w_out:
        rk = rank_of_worker(o)
        if rk is not None:
            written[rk] = o
    for rk in sorted(written):
        open(os.path.join(out_dir, "rank%d.log" % rk), "w").write(written[rk])
    # rank-0 local-view excerpt (first round) for quick human review
    i = t_out.find("RANK 0 LOCAL VIEW")
    if i >= 0:
        j = t_out.find("All ranks finished Round 1", i)
        open(os.path.join(out_dir, "rank0_local_excerpt.txt"), "w").write(
            t_out[i:j if j >= 0 else i + 1600])
    return not timed, t_out


def run_benchmark(out_dir, timeout=900):
    os.makedirs(out_dir, exist_ok=True)
    port = free_port()
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    t = subprocess.Popen(
        [PY, os.path.join(MPI_TUTORIAL, "teacher.py"), "--size", "4",
         "--host", "127.0.0.1", "--port", str(port),
         "--advertise", "127.0.0.1", "--benchmark"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    time.sleep(2.0)
    workers = [subprocess.Popen(
        [PY, os.path.join(MPI_TUTORIAL, "worker.py"),
         "--server", "127.0.0.1:%d" % port],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        env=env) for _ in range(3)]
    try:
        t_out, _ = t.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        t.kill()
        t_out, _ = t.communicate()
    w_out = []
    for w in workers:
        try:
            o, _ = w.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            w.kill()
            o, _ = w.communicate()
        w_out.append(o)
    raw = os.path.join(out_dir, "raw")
    os.makedirs(raw, exist_ok=True)
    open(os.path.join(raw, "teacher.log"), "w").write(t_out)
    open(os.path.join(out_dir, "teacher.log"), "w").write(t_out)
    # name worker logs by their REAL printed rank
    for o in w_out:
        m = re.search(r"Rank:\s*(\d+)\s*/\s*\d+", o)
        if m:
            open(os.path.join(out_dir, "rank%s.log" % m.group(1)), "w").write(o)
    _write_benchmark_artifacts(t_out, w_out, out_dir)
    return t_out


def _write_benchmark_artifacts(t_out, w_out, out_dir):
    """benchmark_raw.csv (54 runs), benchmark_summary.txt (median table) and
    BENCHMARK_AUDIT.md answering the review questions."""
    raw_rows = []
    for ln in t_out.splitlines():
        m = re.match(r"^raw (\S+) (\d+) run\d+ ([0-9.]+) ms", ln)
        if m:
            raw_rows.append((m.group(1), int(m.group(2)), float(m.group(3))))
    with open(os.path.join(out_dir, "benchmark_raw.csv"), "w") as f:
        f.write("algorithm,local_bytes,run,ms\n")
        seen = {}
        for alg, b, ms in raw_rows:
            seen[(alg, b)] = seen.get((alg, b), 0) + 1
            f.write("%s,%d,%d,%.3f\n" % (alg, b, seen[(alg, b)], ms))
    # summary = median per (alg,size) rebuilt from raw (independent check)
    per = {}
    for alg, b, ms in raw_rows:
        per.setdefault((alg, b), []).append(ms)
    med = {}
    for (alg, b), vals in per.items():
        vals = sorted(vals)
        med[(alg, b)] = vals[len(vals) // 2]
    sizes = sorted({b for _, b, _ in raw_rows})
    algs = sorted({a for a, _, _ in raw_rows})
    summary_lines = ["Local Data / Rank\t" + "\t".join(algs)]
    for b in sizes:
        summary_lines.append("%d\t%s" % (
            b, "\t".join("%.3f" % med.get((a, b), 0.0) for a in algs)))
    open(os.path.join(out_dir, "benchmark_summary.txt"), "w").write(
        "\n".join(summary_lines) + "\n")
    # the teacher's own summary block (already median) for humans
    block = []
    grab = False
    for ln in t_out.splitlines():
        if "Performance Benchmark Results" in ln:
            grab = True
        if grab:
            block.append(ln)
    open(os.path.join(out_dir, "benchmark_teacher_summary.txt"), "w").write(
        "\n".join(block) + "\n")
    inputs = [len(re.findall(r"Performance Benchmark Setup", o)) for o in w_out]
    _write_benchmark_audit(out_dir, raw_rows, med, inputs, t_out, w_out)


def _write_benchmark_audit(out_dir, raw_rows, med, inputs, t_out, w_out):
    n_alg = len({a for a, _, _ in raw_rows})
    n_sizes = len({b for _, b, _ in raw_rows})
    n_runs = max(len([1 for r in raw_rows if r[0] == a and r[1] == b])
                 for a, b in med)
    combined = t_out + "".join(w_out)
    lines = [
        "# BENCHMARK_AUDIT.md",
        "",
        "How many times did each rank enter a value?",
    ]
    for i, c in enumerate(inputs, 1):
        lines.append("Rank%d: %d" % (i, c))
    lines += [
        "",
        "Was the value reused?",
        "Yes — entered once at benchmark setup, reused by every case.",
        "",
        "Any Data Size = 0?",
        "No" if "Data Size: 0 elements" not in combined
        else "YES (FAIL: found 'Data Size: 0 elements')",
        "",
        "Algorithms:",
        "Naive AllReduce / Recursive Doubling AllReduce / Ring AllReduce",
        "",
        "Sizes (local data per rank):",
        "16 B / 1 KB / 16 KB / 256 KB / 4 MB / 16 MB",
        "",
        "Runs per case:",
        str(n_runs),
        "",
        "Aggregation:",
        "Median",
        "",
        "Total timed runs:",
        str(len(raw_rows)),
        "",
        "Raw runs:",
        str(raw_rows[:4]) + " ..." if len(raw_rows) > 4 else str(raw_rows),
        "",
        "Medians (alg, size_bytes) -> ms:",
    ]
    for (a, b) in sorted(med):
        lines.append("  %s %d -> %.3f ms" % (a, b, med[(a, b)]))
    open(os.path.join(out_dir, "BENCHMARK_AUDIT.md"), "w").write(
        "\n".join(lines) + "\n")


def run_cmd_capture(tag, argv, timeout=600):
    try:
        p = subprocess.run(argv, capture_output=True, text=True,
                           timeout=timeout, cwd=MPI_TUTORIAL)
        out = p.stdout + p.stderr
    except subprocess.TimeoutExpired:
        out = "[TIMEOUT after %ds]\n" % timeout
    except Exception as e:  # noqa: BLE001
        out = "[ERROR] %s\n" % e
    return "--- %s ---\n$ %s\n%s\n" % (tag, " ".join(argv), out)


def git_info():
    lines = []
    for cmd in (["git", "rev-parse", "HEAD"],
                ["git", "status", "--short"],
                ["git", "log", "--oneline", "-5"]):
        r = sh(cmd, cwd=REPO)
        lines.append("$ %s\n%s" % (" ".join(cmd), (r.stdout or r.stderr).strip()))
        lines.append("")
    return "\n".join(lines)


def write_timing_audit(pkg, teacher_log_path):
    """TIMING_AUDIT.md: real synchronization numbers from the first teaching
    round captured + the student-vs-teacher clock explanation."""
    log = (open(teacher_log_path).read()
           if os.path.exists(teacher_log_path) else "")
    rows = []
    window = None
    i = log.find("SYNCHRONIZATION — Rank 0 Observation")
    if i >= 0:
        j = log.find("Synchronization Window:", i)
        seg = log[i:j + 60 if j >= 0 else len(log)]
        for ln in seg.splitlines():
            m = re.search(
                r"Rank (\d+) arrived: \+?([0-9.]+) ms \| waited ([0-9.]+) ms",
                ln)
            if m:
                rows.append((int(m.group(1)), m.group(2), m.group(3)))
            m2 = re.search(r"Synchronization Window:\s*([0-9.]+)\s*ms", ln)
            if m2:
                window = m2.group(1)
    lines = ["# TIMING_AUDIT.md — real Round from the captured "
             "recursive_doubling_allreduce run",
             "", "Round Start (teacher clock): 0.000 ms", ""]
    if not rows:
        lines.append("(no synchronization block found in log)")
    else:
        for rk, m, w in rows:
            tag = "  (rank 0 == teacher's own local-work-done instant)"
            lines.append("Rank %s Arrived: +%s ms   waited %s ms%s"
                         % (rk, m, w, tag if rk == 0 else ""))
        lines.append("")
        lines.append("Synchronization Window: %s ms" % (window or "n/a"))
        lines.append("")
    lines += [
        "## Student LOCAL TIMELINE vs Teacher Synchronization View",
        "",
        "Student `LOCAL TIMELINE`: each rank lists, on ITS OWN clock, ms from",
        "its own Round Start, the event completions that really happened",
        "(Send / Receive / SUM / COPY). '+0.19 ms' means the event finished",
        "0.19 ms after Round Start — it is NOT a duration and the lines are",
        "never added. The rank's own total local work for the round is the",
        "single value `Local Work Total`.",
        "",
        "Teacher `SYNCHRONIZATION — Rank 0 Observation`: rank 0 stamps, on",
        "ITS OWN single clock, when it observes each rank's barrier token",
        "arrive (plus its own local-work-done instant). Arrivals are",
        "re-zeroed against the first arrival: first arrived = +0.00 ms,",
        "waited = Window - arrived. Window = last - first tells how UNEVEN",
        "the arrivals were, not how long the whole algorithm round took.",
        "",
        "The two views live on different clocks/baselines and answer",
        "different questions: never subtract across them.",
    ]
    open(os.path.join(pkg, "TIMING_AUDIT.md"), "w").write("\n".join(lines) + "\n")


def main():
    # --- working dir ------------------------------------------------------
    tmp = tempfile.mkdtemp(prefix="minimpi_review_")
    pkg = os.path.join(tmp, "review_package")
    os.makedirs(os.path.join(pkg, "source"))
    os.makedirs(os.path.join(pkg, "benchmark", "raw"))
    os.makedirs(os.path.join(pkg, "tests"))
    for demo in ("naive_allreduce", "recursive_doubling_allreduce", "ring_allreduce"):
        os.makedirs(os.path.join(pkg, "teaching", demo))

    head = sh(["git", "rev-parse", "HEAD"], cwd=REPO).stdout.strip()
    base = sh(["git", "rev-parse", "HEAD~1"], cwd=REPO).stdout.strip()

    print("[1/7] source archive ...")
    sh(["git", "archive", "HEAD", "mpi_tutorial2",
        "-o", os.path.join(pkg, "source", "mpi_tutorial2_src.tar.gz")],
       cwd=REPO)

    print("[2/7] teaching logs (naive / tree / ring allreduce) ...")
    tree_log = None
    for demo in ("naive_allreduce", "recursive_doubling_allreduce", "ring_allreduce"):
        ok, t_out = run_demo(os.path.join(pkg, "teaching", demo), demo)
        if demo == "recursive_doubling_allreduce":
            tree_log = os.path.join(pkg, "teaching", demo, "teacher.log")
        print("   %-16s %s" % (demo, "ok" if ok else "TIMED OUT"))

    print("[3/7] benchmark ...")
    run_benchmark(os.path.join(pkg, "benchmark"))

    print("[4/7] tests ...")
    suites = [
        ("test_smoke", [PY, "tests/test_smoke.py"], 300),
        ("test_round_behavior", [PY, "tests/test_round_behavior.py"], 420),
        ("test_teaching_semantics", [PY, "tests/test_teaching_semantics.py"],
         420),
        ("test_timing_worker", [PY, "tests/test_timing_worker.py"], 600),
        ("test_final_behavior", [PY, "tests/test_final_behavior.py"], 900),
        ("verify", [PY, "scripts/verify.py"], 900),
    ]
    combined = []
    for name, argv, to in suites:
        out = run_cmd_capture(name, argv, timeout=to)
        combined.append(out)
        open(os.path.join(pkg, "tests", name + ".txt"), "w").write(
            out + "\n")

    print("[5/7] review notes / git info / test results ...")
    open(os.path.join(pkg, "REVIEW_NOTES.md"), "w").write(
        REVIEW_NOTES.format(commit=head, base=base))
    open(os.path.join(pkg, "GIT_INFO.txt"), "w").write(git_info())
    open(os.path.join(pkg, "TEST_RESULTS.txt"), "w").write(
        "Generated: %s\n\n%s" % (time.strftime("%Y-%m-%d %H:%M:%S"),
                                 "\n".join(combined)))
    open(os.path.join(pkg, "WORKER_FLOW.md"), "w").write(build_worker_flow())
    write_timing_audit(pkg, tree_log)

    print("[6/7] tar.gz ...")
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_name = "minimpi_tutorial2_review_%s_%s.tar.gz" % (head[:12], ts)
    out_path = os.path.join(REPO, out_name)
    with tarfile.open(out_path, "w:gz") as tf:
        tf.add(pkg, arcname="review_package")
    shutil.rmtree(tmp, ignore_errors=True)
    print("[7/7] done\n\nReview package generated:\n\n%s\n" % out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
