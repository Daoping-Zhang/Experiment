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
        │   ├── tree_allreduce/  ...
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
REPO = os.path.dirname(MPI_TUTORIAL)          # git repo root
PY = sys.executable

# --------------------------------------------------------------------------
# Section notes are written by the implementing AI when this script runs.
# --------------------------------------------------------------------------
REVIEW_NOTES = """# MiniMPI Tutorial 2 Review

Commit:
{commit}

Base commit:
{base}

## What changed

Two final semantics rounds on top of the Teaching-View refactor:

* Timing is now a true synchronization model, not send-vs-round:
  - Student Local Clock (per rank, own clock, ms from its own Round Start):
    Send / Receive / SUM / COPY / Local Work Completed At. Every timestamp is
    recorded at the REAL operation site (World.send/recv return;
    `comm.note_operation_complete("sum"|"copy")` one line after the real
    combine()/copy in the collectives — observability only, algorithm
    unchanged; world.sync_round() fixes Local Work Completed immediately at
    the end of the algorithm work, before UI printing / event upload, so the
    timing boundary is never polluted (tests F/B/P).
  - Teacher Rank-0 single clock: per teaching round the teacher observes when
    each rank's barrier token arrives (stamped on the ROOT's clock by the
    transport reader thread) plus its own local-work-done instant, then shows

        TIMING — Rank 0 Observation
        Rank r ready at: ... ms | wait for others: ... ms
        Whole Round Finished: ...

    All from one clock, so wait_for_others == whole - ready is meaningful
    (Test N), and a slow rank is visible (Test O, ~1 s delay -> its ready is
    ~1 s later and the other three report ~1 s wait).
  - `ready` is documented as "observed at rank 0" (includes a tiny token
    transmission), never "finished exactly at".
* worker.py main flow is now a student-readable real MPI program:
  Init -> COMM_WORLD/rank/size -> wait_for_run() -> input -> data ->
  comm.Barrier() -> collective -> result -> repeat -> Finalize. Control-plane
  reading moved to minimpi/classroom_worker.py (one queue; main thread
  executes RUNs sequentially — no daemon callbacks, no `M._session` in
  worker.py, no passive wait loop). See WORKER_FLOW.md.

## Teacher View

Global Communication + Rank 0 Local View (unchanged, §previous round): edges
with payload size and Ring chunk ids, OPERATIONS +SUM/+COPY, real rank-0
BEFORE/SEND/RECEIVE/OPERATION/AFTER. Timing section is now the Rank-0
Observation barrier-arrival table (above).

## Student View

Same BEFORE/SEND/RECEIVE/OPERATION/AFTER local semantics; the timing block is
now `TIMING — Local Clock` with only the Completed-At events that really
happened this round (e.g. receiver+reduce shows Receive / SUM / Local Work
Completed At) and "Waiting at round synchronization...".

## Tree

Reduce / Broadcast phase labels and real rank-0 semantics unchanged.

## Ring

Reduce-Scatter shows real chunk SUM arithmetic (Send/Receive Completed At +
SUM Completed At on the local clock); AllGather shows MY REDUCED CHUNK +
COLLECTED RESULT table (COPY Completed At) ending with the Final vector.

## Start Barrier

Visible in Teaching Mode; teacher ENTER happens before record_start() so the
pause never enters Round 1 / Collective Time.

## Performance Mode

No teaching state printed; the local timeline and ready observation are only
recorded when mode == teaching (detailed timings never go over the control
plane).

## Tests

scripts/run_teaching_review.py runs: test_smoke, test_round_behavior (A-F),
test_teaching_semantics (G-K), test_timing_worker (L-S), verify.py.
Real outputs are in TEST_RESULTS.txt and tests/*.txt.

## Remaining Issues

* Student Local Clock values are per-rank clocks; the teacher Ready table is
  the rank-0 clock — the two are never subtracted across clocks (README).
* Remote "ready at" includes a small token transmission observed at rank 0.
* Barrier is a star-shaped AllReduce-of-1 teaching implementation.
* Benchmark rows are single runs per size (class experiment runner should
  use 3 runs / median when the PPT is prepared).
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


def run_benchmark(out_dir, timeout=300):
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
    for w in workers:
        try:
            w.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            w.kill()
            w.communicate()
    raw = os.path.join(out_dir, "raw")
    os.makedirs(raw, exist_ok=True)
    open(os.path.join(raw, "teacher.log"), "w").write(t_out)
    _write_benchmark_artifacts(t_out, out_dir)
    return t_out


def _write_benchmark_artifacts(t_out, out_dir):
    summary = []
    csv_rows = [["local_bytes", "naive_allreduce_ms", "tree_allreduce_ms",
                 "ring_allreduce_ms"]]
    for ln in t_out.splitlines():
        if re.match(r"^\s*\d+\s+[0-9.]+ ms", ln):
            parts = ln.split()
            if len(parts) >= 4:
                b = parts[0]
                ms = []
                for tok in parts[1:4]:
                    m = re.match(r"^([0-9.]+)", tok)
                    ms.append(m.group(1) if m else "")
                summary.append(ln)
                csv_rows.append([b] + ms)
    open(os.path.join(out_dir, "benchmark_summary.txt"), "w").write(
        "\n".join(summary) + "\n" if summary else "(no benchmark rows parsed)\n")
    with open(os.path.join(out_dir, "benchmark.csv"), "w") as f:
        for row in csv_rows:
            f.write(",".join(row) + "\n")


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
    """TIMING_AUDIT.md: real numbers from the first teaching round captured,
    plus the student-vs-teacher clock explanation."""
    log = open(teacher_log_path).read() if os.path.exists(teacher_log_path) else ""
    rows = []
    whole = None
    i = log.find("TIMING — Rank 0 Observation")
    if i >= 0:
        seg = log[i:]
        for ln in seg.splitlines():
            m = re.search(
                r"Rank (\d+) ready at:\s+([0-9.]+) ms \| wait for others:\s+"
                r"([0-9.]+) ms", ln)
            if m:
                rows.append((int(m.group(1)), m.group(2), m.group(3)))
            m2 = re.search(r"Whole Round Finished:\s*([0-9.]+)\s*ms", ln)
            if m2:
                whole = m2.group(1)
                break
    lines = ["# TIMING_AUDIT.md — real Round from the captured tree_allreduce run",
             "", "Teacher Round Start: 0.000 ms (Round Start on the rank-0 clock)",
             ""]
    if not rows:
        lines.append("(no timing block found in log)")
    else:
        for rk, m, w in rows:
            tag = "  (rank 0 == teacher's own local-work-done instant)"
            lines.append("Rank %s Ready: %s ms   wait for others: %s ms%s"
                         % (rk, m, w, tag if rk == 0 else ""))
        lines.append("")
        lines.append("Whole Round: %s ms" % (whole or "n/a"))
        lines.append("")
    lines += [
        "## Student Local Clock vs Teacher Rank-0 Observation Clock",
        "",
        "Student `TIMING — Local Clock`: each rank measures its OWN send /",
        "recv / SUM / COPY / Local-Work Completed At on ITS OWN clock, ms from",
        "its own Round Start. Those values are for understanding the rank's",
        "own execution phases and are NEVER subtracted across clocks.",
        "",
        "Teacher `TIMING — Rank 0 Observation`: rank 0 stamps, on ITS OWN",
        "single clock, when it observes each rank reach the round barrier",
        "(its barrier token arriving at rank 0's transport) plus its own",
        "local-work-done instant. wait_for_others == Whole - ready is only",
        "valid INSIDE this table (same clock); it answers \"who is waiting",
        "for whom\" and \"what the whole round really waited for\".",
        "",
        "Do not compute  TeacherReady - StudentLocalWork  across clocks.",
    ]
    open(os.path.join(pkg, "TIMING_AUDIT.md"), "w").write("\n".join(lines) + "\n")


def main():
    # --- working dir ------------------------------------------------------
    tmp = tempfile.mkdtemp(prefix="minimpi_review_")
    pkg = os.path.join(tmp, "review_package")
    os.makedirs(os.path.join(pkg, "source"))
    os.makedirs(os.path.join(pkg, "benchmark", "raw"))
    os.makedirs(os.path.join(pkg, "tests"))
    for demo in ("naive_allreduce", "tree_allreduce", "ring_allreduce"):
        os.makedirs(os.path.join(pkg, "teaching", demo))

    head = sh(["git", "rev-parse", "HEAD"], cwd=REPO).stdout.strip()
    base = sh(["git", "rev-parse", "HEAD~1"], cwd=REPO).stdout.strip()

    print("[1/7] source archive ...")
    sh(["git", "archive", "HEAD", "mpi_tutorial2",
        "-o", os.path.join(pkg, "source", "mpi_tutorial2_src.tar.gz")],
       cwd=REPO)

    print("[2/7] teaching logs (naive / tree / ring allreduce) ...")
    tree_log = None
    for demo in ("naive_allreduce", "tree_allreduce", "ring_allreduce"):
        ok, t_out = run_demo(os.path.join(pkg, "teaching", demo), demo)
        if demo == "tree_allreduce":
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
