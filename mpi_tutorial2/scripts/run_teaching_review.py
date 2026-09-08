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
        ├── source/
        │   └── mpi_tutorial2_src.tar.gz        (git archive HEAD)
        ├── teaching/
        │   ├── naive_allreduce/{teacher,rank1..3}.log
        │   ├── tree_allreduce/  {teacher,rank1..3}.log
        │   └── ring_allreduce/  {teacher,rank1..3}.log
        └── benchmark/
            ├── benchmark_summary.txt
            ├── benchmark.csv
            └── raw/teacher.log

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

Teaching Mode presentation layer (no collectives/*.py changes, no new
architecture):

* New minimpi/teaching.py — pure presentation semantics shared by teacher and
  workers: per-algorithm phase / round mapping (e.g. Tree AllReduce P=4:
  rounds 1-2 `Phase 1: Reduce`, rounds 3-4 `Phase 2: Broadcast`; Ring: 3x
  Reduce-Scatter then 3x AllGather), the Ring chunk schedule (identical
  arithmetic to collectives/ring_allreduce.py), bounded vector preview, and
  the per-round local semantics BEFORE / SEND / RECEIVE / OPERATION / AFTER.

  Precise data model: the local views are RECONSTRUCTED from each rank's real
  communication events (send value_before / recv value_after) and its real
  initial data, applying the same operation semantics as the collectives
  (presentation-side replay; collective internals are not instrumented and
  no collective file was changed).
* Ring AllGather presentation: during AllGather the rank's own vector does
  not change (the module assembles the result internally), so the local view
  shows the thing that really grows — MY REDUCED CHUNK, per-round SEND /
  RECEIVE of already-reduced chunks, "COPY Chunk k into AllGather result",
  a COLLECTED RESULT table (Chunk 0..P-1, "-" = not yet) and "n / P chunks
  collected", ending with the full Final vector at 4/4.
* Teacher (rank 0) is a real MPI participant and now shows, per teaching
  round, the fixed 4-part view: 1) GLOBAL COMMUNICATION (who->whom, payload
  size, Ring chunk ids) + OPERATIONS summary, 2) RANK 0 LOCAL VIEW (real
  rank-0 data), 3) TIMING (only ranks that really sent; Whole Round
  Finished), 4) Teacher Control ([ENTER] pacing, never inside round time).
* Start Barrier is visible in Teaching Mode: students print "Local data
  ready / Entering MPI Barrier...", rank 0 prints the barrier block and
  waits for ENTER before record_start() — the pause never enters Round 1 or
  Collective Time.
* Student terminal per round: Algorithm / Phase / Round x/y / Rank, My Role
  (Sender / Receiver / +Reduce / +Copy / explicit Idle), BEFORE, SEND,
  RECEIVE, OPERATION (SUM/COPY with the real arithmetic), AFTER, TIMING
  (My Send Finished / N/A) and "Waiting for the whole round...".
* Final blocks: Collective Complete (allreduce: Result + same-on-all-ranks;
  reduce: only root owns the result). Performance Mode prints none of the
  teaching state.

## Teacher View

Global View:
GLOBAL COMMUNICATION shows only topology edges: "Rank a -> Rank b <bytes>"
(+ "Chunk k" for Ring). OPERATIONS lists data-changing receives per rank
("+ SUM" / "+ COPY"). No other rank's full buffer is printed.

Rank 0 Local View:
Per teaching round the teacher shows its OWN real data with the same
Before/Send/Receive/Operation/After model as students (built from real
rank-0 execution events, verified against the final real result).

## Student View

Before / Send / Receive / Operation / After:
Same model on the student terminal; payload values come from the real send
/ recv records (not from a UI-side mock). Vectors are previewed (<= 8
elements) with an element count for large data.

## Tree

Reduce / Broadcast:
P=4 Tree AllReduce prints Round 1/4 & 2/4 as `Phase 1: Reduce`, Round 3/4 &
4/4 as `Phase 2: Broadcast`. Rank-0 local view round 1: Before [1,1,1,1],
Receive [2,2,2,2] (from rank 1), After [3,3,3,3]. Ranks that already sent
show explicit "My Role: Idle" on intermediate rounds.

## Ring

Reduce-Scatter / AllGather:
P=4 -> 6 rounds: 3x Reduce-Scatter then 3x AllGather. Global view labels
each message with its Chunk id (e.g. "Rank 3 -> Rank 0    Chunk 3    4 B").
Reduce-Scatter shows real SUM arithmetic on the received chunk
("4 + 3 = 7"); AllGather shows COPY semantics, never SUM.

Chunk semantics:
Ring chunk indices are computed by the presentation layer with the exact
schedule of collectives/ring_allreduce.py (send_idx=(rank-step)%P in RS;
owned=(rank+1)%P in AG), so the labels match what the algorithm really
moves.

## Start Barrier

Teaching Mode prints a Start Barrier block ("Waiting for all ranks... / All
ranks ready.") and the teacher presses ENTER before the collective starts;
record_start() happens after ENTER, so the pause is excluded from Round 1
and Collective Time. Start Barrier stays an AllReduce-of-1 (MiniMPI
teaching implementation).

## Performance Mode

No BEFORE/AFTER/MY ROLE/CHUNK/GLOBAL COMMUNICATION/OPERATIONS/Whole Round
Finished/Start Barrier text is printed; benchmark measures Collective Time
(Start Barrier complete -> all ranks done) and Session wall time.

## Tests

Run via scripts/run_teaching_review.py (see TEST_RESULTS.txt):
- tests/test_smoke.py
- tests/test_round_behavior.py  (A-F)
- tests/test_teaching_semantics.py (G-K)
- scripts/verify.py

## Remaining Issues

* Per-rank "My Send Finished" uses each rank's own clock; Whole Round
  Finished uses the teacher clock (fine on one machine / classroom LAN).
* Benchmark rows in this package are single runs per size (for the class
  performance experiment the runner should later use 3 runs / median).
* The Barrier is a star-shaped AllReduce-of-1 (teaching implementation, not
  production MPI's barrier algorithm).
"""


def sh(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def free_port():
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


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
        m = re.search(r"Rank:\s*(\d+)\s*/\s*\d+", o)
        rk = int(m.group(1)) if m else None
        if rk is not None:
            written[rk] = o
    for rk in sorted(written):
        open(os.path.join(out_dir, "rank%d.log" % rk), "w").write(written[rk])
    if len(written) < len(w_out):
        open(os.path.join(out_dir, "NOTE.txt"), "w").write(
            "WARNING: %d worker logs could not be matched to a real rank "
            "(join-order race) — see raw outputs below.\n\n%s"
            % (len(w_out) - len(written), "\n".join(w_out)))
    return not timed


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
    return "--- %s ---\n$ %s\n%s\n" % (
        tag, " ".join(argv), out)


def git_info():
    lines = []
    for cmd in (["git", "rev-parse", "HEAD"],
                ["git", "status", "--short"],
                ["git", "log", "--oneline", "-5"]):
        r = sh(cmd, cwd=REPO)
        lines.append("$ %s\n%s" % (" ".join(cmd), (r.stdout or r.stderr).strip()))
        lines.append("")
    return "\n".join(lines)


def main():
    # --- working dir ------------------------------------------------------
    tmp = tempfile.mkdtemp(prefix="minimpi_review_")
    pkg = os.path.join(tmp, "review_package")
    os.makedirs(os.path.join(pkg, "source"))
    os.makedirs(os.path.join(pkg, "benchmark", "raw"))
    for demo in ("naive_allreduce", "tree_allreduce", "ring_allreduce"):
        os.makedirs(os.path.join(pkg, "teaching", demo))

    head = sh(["git", "rev-parse", "HEAD"], cwd=REPO).stdout.strip()
    base = sh(["git", "rev-parse", "HEAD~1"], cwd=REPO).stdout.strip()

    print("[1/6] source archive ...")
    sh(["git", "archive", "HEAD", "mpi_tutorial2",
        "-o", os.path.join(pkg, "source", "mpi_tutorial2_src.tar.gz")],
       cwd=REPO)

    print("[2/6] teaching logs (naive / tree / ring allreduce) ...")
    for demo in ("naive_allreduce", "tree_allreduce", "ring_allreduce"):
        ok = run_demo(os.path.join(pkg, "teaching", demo), demo)
        print("   %-16s %s" % (demo, "ok" if ok else "TIMED OUT"))

    print("[3/6] benchmark ...")
    run_benchmark(os.path.join(pkg, "benchmark"))

    print("[4/6] tests ...")
    results = []
    results.append(run_cmd_capture("test_smoke.py",
                                   [PY, "tests/test_smoke.py"], 300))
    results.append(run_cmd_capture("test_round_behavior.py",
                                   [PY, "tests/test_round_behavior.py"], 420))
    results.append(run_cmd_capture("test_teaching_semantics.py",
                                   [PY, "tests/test_teaching_semantics.py"], 420))
    results.append(run_cmd_capture("verify.py",
                                   [PY, "scripts/verify.py"], 900))

    print("[5/6] review notes / git info / test results ...")
    open(os.path.join(pkg, "REVIEW_NOTES.md"), "w").write(
        REVIEW_NOTES.format(commit=head, base=base))
    open(os.path.join(pkg, "GIT_INFO.txt"), "w").write(git_info())
    open(os.path.join(pkg, "TEST_RESULTS.txt"), "w").write(
        "Generated: %s\n\n%s" % (time.strftime("%Y-%m-%d %H:%M:%S"),
                                 "\n".join(results)))

    print("[6/6] tar.gz ...")
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_name = "minimpi_tutorial2_review_%s_%s.tar.gz" % (head[:12], ts)
    out_path = os.path.join(REPO, out_name)
    with tarfile.open(out_path, "w:gz") as tf:
        tf.add(pkg, arcname="review_package")
    shutil.rmtree(tmp, ignore_errors=True)
    print("\nReview package generated:\n\n%s\n" % out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
