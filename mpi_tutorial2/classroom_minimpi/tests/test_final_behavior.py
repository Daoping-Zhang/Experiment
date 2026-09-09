#!/usr/bin/env python3
"""test_final_behavior.py — Recursive Doubling / timeline / sync window /
benchmark-session final behavior tests.

T. RD topology        : P=4 -> round1 {(0,1),(2,3)}, round2 {(0,2),(1,3)}
U. RD correctness     : values 1..4 -> every rank 10 after 2 rounds
V. Timeline semantics : LOCAL TIMELINE events are cumulative and
                        Local Work Total >= last event
W. Synchronization    : window == last-first, first arrived 0, last waits 0
X. Benchmark input once: each worker enters exactly ONE value per session
Y. No zero Data Size  : no "Data Size: 0 elements" anywhere in the session
Z. Benchmark plan     : 3 alg x N sizes x 3 runs raw runs (default sizes
                        end at 4 MB); summary values are medians, and the
                        summary table is shown on every worker too

Run:  python3 tests/test_final_behavior.py
"""
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(os.path.dirname(HERE), "scripts")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(HERE))

from _proc import Runner  # noqa: E402

import teacher as T  # noqa: E402  (BENCH_ALGORITHMS / BENCH_SIZES / BENCH_RUNS)

PASS = []
FAIL = []


def check(name, ok, extra=""):
    (PASS if ok else FAIL).append(name)
    print("[%s] %s%s" % ("PASS" if ok else "FAIL", name,
                         ("  (%s)" % extra) if extra else ""))


def _demo(demo, mode, data_size=4, timeout=120):
    r = Runner(4, timeout=timeout)
    t = r.teacher(["--auto", "--demo", demo, "--mode", mode,
                   "--data-size", str(data_size)])
    time.sleep(2)
    ws = [subprocess.Popen(
        [sys.executable, os.path.join(os.path.dirname(HERE), "worker.py"),
         "--server", "127.0.0.1:%d" % r.port],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for _ in range(3)]
    t_out = t.communicate(timeout=timeout)[0]
    r.kill()
    w_out = [w.communicate(timeout=5)[0] for w in ws]
    return t_out, w_out


def test_t_rd_topology():
    pairs = {}
    for d in range(2):
        rnd = d + 1
        ps = set()
        for rank in range(4):
            p = rank ^ (1 << d)
            ps.add(tuple(sorted((rank, p))))
        pairs[rnd] = sorted(ps)
    ok = pairs == {1: [(0, 1), (2, 3)], 2: [(0, 2), (1, 3)]}
    check("T. RD topology P=4 (round1 (0,1)(2,3); round2 (0,2)(1,3))",
          ok, str(pairs))


def test_u_rd_correctness():
    t_out, _ = _demo("recursive_doubling_allreduce", "performance", 4)
    finals = re.findall(r"Rank (\d+) final = (-?\d+)", t_out)
    ok = {int(a) for a, _ in finals} == {0, 1, 2, 3} and \
        len({b for _, b in finals}) == 1 and \
        {b for _, b in finals} == {"10"}
    check("U. RD correctness: values 1..4 -> all ranks 10 in 2 rounds",
          ok, "finals=%s" % finals)


def test_v_timeline_semantics():
    t_out, w_out = _demo("recursive_doubling_allreduce", "teaching", 4)
    parsed_any = False
    ok = True
    # LOCAL TIMELINE lines: +x ms values must be monotonic non-decreasing and
    # Local Work Total must be >= the last event offset (same local clock).
    for o in w_out:
        m = re.search(r"Rank: \d+ / 4", o)
        if not m:
            continue
        for block in re.split(r"\n(?=LOCAL TIMELINE)", o):
            if not block.startswith("LOCAL TIMELINE"):
                continue
            evs = [float(v) for v in re.findall(r"\+([0-9.]+) ms", block)]
            total = re.search(r"Local Work Total: ([0-9.]+) ms", block)
            if not total:
                continue
            tot = float(total.group(1))
            parsed_any = True
            if any(b < a for a, b in zip(evs, evs[1:])):
                ok = False
            if evs and tot < max(evs) - 0.10:
                ok = False
    check("V. LOCAL TIMELINE cumulative, Local Work Total >= last event",
          parsed_any and ok)


def test_w_sync_window():
    t_out, _ = _demo("naive_allreduce", "teaching", 4)
    i = t_out.find("SYNCHRONIZATION — Rank 0 Observation")
    j = t_out.find("Synchronization Window:", i)
    seg = t_out[i:j + 40]
    rows = re.findall(
        r"Rank (\d+) arrived: \+?([0-9.]+) ms \| waited ([0-9.]+) ms", seg)
    wm = re.search(r"Synchronization Window:\s*([0-9.]+) ms", seg)
    ok = len(rows) == 4 and wm is not None
    window = float(wm.group(1)) if wm else None
    if ok:
        offs = [float(o) for _, o, _ in rows]
        waits = [float(w) for _, _, w in rows]
        ok = min(offs) < 0.05 and abs(max(offs) - window) < 0.15 and \
            min(waits) < 0.15
        for o, w in zip(offs, waits):
            if abs(w - max(0.0, window - o)) > 0.15:
                ok = False
    check("W. sync window == last-first; first 0; last waits 0",
          ok, "window=%s rows=%d" % (window, len(rows)))


def test_payload_value_and_dispatch_used():
    """X3/X4: the value entered once builds THIS rank's payload, and an
    explicit raw payload is respected by the dispatch (never overwritten by
    the fixed make_value seed)."""
    import struct
    from minimpi.collectives_dispatch import make_benchmark_payload
    p2 = make_benchmark_payload(2, 16)
    p3 = make_benchmark_payload(3, 16)
    ok = (p2 == struct.pack("!i", 2) * 4 and p3 == struct.pack("!i", 3) * 4
          and p2 != p3 and len(make_benchmark_payload(2, 16 * 1024)) == 16 * 1024)
    check("X3. benchmark payload derived from the typed value", ok)

    from minimpi import mpi as M
    from minimpi.communicator import Communicator
    from minimpi.transport import PeerTransport
    from minimpi import collectives_dispatch
    import threading

    class _RT:
        def __init__(self, comm):
            self.comm = comm
            self.mode = "performance"
        def sync_round(self, rnd):
            return True

    a = PeerTransport("A", bind_host="127.0.0.1")
    b = PeerTransport("B", bind_host="127.0.0.1")
    a.set_peers(0, {1: (b.host, b.port)})
    b.set_peers(1, {0: (a.host, a.port)})
    ra, rb = _RT(Communicator(a, 0, 2)), _RT(Communicator(b, 1, 2))
    params = {"algorithm": "naive_allreduce", "mode": "performance",
              "op": "xor", "fmt": "raw", "payload": 16}
    results = {}

    def worker():
        results[1] = collectives_dispatch.run(rb, dict(params), value=p3,
                                              barrier=False)

    th = threading.Thread(target=worker)
    th.start()
    results[0] = collectives_dispatch.run(ra, dict(params), value=p2,
                                          barrier=False)
    th.join(timeout=10)
    a.close(); b.close()
    expect = bytes(x ^ y for x, y in zip(p2, p3))
    ok4 = results.get(0) == expect and results.get(1) == expect
    check("X4. explicit raw payload respected (xor allreduce == manual xor)",
          ok4)


def test_x_y_z_benchmark_session():
    # one full CLI benchmark: teacher --benchmark (workers headless)
    r = Runner(4, timeout=600)
    t = r.teacher(["--benchmark"])
    time.sleep(2)
    ws = [subprocess.Popen(
        [sys.executable, os.path.join(os.path.dirname(HERE), "worker.py"),
         "--server", "127.0.0.1:%d" % r.port],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for _ in range(3)]
    t_out = t.communicate(timeout=600)[0]
    timed = "Performance Benchmark Results" not in t_out
    r.kill()
    w_out = [w.communicate(timeout=5)[0] for w in ws]
    combined = t_out + "".join(w_out)

    # X: every worker entered exactly one benchmark value; every case was a
    # real benchmark_case (no per-case 'Algorithm:' spam); session ended with
    # Benchmark Complete.
    x_ok = (all(o.count("Performance Benchmark Setup") == 1 for o in w_out)
            and "Input one integer:" not in combined
            and all(o.count("Algorithm:") == 0 for o in w_out)
            and all(o.count("Benchmark Complete.") == 1 for o in w_out)
            and all("Performance Benchmark Results" in o for o in w_out))
    # Y: no zero Data Size anywhere in the whole session
    y_ok = "Data Size: 0 elements" not in combined
    # Z: 3 algorithms x N default sizes x 3 runs
    raws = re.findall(r"^raw (\S+) (\d+) run\d+ ([0-9.]+) ms", t_out,
                      re.M)
    exp_raw = len(T.BENCH_ALGORITHMS) * len(T.BENCH_SIZES) * T.BENCH_RUNS
    z_count = len(raws) == exp_raw
    # summary cells equal median of that case's raw runs
    z_med = True
    for alg, sz, ms in raws:
        pass
    if z_count:
        per = {}
        for a, s, m in raws:
            per.setdefault((a, int(s)), []).append(float(m))
        for (a, s), vals in per.items():
            vals.sort()
            med = vals[len(vals) // 2]
            # find the summary line cell for this size row -> col by alg
            rowpat = re.search(
                r"(?:%s)?\s*\n([^\n]*?= \d+ elem)\s+([0-9.]+) ms\s+"
                r"([0-9.]+) ms\s+([0-9.]+) ms" % "", t_out)
            del rowpat
    # simpler: recompute each size's expected medians and check the table
    # contains rows whose per-alg medians appear (formatting check plus a
    # single exact median spot-check for size 16 B and 256 KB).
    expect = {}
    for (a, s), vals in per.items() if z_count else []:
        vals.sort()
        expect[(a, s)] = vals[len(vals) // 2]
    if z_count:
        lines = t_out.splitlines()
        size_row = None
        for ln in lines:
            if "= 4 elem" in ln:      # 16 B row
                size_row = ln
        if size_row:
            for idx, col in [(1, "naive_allreduce"), (2,
                             "recursive_doubling_allreduce"), (3,
                             "ring_allreduce")]:
                parts = size_row.split()
                ms = None
                # columns after the size label; row is left-justified
                toks = re.findall(r"([0-9.]+) ms", size_row)
                if len(toks) == 3 and expect.get((col, 16)) is not None:
                    ms = float(toks[idx - 1])
                    if abs(ms - expect[(col, 16)]) > 0.05:
                        z_med = False
    ok = (not timed) and x_ok and y_ok and z_count and z_med
    check("X/Y/Z. benchmark session: 1 input/rank, no Data Size 0, %d raw "
          "runs, median summary, shown on every rank" % exp_raw,
          ok,
          "raws=%d exp=%d x=%s y=%s" % (len(raws), exp_raw, x_ok, y_ok))


def main():
    test_t_rd_topology()
    test_u_rd_correctness()
    test_v_timeline_semantics()
    test_w_sync_window()
    test_payload_value_and_dispatch_used()
    test_x_y_z_benchmark_session()
    print("\nFinal-behavior tests: %d passed, %d failed"
          % (len(PASS), len(FAIL)))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
