#!/usr/bin/env python3
"""local_benchmark.py — run the SAME Performance Benchmark as teacher menu 7,
but with ALL ranks on THIS machine (loopback).

Use it to get a local baseline: compare these numbers with the same run
across real machines (WiFi/LAN) to see what the network adds to collective
communication.

Usage:
    python3 scripts/local_benchmark.py --size 4
    python3 scripts/local_benchmark.py --size 8 --sizes 16,1024,16384,262144
    python3 scripts/local_benchmark.py --size 4 --timeout 900

--size       number of ranks (teacher + size-1 local workers), default 4
--sizes      local payload sizes in BYTES, comma separated
             (default = teacher's BENCH_SIZES: 16 B .. 4 MB)
--timeout    seconds to wait for the benchmark (default 900)

Values are generated locally (headless workers use rank+1), so nothing has
to be typed. The teacher's raw lines and median summary are printed; the
summary is also broadcast to every worker by design.
"""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from _proc import Runner  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=4,
                    help="number of ranks incl. teacher (default 4)")
    ap.add_argument("--sizes", default="",
                    help="comma-separated payload bytes (default: 16..4MB)")
    ap.add_argument("--timeout", type=float, default=900.0)
    args = ap.parse_args()

    if args.size < 2:
        sys.exit("need at least 2 ranks")
    if args.sizes.strip():
        os.environ["MINIMPI_BENCH_SIZES"] = args.sizes.strip()

    print("=" * 62)
    print("MiniMPI LOCAL benchmark (all ranks on this machine / loopback)")
    print("Ranks      : %d" % args.size)
    print("Sizes      : %s" % (args.sizes.strip() or "16,1024,16384,262144,4194304 (default)"))
    print("Compare    : run the same thing across real machines (WiFi/LAN)")
    print("=" * 62)

    r = Runner(args.size, timeout=args.timeout)
    t0 = time.time()
    t = r.teacher(["--benchmark"])
    time.sleep(2.0)                      # let the teacher listen
    workers = r.workers()

    deadline = time.time() + args.timeout
    while time.time() < deadline and t.poll() is None:
        time.sleep(0.3)
    timed_out = t.poll() is None
    if timed_out:
        r.kill()
        sys.exit("[TIMEOUT] benchmark did not finish in %gs" % args.timeout)

    out, _ = t.communicate(timeout=10)
    elapsed = time.time() - t0

    # workers exit after the teacher's SHUTDOWN; collect a short confirmation
    w_notes = []
    for w in workers:
        try:
            o, _ = w.communicate(timeout=5)
        except Exception:  # noqa: BLE001
            o = ""
        w_notes.append("Performance Benchmark Results" in (o or ""))
    r.kill()

    print(out or "")
    print("-" * 62)
    print("elapsed: %.1f s   |   workers that also showed the summary: %d/%d"
          % (elapsed, sum(w_notes), len(w_notes)))
    print("(loopback baseline — compare with the same command on real machines)")


if __name__ == "__main__":
    main()
