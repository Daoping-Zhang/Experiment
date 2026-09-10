#!/usr/bin/env python3
"""local_benchmark.py — run the SAME Performance Benchmark as teacher menu 7,
but with ALL ranks on THIS machine (loopback).

Use it to get a local baseline: compare these numbers with the same run
across real machines (WiFi/LAN) to see what the network adds to collective
communication.

Usage:
    python3 scripts/local_benchmark.py --size 4
    python3 scripts/local_benchmark.py --size 8
    python3 scripts/local_benchmark.py --size 4 --sizes 16,1024,16384
    python3 scripts/local_benchmark.py --size 4 --timeout 600

Output is LIVE (the teacher/worker processes write straight to your
terminal), so you can watch the `raw ...` lines and the median table appear.
Want to keep a copy?  Pipe it:

    python3 scripts/local_benchmark.py --size 4 | tee local4.log
"""
import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)          # classroom_minimpi
sys.path.insert(0, HERE)

from _proc import free_port  # noqa: E402


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
    port = free_port()
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    if args.sizes.strip():
        env["MINIMPI_BENCH_SIZES"] = args.sizes.strip()

    print("=" * 62)
    print("MiniMPI LOCAL benchmark (all ranks on this machine / loopback)")
    print("Ranks      : %d" % args.size)
    print("Sizes      : %s" % (args.sizes.strip()
                               or "16,1024,16384,262144,4194304 (default)"))
    print("Compare    : run the same thing across real machines (WiFi/LAN)")
    print("=" * 62)
    print("Starting teacher + %d local workers on 127.0.0.1:%d ..."
          % (args.size - 1, port))
    print("(output below is live; the benchmark itself takes a while — "
          "watch the 'raw ...' lines)\n")
    sys.stdout.flush()

    t0 = time.time()
    teacher = subprocess.Popen(
        [sys.executable, os.path.join(ROOT, "teacher.py"),
         "--size", str(args.size), "--host", "127.0.0.1",
         "--port", str(port), "--advertise", "127.0.0.1", "--benchmark"],
        cwd=ROOT, env=env)                      # stdout inherited => live
    time.sleep(2.0)
    workers = [subprocess.Popen(
        [sys.executable, os.path.join(ROOT, "worker.py"),
         "--server", "127.0.0.1:%d" % port], cwd=ROOT, env=env)
        for _ in range(args.size - 1)]

    try:
        rc = teacher.wait(timeout=args.timeout)
        timed_out = False
    except subprocess.TimeoutExpired:
        rc, timed_out = None, True

    for p in ([teacher] + workers):             # clean up (also if timed out)
        if p.poll() is None:
            try:
                p.kill()
            except Exception:  # noqa: BLE001
                pass
    for p in workers:
        try:
            p.wait(timeout=10)
        except Exception:  # noqa: BLE001
            pass

    elapsed = time.time() - t0
    print("-" * 62)
    if timed_out:
        print("[TIMEOUT] benchmark did not finish in %gs (processes killed)"
              % args.timeout)
        return 1
    print("elapsed: %.1f s   |   teacher exit code: %s" % (elapsed, rc))
    print("(loopback baseline — now run the same on real machines and compare)")
    return 0 if rc == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
