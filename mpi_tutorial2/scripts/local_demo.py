#!/usr/bin/env python3
"""local_demo.py — run a MiniMPI demo entirely on one machine.

Launches the real teacher.py plus (size-1) worker.py processes on localhost,
so the full classroom path can be developed, CI-tested and rehearsed without
needing real machines.

Usage:
    python3 scripts/local_demo.py --size 4 --demo tree_allreduce --mode teaching
    python3 scripts/local_demo.py --size 4 --demo naive_reduce --mode performance
    python3 scripts/local_demo.py --size 8 --demo tree_reduce
"""
import argparse
import sys

sys.path.insert(0, __import__("os").path.dirname(__file__))

from _proc import Runner  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=4)
    ap.add_argument("--demo", default="naive_allreduce")
    ap.add_argument("--mode", default="performance")
    ap.add_argument("--timeout", type=int, default=120)
    args = ap.parse_args()

    r = Runner(args.size, timeout=args.timeout)
    log, ok, timed = r.run_demo(["--demo", args.demo, "--mode", args.mode])
    sys.stdout.write(log)
    r.close()
    if timed:
        print("\n[FAIL] demo timed out")
        return 1
    if "Errors" in log and not log.strip().endswith("Errors: {}"):
        print("\n[FAIL] see Errors above")
        return 1
    print("\n[local_demo] finished OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
