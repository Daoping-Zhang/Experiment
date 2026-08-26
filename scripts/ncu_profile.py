#!/usr/bin/env python3
"""Best-effort NVIDIA Nsight Compute (ncu) profiler — extract only the metrics
relevant to a teaching demo, instead of dumping hundreds of lines.

Usage:
  ncu_profile.py --patterns "sector,transaction,dram" --label "stride 32" \
                 -- ./bin/exp2_gpu_memory --stride 32 ...

It queries `ncu --query-metrics` for the metrics available on THIS GPU /
Nsight version (so metric-name differences across versions do not matter),
selects those whose names contain any of the given substrings, runs ncu with
`--csv`, and prints a small aligned table. If ncu is missing it exits 0 with a
friendly note so live demos never crash.
"""
import argparse
import csv
import re
import shutil
import subprocess
import sys

METRIC_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_.]*$")


def query_metrics():
    if shutil.which("ncu") is None:
        return []
    try:
        out = subprocess.run(["ncu", "--query-metrics"], capture_output=True, text=True,
                             timeout=120)
    except Exception:
        return []
    names, seen = [], set()
    for line in out.stdout.splitlines():
        for tok in line.split():
            tok = tok.strip('",:()')
            if METRIC_RE.match(tok) and "." in tok and tok not in seen:
                seen.add(tok)
                names.append(tok)
    return names


def select(metrics, patterns):
    pats = [p.strip().lower() for p in patterns if p.strip()]
    return [m for m in metrics if any(p in m.lower() for p in pats)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patterns", required=True)
    ap.add_argument("--label", default="profile")
    ap.add_argument("--limit", type=int, default=12)
    args, rest = ap.parse_known_args()
    if rest and rest[0] == "--":
        rest = rest[1:]
    if not rest:
        print("no command to profile")
        sys.exit(1)

    if shutil.which("ncu") is None:
        print("ncu not found (skipping hardware profile)")
        sys.exit(0)

    metrics = query_metrics()
    if not metrics:
        print("ncu --query-metrics returned nothing (skipping)")
        sys.exit(0)

    sel = select(metrics, args.patterns.split(","))
    if not sel:
        print(f"no metrics matched patterns: {args.patterns}")
        sys.exit(0)
    sel = sel[: args.limit]

    print(f"\n=== {args.label} ===\n")
    print(f"metrics matched: {', '.join(sel)}\n")

    run = ["ncu", "--csv", "--metrics", ",".join(sel)] + rest
    try:
        res = subprocess.run(run, capture_output=True, text=True, timeout=900)
    except Exception as e:  # noqa: BLE001
        print("ncu run failed:", e)
        sys.exit(0)

    try:
        rows = list(csv.reader(res.stdout.splitlines()))
    except Exception:  # noqa: BLE001
        print(res.stdout[-2000:])
        sys.exit(0)
    if not rows:
        print("ncu produced no CSV output")
        sys.exit(0)

    header, data = rows[0], (rows[1] if len(rows) > 1 else None)
    for m in sel:
        idx = next((i for i, h in enumerate(header) if h == m or m in h), None)
        val = data[idx] if (idx is not None and data and idx < len(data)) else "N/A"
        print(f"  {m:<58} {val}")
    print()


if __name__ == "__main__":
    main()
