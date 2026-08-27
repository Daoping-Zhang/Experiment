#!/usr/bin/env python3
"""Best-effort NVIDIA Nsight Compute (ncu) profiler with a SEMANTIC layer.

Instead of dumping raw metric names, it maps teaching concepts to whatever
metric names the installed Nsight Compute version actually exposes. It queries
`ncu --query-metrics --query-metrics-mode all` (so suffixed/full metric names
are included), selects the first matching metric per concept, runs ncu, and
prints a small fixed-label table (source metric shown in small print).

Usage:
  ncu_profile.py --preset memory|divergence|occupancy --label "..." -- CMD...
  ncu_profile.py --patterns "sector,dram" --label "..." -- CMD...   # raw fallback

If ncu is missing, or no metric matches, it prints a friendly note and exits 0
so live demos never crash.
"""
import argparse
import csv
import re
import shutil
import subprocess
import sys

METRIC_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_.]*$")

# Concept -> (display label, ordered candidate substrings, lowercase).
PRESETS = {
    "memory": [
        ("Kernel duration",       ["gpu__time_duration"]),
        ("DRAM throughput",       ["dram__throughput", "dram__bytes"]),
        ("Global load sectors",   ["l1tex__t_sectors_pipe_lsu_mem_global_op_ld",
                                   "lts__t_sectors_srcunit_tex_op_read", "gld_transactions"]),
        ("Global store sectors",  ["l1tex__t_sectors_pipe_lsu_mem_global_op_st",
                                   "lts__t_sectors_srcunit_tex_op_write", "gst_transactions"]),
        ("L2 sectors read",       ["lts__t_sectors_srcunit_tex_op_read", "lts__t_sectors"]),
        ("L2 sectors write",      ["lts__t_sectors_srcunit_tex_op_write", "lts__t_sectors"]),
    ],
    "divergence": [
        ("Kernel duration",       ["gpu__time_duration"]),
        ("Executed instructions", ["smsp__inst_executed", "inst_executed"]),
        ("Thread instructions",   ["smsp__thread_inst_executed", "thread_inst_executed"]),
        ("Uniform branch %",      ["smsp__sass_average_branch_targets_threads_uniform",
                                   "branch_targets_threads_uniform", "branch_efficiency"]),
        ("Active warps",          ["sm__warps_active", "average_warps_active", "warps_active"]),
        ("Warp cycles/issue",     ["smsp__average_warps_active_per_issue_active"]),
    ],
    "occupancy": [
        ("Kernel duration",       ["gpu__time_duration"]),
        ("Achieved occupancy",    ["sm__warps_active", "achieved_occupancy", "average_warps_active"]),
        ("SM throughput",         ["sm__throughput", "sm__inst_executed_pipe"]),
        ("Compute+mem throughput",["gpu__compute_memory_throughput"]),
        ("DRAM throughput",       ["dram__throughput", "dram__bytes"]),
        ("Executed instructions", ["smsp__inst_executed", "inst_executed"]),
    ],
}


def query_metrics():
    if shutil.which("ncu") is None:
        return []
    try:
        out = subprocess.run(["ncu", "--query-metrics", "--query-metrics-mode", "all"],
                             capture_output=True, text=True, timeout=180)
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


def find_metric(metrics, candidates):
    lowered = [m.lower() for m in metrics]
    for c in candidates:
        c = c.lower()
        for m, ml in zip(metrics, lowered):
            if c in ml:
                return m
    return None


def resolve_preset(metrics, preset):
    """Return list of (label, metric)."""
    out = []
    for label, candidates in PRESETS[preset]:
        m = find_metric(metrics, candidates)
        if m is not None:
            out.append((label, m))
    return out


def run_ncu(metrics, cmd):
    run = ["ncu", "--csv", "--metrics", ",".join(metrics)] + cmd
    try:
        res = subprocess.run(run, capture_output=True, text=True, timeout=900)
    except Exception as e:  # noqa: BLE001
        print("ncu run failed:", e)
        return None
    try:
        rows = list(csv.reader(res.stdout.splitlines()))
    except Exception:  # noqa: BLE001
        print(res.stdout[-2000:])
        return None
    if not rows:
        return None
    header = rows[0]
    data = rows[1] if len(rows) > 1 else None

    def value(m):
        idx = next((i for i, h in enumerate(header) if h == m or m in h), None)
        return data[idx] if (idx is not None and data and idx < len(data)) else "N/A"

    return [(m, value(m)) for m in metrics]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=sorted(PRESETS.keys()))
    ap.add_argument("--patterns", default="")
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

    print(f"\n=== {args.label} ===\n")

    if args.preset:
        pairs = resolve_preset(metrics, args.preset)
        if pairs:
            values = run_ncu([m for _, m in pairs], rest)
            if values is None:
                print("ncu produced no CSV output")
                sys.exit(0)
            valmap = dict(values)
            for label, m in pairs:
                print(f"  {label:<28} {valmap.get(m, 'N/A')}   [{m}]")
            print()
            sys.exit(0)
        print(f"no preset metrics matched for '{args.preset}'; falling back to patterns\n")

    # raw pattern fallback
    pats = [p.strip().lower() for p in args.patterns.split(",") if p.strip()]
    sel = [m for m in metrics if any(p in m.lower() for p in pats)][: args.limit]
    if not sel:
        print(f"no metrics matched patterns: {args.patterns}")
        sys.exit(0)
    values = run_ncu(sel, rest)
    if values is None:
        print("ncu produced no CSV output")
        sys.exit(0)
    for m, v in values:
        print(f"  {m:<58} {v}")
    print()


if __name__ == "__main__":
    main()
