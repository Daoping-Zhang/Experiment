#!/usr/bin/env python3
"""Compare CPU vs GPU for ONE Experiment 1 case (side-by-side table).

Usage:
  python3 compare_exp1.py dependent
  python3 compare_exp1.py independent   # all chains
  python3 compare_exp1.py branch
"""
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results", "exp1")


def load(name):
    p = os.path.join(RES, name)
    if not os.path.exists(p) or os.path.getsize(p) == 0:
        return None
    return pd.read_csv(p)


def num(s):
    return pd.to_numeric(s, errors="coerce")


def cell(v, unit=""):
    if v is None:
        return "N/A"
    return f"{v:.3f}{unit}"


def compare_dependent():
    df = load("dependent.csv")
    if df is None:
        print("no data for dependent"); return
    cpu = df[df["platform"].astype(str).str.upper() == "CPU"]
    gpu = df[df["platform"].astype(str).str.upper() == "GPU"]
    print("\n=== Experiment 1A: Dependent Chain (CPU 1 thread vs GPU 1 thread) ===\n")
    print(f"{'':16}{'CPU':>12}{'GPU':>12}")
    rows = [
        ("Latency", cell(num(cpu["latency_ms"]).median(), " ms") if len(cpu) else "N/A",
                    cell(num(gpu["latency_ms"]).median(), " ms") if len(gpu) else "N/A"),
        ("GFLOPS", cell(num(cpu["gflops"]).median()) if len(cpu) else "N/A",
                   cell(num(gpu["gflops"]).median()) if len(gpu) else "N/A"),
    ]
    for name, c, g in rows:
        print(f"{name:16}{c:>12}{g:>12}")
    print()


def compare_independent():
    df = load("independent.csv")
    if df is None:
        print("no data for independent"); return
    df = df.copy()
    df["chains"] = df["variant"].astype(str).str.extract(r"chains=(\d+)").astype(float)
    print("\n=== Experiment 1B: Independent Chains (GFLOPS) ===\n")
    print(f"{'chains':>8}{'CPU':>12}{'GPU':>12}")
    for c in sorted(df["chains"].dropna().unique()):
        sub = df[df["chains"] == c]
        cpu = num(sub[sub["platform"].astype(str).str.upper() == "CPU"]["gflops"]).median()
        gpu = num(sub[sub["platform"].astype(str).str.upper() == "GPU"]["gflops"]).median()
        print(f"{int(c):>8}{cell(cpu):>12}{cell(gpu):>12}")
    print()


def compare_branch():
    df = load("branch.csv")
    if df is None:
        print("no data for branch"); return
    print("\n=== Experiment 1C: Branch Prediction (throughput, elem/s) ===\n")
    print(f"{'':16}{'CPU':>12}{'GPU':>12}")
    for variant in ("predictable", "random"):
        sub = df[df["variant"] == variant]
        cpu = num(sub[sub["platform"].astype(str).str.upper() == "CPU"]["throughput_ops_s"]).median()
        gpu = num(sub[sub["platform"].astype(str).str.upper() == "GPU"]["throughput_ops_s"]).median()
        print(f"{variant:16}{cell(cpu):>12}{cell(gpu):>12}")

    # loss
    def loss(plat):
        sub = df[df["platform"].astype(str).str.upper() == plat]
        p = num(sub[sub["variant"] == "predictable"]["throughput_ops_s"]).median()
        r = num(sub[sub["variant"] == "random"]["throughput_ops_s"]).median()
        if not pd.isna(p) and not pd.isna(r) and p:
            return f"{max(0.0, 1 - r / p) * 100:.1f}%"
        return "N/A"

    print(f"{'loss':16}{loss('CPU'):>12}{loss('GPU'):>12}")
    print()


def main():
    case = sys.argv[1] if len(sys.argv) > 1 else "dependent"
    {"dependent": compare_dependent, "independent": compare_independent,
     "branch": compare_branch}.get(case, lambda: print(f"unknown case: {case}"))()


if __name__ == "__main__":
    main()
