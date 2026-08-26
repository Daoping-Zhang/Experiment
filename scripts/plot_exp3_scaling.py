#!/usr/bin/env python3
"""Plot the core thread-scaling curve (Experiment 3) on its own (log-log)."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)


def load(rel):
    p = os.path.join(ROOT, "results", "exp3", rel)
    if not os.path.exists(p) or os.path.getsize(p) == 0:
        return None
    return pd.read_csv(p)


cpu = load("cpu_scaling.csv")
gpu = load("gpu_scaling.csv")
if cpu is None and gpu is None:
    print("no exp3 scaling data — run experiment 3 first")
    raise SystemExit(1)

fig, ax = plt.subplots(figsize=(8, 5))
if cpu is not None:
    c = cpu.sort_values("threads")
    ax.plot(pd.to_numeric(c["threads"]), pd.to_numeric(c["throughput_elem_s"]), "o-",
            color="tab:blue", label="CPU")
if gpu is not None:
    g = gpu.sort_values("threads")
    ax.plot(pd.to_numeric(g["threads"]), pd.to_numeric(g["throughput_elem_s"]), "s-",
            color="tab:red", label="GPU")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Number of threads (log)")
ax.set_ylabel("Throughput (elements/s, log)")
ax.set_title("Few strong CPU threads vs massive lightweight GPU threads")
ax.legend()
ax.grid(True, alpha=0.3, which="both")
fig.tight_layout()
out = os.path.join(FIG, "exp3_thread_scaling_throughput.png")
fig.savefig(out, dpi=150)
print(f"[figure] {out}")
