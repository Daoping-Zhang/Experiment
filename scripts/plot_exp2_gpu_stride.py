#!/usr/bin/env python3
"""Plot the GPU stride-vs-bandwidth curve (Experiment 2A) on its own."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

p = os.path.join(ROOT, "results", "exp2", "gpu_memory_mapping.csv")
if not os.path.exists(p):
    print(f"missing {p} — run experiment 2A first")
    raise SystemExit(1)

df = pd.read_csv(p)
df = df[df["platform"].astype(str).str.upper() == "GPU"].copy()
df["stride"] = pd.to_numeric(df["stride"], errors="coerce")
df = df.dropna(subset=["stride"]).sort_values("stride")
bw = pd.to_numeric(df["effective_bandwidth_gbs"], errors="coerce")

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(df["stride"], bw, "o-", color="tab:red")
ax.set_xlabel("Stride")
ax.set_ylabel("Effective bandwidth (GB/s)")
ax.set_title("GPU: coalesced (stride 1) vs strided access")
ax.set_xticks(df["stride"])
ax.grid(True, alpha=0.3)
fig.tight_layout()
out = os.path.join(FIG, "exp2_gpu_stride_bandwidth.png")
fig.savefig(out, dpi=150)
print(f"[figure] {out}")
