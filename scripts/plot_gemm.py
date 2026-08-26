#!/usr/bin/env python3
"""Plot the GEMM CPU-vs-GPU GFLOPS curve on its own."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)


def load(rel):
    p = os.path.join(ROOT, "results", "gemm", rel)
    if not os.path.exists(p) or os.path.getsize(p) == 0:
        return None
    return pd.read_csv(p)


cpu = load("cpu_gemm.csv")
gpu = load("gpu_gemm.csv")
if cpu is None and gpu is None:
    print("no GEMM data — run scripts/run_gemm.sh first")
    raise SystemExit(1)

fig, ax = plt.subplots(figsize=(8, 5))
if cpu is not None:
    c = cpu.sort_values("matrix_size")
    ax.plot(pd.to_numeric(c["matrix_size"]), pd.to_numeric(c["gflops"]), "o-",
            color="tab:blue", label="CPU BLAS")
if gpu is not None:
    g = gpu.sort_values("matrix_size")
    ax.plot(pd.to_numeric(g["matrix_size"]), pd.to_numeric(g["gflops"]), "s-",
            color="tab:red", label="GPU cuBLAS")
ax.set_xlabel("Matrix size (n)")
ax.set_ylabel("GFLOPS")
ax.set_title("GEMM: optimized CPU BLAS vs cuBLAS")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
out = os.path.join(FIG, "ai_gemm_gflops.png")
fig.savefig(out, dpi=150)
print(f"[figure] {out}")
