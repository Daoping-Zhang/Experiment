#!/usr/bin/env python3
"""Generate all figures for the Lecture 01 tutorial from results/*.csv.

Figures are written to ../figures/. Every figure is defensive: if the input
CSV is missing or lacks a platform (e.g. no GPU), that curve is skipped rather
than crashing, so the script still produces the CPU-side figures.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "figure.figsize": (8, 5),
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
})


def load(rel):
    p = os.path.join(RES, rel)
    if not os.path.exists(p) or os.path.getsize(p) == 0:
        return None
    try:
        return pd.read_csv(p)
    except Exception as e:  # noqa: BLE001
        print(f"[warn] could not read {p}: {e}")
        return None


def platform(df, name):
    if df is None:
        return None
    d = df[df["platform"].astype(str).str.upper() == name.upper()]
    return d if len(d) else None


def num(df, col):
    return pd.to_numeric(df[col], errors="coerce")


def save(fig, name):
    out = os.path.join(FIG, name)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[figure] {out}")


# ---------------------------------------------------------------------------
# Figure 1 — CPU IPC vs independent chains
# ---------------------------------------------------------------------------
def fig1():
    df = platform(load("exp1/independent.csv"), "CPU")
    if df is None:
        return
    df = df.copy()
    df["chains"] = df["variant"].astype(str).str.extract(r"chains=(\d+)").astype(float)
    df = df.dropna(subset=["chains"]).sort_values("chains")
    ipc = num(df, "ipc")
    if ipc.notna().all():
        fig, ax = plt.subplots()
        ax.plot(df["chains"], ipc, "o-", color="tab:blue")
        ax.set_xlabel("Independent chains")
        ax.set_ylabel("IPC (instructions / cycle)")
        ax.set_title("CPU single thread: ILP exploitation vs independent chains")
        ax.set_xticks(df["chains"])
        save(fig, "exp1_cpu_ipc_vs_independent_chains.png")
    else:
        print("[warn] IPC missing (no perf) — skipping fig1")


# ---------------------------------------------------------------------------
# Figure 2 — CPU vs GPU single-thread throughput (independent chains)
# ---------------------------------------------------------------------------
def fig2():
    df = load("exp1/independent.csv")
    if df is None:
        return
    df = df.copy()
    df["chains"] = df["variant"].astype(str).str.extract(r"chains=(\d+)").astype(float)
    df = df.dropna(subset=["chains"])
    fig, ax = plt.subplots()
    for plat, color in (("CPU", "tab:blue"), ("GPU", "tab:red")):
        d = platform(df, plat)
        if d is None or len(d) == 0:
            continue
        d = d.sort_values("chains")
        ax.plot(d["chains"], num(d, "gflops"), "o-", color=color, label=f"{plat} 1 thread")
    ax.set_xlabel("Independent chains")
    ax.set_ylabel("GFLOPS")
    ax.set_title("Single worker compute capability (CPU 1 thread vs GPU 1 thread)")
    ax.legend()
    save(fig, "exp1_cpu_gpu_single_thread_throughput.png")


# ---------------------------------------------------------------------------
# Figure 3 — branch prediction performance loss
# ---------------------------------------------------------------------------
def fig3():
    df = load("exp1/branch.csv")
    if df is None:
        return
    losses = []
    for plat in ("CPU", "GPU"):
        d = platform(df, plat)
        if d is None or len(d) == 0:
            continue
        pred = d[d["variant"] == "predictable"]
        rnd = d[d["variant"] == "random"]
        if len(pred) and len(rnd):
            tp = num(pred, "throughput_ops_s").median()
            tr = num(rnd, "throughput_ops_s").median()
            losses.append((plat, max(0.0, 1.0 - tr / tp) * 100.0))
    if not losses:
        return
    fig, ax = plt.subplots()
    ax.bar([p for p, _ in losses], [v for _, v in losses], color=["tab:blue", "tab:red"][:len(losses)])
    ax.set_ylabel("Throughput loss (%)")
    ax.set_title("Branch prediction: predictable -> random")
    for i, (_, v) in enumerate(losses):
        ax.text(i, v + 0.5, f"{v:.1f}%", ha="center")
    save(fig, "exp1_branch_performance_loss.png")


# ---------------------------------------------------------------------------
# Figure 4 — CPU memory mapping bandwidth
# ---------------------------------------------------------------------------
def fig4():
    df = platform(load("exp2/cpu_memory_mapping.csv"), "CPU")
    if df is None:
        return
    fig, ax = plt.subplots()
    ax.bar(df["mapping"], num(df, "effective_bandwidth_gbs"), color=["tab:green", "tab:orange"])
    ax.set_ylabel("Effective bandwidth (GB/s)")
    ax.set_title("CPU: block vs cyclic partitioning")
    for i, v in enumerate(num(df, "effective_bandwidth_gbs")):
        ax.text(i, v + 0.1, f"{v:.1f}", ha="center")
    save(fig, "exp2_cpu_mapping_bandwidth.png")


# ---------------------------------------------------------------------------
# Figure 5 — GPU stride bandwidth
# ---------------------------------------------------------------------------
def fig5():
    df = platform(load("exp2/gpu_memory_mapping.csv"), "GPU")
    if df is None:
        return
    df = df.copy()
    df["stride"] = num(df, "stride")
    df = df.dropna(subset=["stride"]).sort_values("stride")
    fig, ax = plt.subplots()
    ax.plot(df["stride"], num(df, "effective_bandwidth_gbs"), "o-", color="tab:red")
    ax.set_xlabel("Stride")
    ax.set_ylabel("Effective bandwidth (GB/s)")
    ax.set_title("GPU: coalesced (stride 1) vs strided access")
    ax.set_xticks(df["stride"])
    save(fig, "exp2_gpu_stride_bandwidth.png")


# ---------------------------------------------------------------------------
# Figure 6 — CPU vs GPU loss comparison (summary of Experiment 2)
# ---------------------------------------------------------------------------
def fig6():
    cpu_mem = platform(load("exp2/cpu_memory_mapping.csv"), "CPU")
    gpu_mem = platform(load("exp2/gpu_memory_mapping.csv"), "GPU")
    cpu_ctrl = platform(load("exp2/cpu_work_mapping.csv"), "CPU")
    gpu_ctrl = platform(load("exp2/gpu_work_mapping.csv"), "GPU")

    mem_cpu = mem_gpu = work_cpu = work_gpu = None

    if cpu_mem is not None:
        b = cpu_mem[cpu_mem["mapping"] == "block"]
        c = cpu_mem[cpu_mem["mapping"] == "cyclic"]
        if len(b) and len(c):
            mem_cpu = 1.0 - num(c, "effective_bandwidth_gbs").median() / num(b, "effective_bandwidth_gbs").median()

    if gpu_mem is not None:
        g = gpu_mem.copy()
        g["stride"] = num(g, "stride")
        s1 = g[g["stride"] == 1]
        smax = g[g["stride"] == g["stride"].max()]
        if len(s1) and len(smax):
            mem_gpu = 1.0 - num(smax, "effective_bandwidth_gbs").median() / num(s1, "effective_bandwidth_gbs").median()

    if cpu_ctrl is not None:
        grp = cpu_ctrl[cpu_ctrl["distribution"] == "grouped"]
        mix = cpu_ctrl[cpu_ctrl["distribution"] == "mixed"]
        if len(grp) and len(mix):
            work_cpu = 1.0 - num(grp, "throughput_tasks_s").median() / num(mix, "throughput_tasks_s").median()

    if gpu_ctrl is not None:
        grp = gpu_ctrl[gpu_ctrl["distribution"] == "grouped"]
        mix = gpu_ctrl[gpu_ctrl["distribution"] == "mixed"]
        if len(grp) and len(mix):
            work_gpu = 1.0 - num(mix, "throughput_tasks_s").median() / num(grp, "throughput_tasks_s").median()

    rows = []
    if mem_cpu is not None or mem_gpu is not None:
        rows.append(("Memory", mem_cpu, mem_gpu))
    if work_cpu is not None or work_gpu is not None:
        rows.append(("Work/Control", work_cpu, work_gpu))
    if not rows:
        return

    x = range(len(rows))
    width = 0.35
    fig, ax = plt.subplots()
    cvals = [r[1] * 100 if r[1] is not None else 0 for r in rows]
    gvals = [r[2] * 100 if r[2] is not None else 0 for r in rows]
    ax.bar([i - width / 2 for i in x], cvals, width, label="CPU", color="tab:blue")
    ax.bar([i + width / 2 for i in x], gvals, width, label="GPU", color="tab:red")
    ax.set_xticks(list(x))
    ax.set_xticklabels([r[0] for r in rows])
    ax.set_ylabel("Performance loss (%)")
    ax.set_title("Same math, different mapping: CPU vs GPU loss")
    ax.legend()
    save(fig, "exp2_cpu_gpu_loss_comparison.png")


# ---------------------------------------------------------------------------
# Figure 7 (CORE) — thread scaling throughput (log-log)
# ---------------------------------------------------------------------------
def fig7():
    cpu = platform(load("exp3/cpu_scaling.csv"), "CPU")
    gpu = platform(load("exp3/gpu_scaling.csv"), "GPU")
    if cpu is None and gpu is None:
        return
    fig, ax = plt.subplots()
    if cpu is not None:
        c = cpu.sort_values("threads")
        ax.plot(num(c, "threads"), num(c, "throughput_elem_s"), "o-", color="tab:blue", label="CPU")
    if gpu is not None:
        g = gpu.sort_values("threads")
        ax.plot(num(g, "threads"), num(g, "throughput_elem_s"), "s-", color="tab:red", label="GPU")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of threads (log)")
    ax.set_ylabel("Throughput (elements/s, log)")
    ax.set_title("Few strong CPU threads vs massive lightweight GPU threads")
    ax.legend()
    save(fig, "exp3_thread_scaling_throughput.png")


# ---------------------------------------------------------------------------
# Figure 8 — GEMM GFLOPS
# ---------------------------------------------------------------------------
def fig8():
    cpu = platform(load("gemm/cpu_gemm.csv"), "CPU")
    gpu = platform(load("gemm/gpu_gemm.csv"), "GPU")
    if cpu is None and gpu is None:
        return
    fig, ax = plt.subplots()
    if cpu is not None:
        c = cpu.sort_values("matrix_size")
        ax.plot(num(c, "matrix_size"), num(c, "gflops"), "o-", color="tab:blue", label="CPU BLAS")
    if gpu is not None:
        g = gpu.sort_values("matrix_size")
        ax.plot(num(g, "matrix_size"), num(g, "gflops"), "s-", color="tab:red", label="GPU cuBLAS")
    ax.set_xlabel("Matrix size (n)")
    ax.set_ylabel("GFLOPS")
    ax.set_title("GEMM: optimized CPU BLAS vs cuBLAS")
    ax.legend()
    save(fig, "ai_gemm_gflops.png")


# ---------------------------------------------------------------------------
# Figure 9 (extra) — per-worker throughput
# ---------------------------------------------------------------------------
def fig9():
    cpu = platform(load("exp3/cpu_scaling.csv"), "CPU")
    gpu = platform(load("exp3/gpu_scaling.csv"), "GPU")
    if cpu is None and gpu is None:
        return
    fig, ax = plt.subplots()
    if cpu is not None:
        c = cpu.sort_values("threads")
        per = num(c, "throughput_elem_s") / num(c, "threads")
        ax.plot(num(c, "threads"), per, "o-", color="tab:blue", label="CPU per-worker")
    if gpu is not None:
        g = gpu.sort_values("threads")
        per = num(g, "throughput_elem_s") / num(g, "threads")
        ax.plot(num(g, "threads"), per, "s-", color="tab:red", label="GPU per-worker")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of threads (log)")
    ax.set_ylabel("Per-worker throughput (elements/s, log)")
    ax.set_title("Per-worker throughput: strong CPU worker vs lightweight GPU worker")
    ax.legend()
    save(fig, "exp3_per_worker_throughput.png")


def main():
    for f in (fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9):
        try:
            f()
        except Exception as e:  # noqa: BLE001
            print(f"[warn] {f.__name__} failed: {e}")
    print("Done. Figures in", FIG)


if __name__ == "__main__":
    main()
