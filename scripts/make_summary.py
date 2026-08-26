#!/usr/bin/env python3
"""Generate results/summary.md and enriched scaling CSVs from results/*.csv.

Every conclusion is computed from the measured CSV data (never hard-coded).
Missing platforms (e.g. no GPU) are reported explicitly as N/A rather than
invented.
"""
import os

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
RES = os.path.join(ROOT, "results")
SUMMARY = os.path.join(RES, "summary.md")


def load(rel):
    p = os.path.join(RES, rel)
    if not os.path.exists(p) or os.path.getsize(p) == 0:
        return None
    try:
        return pd.read_csv(p)
    except Exception:  # noqa: BLE001
        return None


def platform(df, name):
    if df is None:
        return None
    d = df[df["platform"].astype(str).str.upper() == name.upper()]
    return d if len(d) else None


def num(s):
    return pd.to_numeric(s, errors="coerce")


def med(s):
    s = num(s).dropna()
    return s.median() if len(s) else None


def pct(v):
    return "N/A" if v is None else f"{v * 100:.1f}%"


def flt(v):
    return "N/A" if v is None else f"{v:.3f}"


def ratio(a, b):
    if a is None or b is None or b == 0:
        return None
    return a / b


def enrich_scaling():
    """Write speedup / parallel-efficiency enriched CSVs for Experiment 3."""
    for plat, rel in (("CPU", "exp3/cpu_scaling.csv"), ("GPU", "exp3/gpu_scaling.csv")):
        df = load(rel)
        if df is None:
            continue
        d = platform(df, plat)
        if d is None or len(d) == 0:
            continue
        d = d.copy()
        d["threads"] = num(d["threads"])
        d = d.sort_values("threads")
        t1 = med(d[d["threads"] == 1]["throughput_elem_s"])
        out = d.copy()
        if t1 is not None:
            out["speedup"] = num(out["throughput_elem_s"]) / t1
            out["parallel_efficiency"] = out["speedup"] / num(out["threads"])
        else:
            out["speedup"] = None
            out["parallel_efficiency"] = None
        dest = os.path.join(RES, "exp3", f"{plat.lower()}_scaling_with_speedup.csv")
        out.to_csv(dest, index=False)
        print(f"[summary] wrote {dest}")


def build():
    exp1_dep = load("exp1/dependent.csv")
    exp1_ind = load("exp1/independent.csv")
    exp1_br = load("exp1/branch.csv")
    cpu_mem = platform(load("exp2/cpu_memory_mapping.csv"), "CPU")
    gpu_mem = platform(load("exp2/gpu_memory_mapping.csv"), "GPU")
    cpu_ctrl = platform(load("exp2/cpu_work_mapping.csv"), "CPU")
    gpu_ctrl = platform(load("exp2/gpu_work_mapping.csv"), "GPU")
    cpu_scale = platform(load("exp3/cpu_scaling.csv"), "CPU")
    gpu_scale = platform(load("exp3/gpu_scaling.csv"), "GPU")
    cpu_gemm = platform(load("gemm/cpu_gemm.csv"), "CPU")
    gpu_gemm = platform(load("gemm/gpu_gemm.csv"), "GPU")

    cpu_dep = platform(exp1_dep, "CPU")
    gpu_dep = platform(exp1_dep, "GPU")
    cpu_ind = platform(exp1_ind, "CPU")
    gpu_ind = platform(exp1_ind, "GPU")
    cpu_br = platform(exp1_br, "CPU")
    gpu_br = platform(exp1_br, "GPU")

    lines = []
    lines.append("# Lecture 01 Tutorial — Measurement Summary\n")
    lines.append("_All numbers below are computed from the measured CSV files; "
                 "N/A means the platform/data was not measured (e.g. no GPU toolchain or no perf)._")
    lines.append("")

    # ---------------- Q1 ----------------
    lines.append("## Q1 — Single thread: CPU vs GPU\n")
    dep_lat_cpu = med(cpu_dep["latency_ms"]) if cpu_dep is not None else None
    dep_lat_gpu = med(gpu_dep["latency_ms"]) if gpu_dep is not None else None
    lines.append("- Dependent workload CPU/GPU latency ratio: "
                 f"**{flt(ratio(dep_lat_cpu, dep_lat_gpu))}** "
                 f"(CPU {flt(dep_lat_cpu)} ms, GPU kernel {flt(dep_lat_gpu)} ms)")
    if cpu_ind is not None:
        c1 = cpu_ind[cpu_ind["variant"] == "chains=1"]
        c8 = cpu_ind[cpu_ind["variant"] == "chains=8"]
        g1 = gpu_ind[gpu_ind["variant"] == "chains=1"] if gpu_ind is not None else None
        g8 = gpu_ind[gpu_ind["variant"] == "chains=8"] if gpu_ind is not None else None
        lines.append("- Independent workload CPU/GPU throughput (GFLOPS) ratio @1 chain: "
                     f"**{flt(ratio(med(c1['gflops']) if len(c1) else None, med(g1['gflops']) if g1 is not None and len(g1) else None))}**")
        lines.append("- Independent workload CPU/GPU throughput (GFLOPS) ratio @8 chains: "
                     f"**{flt(ratio(med(c8['gflops']) if len(c8) else None, med(g8['gflops']) if g8 is not None and len(g8) else None))}**")
    lines.append("")

    # ---------------- Q2 ----------------
    lines.append("## Q2 — CPU independent chains 1 -> 2 -> 4 -> 8\n")
    if cpu_ind is not None:
        lines.append("| chains | IPC | GFLOPS | throughput (iter/s) |")
        lines.append("|---|---|---|---|")
        for c in ("chains=1", "chains=2", "chains=4", "chains=8"):
            r = cpu_ind[cpu_ind["variant"] == c]
            if len(r) == 0:
                continue
            ipc = med(r["ipc"])
            gf = med(r["gflops"])
            tp = med(r["throughput_ops_s"])
            lines.append(f"| {c.replace('chains=', '')} | {flt(ipc)} | {flt(gf)} | {flt(tp)} |")
        lines.append("")
    else:
        lines.append("N/A\n")

    # ---------------- Q3 ----------------
    lines.append("## Q3 — Branch prediction\n")
    if cpu_br is not None:
        pred = cpu_br[cpu_br["variant"] == "predictable"]
        rnd = cpu_br[cpu_br["variant"] == "random"]
        m_pred = med(pred["branch_miss_rate"]) if len(pred) else None
        m_rnd = med(rnd["branch_miss_rate"]) if len(rnd) else None
        tp_pred = med(pred["throughput_ops_s"]) if len(pred) else None
        tp_rnd = med(rnd["throughput_ops_s"]) if len(rnd) else None
        loss = 1 - ratio(tp_rnd, tp_pred) if (tp_pred and tp_rnd) else None
        lines.append("- CPU branch miss rate: "
                     f"{pct(m_pred) if m_pred is not None else 'N/A'} -> {pct(m_rnd) if m_rnd is not None else 'N/A'}")
        lines.append(f"- CPU throughput loss (predictable -> random): **{pct(loss)}**")
    if gpu_br is not None:
        pred = gpu_br[gpu_br["variant"] == "predictable"]
        rnd = gpu_br[gpu_br["variant"] == "random"]
        tp_pred = med(pred["throughput_ops_s"]) if len(pred) else None
        tp_rnd = med(rnd["throughput_ops_s"]) if len(rnd) else None
        loss = 1 - ratio(tp_rnd, tp_pred) if (tp_pred and tp_rnd) else None
        lines.append(f"- GPU single-thread throughput loss (predictable -> random): **{pct(loss)}**")
    lines.append("")

    # ---------------- Q4 ----------------
    lines.append("## Q4 — Mapping (same math, different mapping)\n")
    mem_loss_cpu = mem_loss_gpu = None
    if cpu_mem is not None:
        b = cpu_mem[cpu_mem["mapping"] == "block"]
        c = cpu_mem[cpu_mem["mapping"] == "cyclic"]
        if len(b) and len(c):
            mem_loss_cpu = 1 - ratio(med(c["effective_bandwidth_gbs"]), med(b["effective_bandwidth_gbs"]))
            lines.append("- CPU memory mapping (block -> cyclic) throughput loss: "
                         f"**{pct(mem_loss_cpu)}**")
            for col in ("cache_references", "cache_misses", "llc_loads", "llc_load_misses"):
                cb = med(b[col]) if len(b) else None
                cc = med(c[col]) if len(c) else None
                if cb is not None and cc is not None:
                    lines.append(f"  - {col}: block {flt(cb)} -> cyclic {flt(cc)}")
    if gpu_mem is not None:
        g = gpu_mem.copy()
        g["stride"] = num(g["stride"])
        s1 = g[g["stride"] == 1]
        smax = g[g["stride"] == g["stride"].max()]
        if len(s1) and len(smax):
            mem_loss_gpu = 1 - ratio(med(smax["effective_bandwidth_gbs"]), med(s1["effective_bandwidth_gbs"]))
            lines.append(f"- GPU memory mapping (stride 1 -> {int(g['stride'].max())}) throughput loss: "
                         f"**{pct(mem_loss_gpu)}** (see results/exp2/ncu/ for transaction metrics)")
    work_loss_cpu = work_loss_gpu = None
    if cpu_ctrl is not None:
        grp = cpu_ctrl[cpu_ctrl["distribution"] == "grouped"]
        mix = cpu_ctrl[cpu_ctrl["distribution"] == "mixed"]
        if len(grp) and len(mix):
            work_loss_cpu = 1 - ratio(med(grp["throughput_tasks_s"]), med(mix["throughput_tasks_s"]))
            lines.append(f"- CPU work/control (balanced -> grouped) throughput loss: **{pct(work_loss_cpu)}**")
            lines.append(f"  - load imbalance ratio: mixed {flt(med(mix['load_imbalance_ratio']))} vs "
                         f"grouped {flt(med(grp['load_imbalance_ratio']))}")
    if gpu_ctrl is not None:
        grp = gpu_ctrl[gpu_ctrl["distribution"] == "grouped"]
        mix = gpu_ctrl[gpu_ctrl["distribution"] == "mixed"]
        if len(grp) and len(mix):
            work_loss_gpu = 1 - ratio(med(mix["throughput_tasks_s"]), med(grp["throughput_tasks_s"]))
            lines.append(f"- GPU work/control (uniform -> divergent) throughput loss: **{pct(work_loss_gpu)}** "
                         "(warp divergence; see results/exp2/ncu/ for branch/lane metrics)")
    lines.append("")

    # ---------------- Q5 ----------------
    lines.append("## Q5 — Scaling\n")
    if cpu_scale is not None:
        c = cpu_scale.sort_values("threads")
        peak = med(c["throughput_elem_s"])
        peak_t = c.loc[num(c["throughput_elem_s"]).idxmax(), "threads"] if len(c) else None
        lines.append(f"- CPU maximum measured threads = {int(num(c['threads']).max())}")
        lines.append(f"- CPU peak throughput = {flt(peak)} elem/s (at threads={peak_t})")
    if gpu_scale is not None:
        g = gpu_scale.sort_values("threads")
        peak = med(g["throughput_elem_s"])
        target = 0.9 * peak
        t90 = None
        for _, r in g.iterrows():
            if num(pd.Series([r["throughput_elem_s"]])).iloc[0] >= target:
                t90 = r["threads"]
                break
        lines.append(f"- GPU peak throughput = {flt(peak)} elem/s")
        lines.append(f"- GPU thread count at ~90% peak throughput = {int(t90) if t90 is not None else 'N/A'}")
    lines.append("")

    # ---------------- Q6 ----------------
    lines.append("## Q6 — GEMM crossover\n")
    if cpu_gemm is not None and gpu_gemm is not None:
        c = cpu_gemm.sort_values("matrix_size")
        g = gpu_gemm.sort_values("matrix_size")
        crossover = None
        for _, r in c.iterrows():
            n = r["matrix_size"]
            gr = g[g["matrix_size"] == n]
            if len(gr) and med(gr["gflops"]) > med(pd.Series([r["gflops"]])):
                crossover = n
                break
        if crossover is None:
            lines.append("No crossover observed in tested range.")
        else:
            lines.append(f"CPU/GPU crossover matrix size = **{int(crossover)}**")
    else:
        lines.append("N/A (need both CPU and GPU GEMM results)")

    with open(SUMMARY, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[summary] wrote {SUMMARY}")


if __name__ == "__main__":
    enrich_scaling()
    build()
