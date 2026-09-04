"""collectives_dispatch — teacher & worker call this with identical params.

params keys:
  algorithm : ping_pong|naive_reduce|naive_allreduce|tree_reduce|
              tree_allreduce|ring_allreduce
  mode      : teaching|performance
  op        : sum|xor
  fmt       : i32|f64|raw
  value spec: n_value (scalar) | vector_len (int vector) | payload (bytes)
"""
from . import protocol as P


def make_value(params, rank=0):
    """Build one rank's local value (deterministic, same everywhere)."""
    if params.get("payload"):
        n = int(params["payload"])
        seed = bytes((i * 31 + 7) & 0xFF for i in range(min(n, 4096)))
        return (seed * (n // len(seed) + 1))[:n] if n else b""
    if params.get("vector_len"):
        seed = params.get("seed", 1)
        return [seed + rank + j for j in range(int(params["vector_len"]))]
    if params.get("fmt", P.FMT_INT32) == P.FMT_FLOAT64:
        return [float(params.get("n_value", 1))]
    return [int(params.get("n_value", 7))]


def _load_module(algo):
    """Load a collective module when minimpi is used either as a sub-package
    (in-tree) or as a top-level package (running teacher.py / worker.py)."""
    try:
        from .. import collectives as C   # in-tree layout
        return getattr(C, algo)
    except Exception:
        import importlib
        return importlib.import_module("collectives." + algo)


def run(rt, params):
    value = make_value(params, rank=rt.rank)
    fmt = params.get("fmt", P.FMT_INT32)
    op = params.get("op", "sum")
    algo = params["algorithm"]

    fn = getattr(_load_module(algo), algo)
    return fn(rt, value, op, fmt) if algo != "ping_pong" else fn(rt, value)
