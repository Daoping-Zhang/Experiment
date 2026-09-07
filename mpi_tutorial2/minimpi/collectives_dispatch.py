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
    """Build one rank's local value.

    Data-size mode (vector_len/data_size > 0):
        value = [base] * N   where base = params["values"][rank] (the
        student-typed initial value). Every element is equal, so the reduced
        result is just the SUM of the students' base values, repeated.
    Legacy scalar mode (ping-pong etc.): a one-element list.
    payload mode (benchmark): raw bytes.
    """
    if params.get("payload"):
        n = int(params["payload"])
        seed = bytes((i * 31 + 7) & 0xFF for i in range(min(n, 4096)))
        return (seed * (n // len(seed) + 1))[:n] if n else b""
    vl = int(params.get("vector_len") or params.get("data_size") or 0)
    if vl > 0:
        values = params.get("values") or {}
        base = values.get(str(rank), values.get(rank, rank + 1))
        base = int(base)
        if params.get("fmt", P.FMT_INT32) == P.FMT_FLOAT64:
            return [float(base)] * vl
        return [base] * vl
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


def run(rt, params, value=None):
    """Run one collective over the MPI-compatible front-end.

    `value` is the rank's OWN typed input ([typed] * data_size is built here);
    when None we fall back to make_value (payload / legacy paths).

    Both teacher rank 0 and student workers call this exact function, so the
    collective data plane is identical for every rank. MPI.Init/Finalize are
    SESSION scoped (worker/teacher call them once) — never per run.
    """
    from . import mpi as M

    comm = M.World(rt)
    comm.configure(algorithm=params["algorithm"],
                   fmt=params.get("fmt", P.FMT_INT32))
    M.COMM_WORLD = comm
    if value is not None:
        vl = int(params.get("vector_len") or params.get("data_size") or 0)
        if vl > 0 and params.get("fmt", P.FMT_INT32) != "raw":
            value = [int(value)] * vl
        elif params.get("fmt", P.FMT_INT32) == "raw":
            value = make_value(params, rank=rt.rank)
    else:
        value = make_value(params, rank=rt.rank)

    fn = getattr(_load_module(params["algorithm"]), params["algorithm"])
    if params["algorithm"] == "ping_pong":
        return fn(comm, value)
    return fn(comm, value, params.get("op", "sum"))
