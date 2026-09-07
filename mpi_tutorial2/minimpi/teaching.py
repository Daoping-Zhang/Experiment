"""teaching.py — Teaching-View semantics for MiniMPI Tutorial 2.

Pure presentation-layer logic shared by teacher (rank 0) and workers:

    * phase / round mapping   : what logical phase is round rnd of algorithm X
                                at world size P, and which op it performs
    * ring chunk schedule     : which chunk index a rank sends / receives in a
                                ring round (mirrors collectives/ring_allreduce)
    * vector preview          : bounded display of a vector
    * per-round local semantics: BEFORE / SEND / RECEIVE / OPERATION / AFTER,
                                built from the rank's REAL events + its REAL
                                initial data — never a UI-side guess

It contains NO sockets / threads / protocol: rendering only. The collective
algorithm files are never touched for the sake of the UI.
"""
from dataclasses import dataclass, field
from typing import Any, List, Optional


# --------------------------------------------------------------------------
# Phase / round mapping  (presentation layer; algorithm files stay untouched)
# --------------------------------------------------------------------------
def pretty_algorithm(alg):
    return {
        "ping_pong": "Point-to-Point (Ping-Pong)",
        "naive_reduce": "Naive Reduce",
        "naive_allreduce": "Naive AllReduce",
        "tree_reduce": "Tree Reduce",
        "tree_allreduce": "Tree AllReduce",
        "ring_allreduce": "Ring AllReduce",
    }.get(alg, alg)


def total_rounds(algorithm, P):
    """How many teaching rounds one run of `algorithm` has (per rank)."""
    if algorithm == "tree_reduce":
        return P.bit_length() - 1
    if algorithm == "tree_allreduce":
        return 2 * (P.bit_length() - 1)
    if algorithm == "ring_allreduce":
        return 2 * (P - 1) if P >= 2 else 0
    if algorithm == "naive_allreduce":
        return 2
    return 1          # ping_pong, naive_reduce


def phase_of(algorithm, P, rnd):
    """Return (phase_label, op) for round rnd of algorithm at world size P.

    op is the semantic of a RECEIVE in that phase:
        "sum"  -> combine into local data (Reduce)
        "copy" -> take the received data as-is (Broadcast / AllGather)
        "none" -> no data-changing receive in this phase
    """
    if algorithm == "ping_pong":
        return "Phase: Point-to-Point", "none"
    if algorithm == "naive_reduce":
        return "Phase: Reduce to Root", "sum"
    if algorithm == "naive_allreduce":
        return ("Phase 1: Reduce to Root" if rnd == 1
                else "Phase 2: Broadcast Result"), ("sum" if rnd == 1 else "copy")
    L = P.bit_length() - 1
    if algorithm == "tree_reduce":
        return "Phase: Tree Reduce", "sum"
    if algorithm == "tree_allreduce":
        if rnd <= L:
            return "Phase 1: Reduce", "sum"
        return "Phase 2: Broadcast", "copy"
    if algorithm == "ring_allreduce":
        if rnd <= P - 1:
            return "Phase 1: Reduce-Scatter", "sum"
        return "Phase 2: AllGather", "copy"
    return "Phase: -", "none"


# --------------------------------------------------------------------------
# Ring chunk schedule  (identical arithmetic to collectives/ring_allreduce.py)
# --------------------------------------------------------------------------
def ring_chunk_index(rank, P, rnd, side):
    """Which chunk index this rank sends/receives in a ring round.

    Reduce-Scatter step s = rnd-1 (1-based rnd):
        send_idx = (rank - s) % P
        recv_idx = (send_idx - 1) % P
    AllGather step s = rnd-P, owned = (rank+1) % P:
        send:  chunk owned - s   (module sends cur starting at `owned`)
        recv:  chunk rank - s    (content identity of what prev sent us)
    """
    if rnd <= P - 1:
        s = rnd - 1
        if side == "send":
            return (rank - s) % P
        return (rank - s - 1) % P
    s = rnd - P
    owned = (rank + 1) % P
    if side == "send":
        return (owned - s) % P
    return (rank - s) % P


# --------------------------------------------------------------------------
# Vector preview
# --------------------------------------------------------------------------
def preview_vector(value, max_elements=8, with_count=True):
    """[1, 1, 1, 1] or [1, 1, ..., 1, 1] (1024 elements)."""
    if value is None:
        return "None"
    if isinstance(value, (bytes, bytearray)):
        return "(raw bytes, %d B)" % len(value)
    vals = list(value)
    n = len(vals)
    if n == 0:
        return "[]"
    if n <= max_elements:
        body = ", ".join(str(v) for v in vals)
    else:
        head = ", ".join(str(v) for v in vals[:max_elements // 2])
        tail = ", ".join(str(v) for v in vals[-(max_elements // 2):])
        body = "%s, ..., %s" % (head, tail)
    if with_count and n > max_elements:
        return "[%s]\n(%d elements)" % (body, n)
    return "[%s]" % body


# --------------------------------------------------------------------------
# Per-round local semantics (BEFORE / SEND / RECEIVE / OPERATION / AFTER)
# --------------------------------------------------------------------------
@dataclass
class MsgView:
    peer: int                 # destination (send) or source (recv)
    data: Any                 # the real payload this rank sent / received
    chunk: Optional[int] = None
    op: str = ""


@dataclass
class RoundView:
    algorithm: str = ""
    phase_label: str = ""
    rnd: int = 0
    total: int = 0
    rank: int = 0
    role: str = ""
    before: Any = None
    sends: List[MsgView] = field(default_factory=list)
    receives: List[MsgView] = field(default_factory=list)
    operation: str = ""       # short op summary, e.g. "SUM"
    op_detail: str = ""       # multi-line real-data arithmetic (optional)
    after: Any = None
    after_note: str = ""


def _role(sends, receives, op):
    parts = []
    if sends:
        parts.append("Sender")
    if receives:
        parts.append("Receiver")
    if op == "sum" and receives:
        parts.append("Reduce")
    elif op == "copy" and receives:
        parts.append("Copy")
    return "Idle" if not parts else " + ".join(parts)


def _elem_sum(a, b):
    return [x + y for x, y in zip(a, b)]


class RoundCtx:
    """Per-run semantic state of ONE rank. Mutated by describe_round().

    Fields are algorithm-specific:
        vector-algs : .value        (this rank's current full vector)
        ring        : .chunks       (P chunk lists, updated during RS)
    """
    def __init__(self, algorithm, P, rank, initial_value):
        self.algorithm = algorithm
        self.P = P
        self.rank = rank
        if algorithm == "ring_allreduce" and P >= 2:
            n = len(list(initial_value))
            cl = n // P
            vals = list(initial_value)
            self.chunks = [vals[i * cl:(i + 1) * cl] for i in range(P)]
            self.value = None
            self.gathered = 0      # chunks copied so far during AllGather
        else:
            self.value = list(initial_value)
            self.chunks = None

    def current_value(self):
        if self.chunks is not None:
            out = []
            for c in self.chunks:
                out.extend(c)
            return out
        return self.value


def describe_round(ctx, rnd, events):
    """Build a RoundView for one round from REAL events of this rank.

    `events`: this rank's CommunicationEvent objects for `rnd`, algorithm
    kind only (barrier traffic excluded). Payload contents come from the
    actual send (value_before) / recv (value_after) records.
    """
    alg = ctx.algorithm
    P = ctx.P
    rank = ctx.rank
    phase_label, op = phase_of(alg, P, rnd)
    total = total_rounds(alg, P)

    sends = []
    recvs = []
    for e in sorted(events, key=lambda e: e.ts_start_ns):
        data = e.value_before if e.side == "send" else e.value_after
        if e.side == "send":
            sends.append(MsgView(peer=e.destination, data=data,
                                 chunk=_chunk_of(alg, P, rank, rnd, "send")))
        else:
            recvs.append(MsgView(peer=e.source, data=data,
                                 chunk=_chunk_of(alg, P, rank, rnd, "recv"),
                                 op=op))
    before = ctx.current_value()
    before_disp = list(before) if before is not None else None

    # ---- AFTER -----------------------------------------------------------
    after = None
    after_note = ""
    op_detail = ""
    if ctx.chunks is not None:                      # ring
        pre = [list(c) for c in ctx.chunks]         # chunk values BEFORE this
        for m in recvs:                             # round's updates (for the
            if op == "sum":                         # real-data op display)
                ci = m.chunk
                ctx.chunks[ci] = _elem_sum(ctx.chunks[ci], list(m.data))
            else:
                ctx.gathered += 1
        after = ctx.current_value()
        if op == "sum":
            for m in recvs:
                op_detail += ("Reduce Chunk %d:\n  %s + %s = %s\n"
                              % (m.chunk, preview_vector(pre[m.chunk], 8, False),
                                 preview_vector(m.data, 8, False),
                                 preview_vector(ctx.chunks[m.chunk], 8, False)))
        else:
            after_note = ("gathering reduced chunks: %d / %d collected so far"
                          % (ctx.gathered, P))
    else:                                           # whole-vector algorithms
        acc = list(before_disp)
        chain = []
        for m in recvs:
            if op == "sum":
                nxt = _elem_sum(acc, list(m.data))
                chain.append((list(acc), list(m.data), list(nxt)))
                acc = nxt
            else:                                   # copy: take received data
                acc = list(m.data)
        ctx.value = acc                             # next round's BEFORE
        after = acc
        if op == "sum":
            for (a, b, c) in chain:
                op_detail += ("%s + %s = %s\n"
                              % (preview_vector(a, 8, False),
                                 preview_vector(b, 8, False),
                                 preview_vector(c, 8, False)))
        elif op == "copy" and recvs:
            op_detail = ("%s (COPY)\n" % preview_vector(acc, 8, False))
        # no receives -> op_detail stays "" (the OPERATION line already shows
        # "None")

    role = _role(sends, recvs, op)
    op_short = "None"
    if recvs:
        op_short = "SUM" if op == "sum" else "COPY"
    return RoundView(algorithm=alg, phase_label=phase_label, rnd=rnd,
                     total=total, rank=rank, role=role, before=before_disp,
                     sends=sends, receives=recvs, operation=op_short,
                     op_detail=op_detail, after=after, after_note=after_note)


def _chunk_of(alg, P, rank, rnd, side):
    if alg == "ring_allreduce":
        return ring_chunk_index(rank, P, rnd, side)
    return None


# --------------------------------------------------------------------------
# Text renderers (student terminal / teacher rank-0 local view share these)
# --------------------------------------------------------------------------
def _msg_line(title, msgs, direction):
    lines = ["\n%s" % title]
    if not msgs:
        lines.append("None")
        return lines
    for m in msgs:
        if direction == "send":
            head = "To Rank %d" % m.peer
        else:
            head = "From Rank %d" % m.peer
        if m.chunk is not None:
            head += "  [Chunk %d]" % m.chunk
        lines.append(head)
        lines.append(preview_vector(m.data))
    return lines


def local_view_text(view):
    """Render one rank's per-round local view (student OR teacher rank 0)."""
    L = []
    bar = "=" * 48
    L.append(bar)
    L.append(pretty_algorithm(view.algorithm))
    L.append(view.phase_label)
    L.append("Round %d / %d" % (view.rnd, view.total))
    L.append("Rank %d" % view.rank)
    L.append(bar)
    L.append("\nMy Role: %s" % view.role)
    L.append("\nBEFORE")
    L.append("My Data:")
    L.append(preview_vector(view.before))
    L.extend(_msg_line("SEND", view.sends, "send"))
    L.extend(_msg_line("RECEIVE", view.receives, "recv"))
    L.append("\nOPERATION")
    L.append(view.operation if view.operation else "None")
    if view.op_detail:
        for ln in view.op_detail.splitlines():
            L.append("  " + ln)
    L.append("\nAFTER")
    L.append("My Data:")
    L.append(preview_vector(view.after))
    if view.after_note:
        L.append("(%s)" % view.after_note)
    return "\n".join(L)
