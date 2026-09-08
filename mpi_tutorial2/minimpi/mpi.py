"""mpi.py — MPI-compatible front-end for the MiniMPI runtime.

Purpose: make the Python calls match the REAL MPI API as closely as the
MiniMPI model allows, so that what students write here transfers directly to
mpi4py / MPI:

    Real MPI                    MiniMPI (this module)
    --------------------------  -------------------------------
    MPI_Init(&argc, &argv)      MPI.Init()
    MPI_COMM_WORLD              MPI.COMM_WORLD
    MPI_Comm_rank(comm, &rank)  comm.Get_rank()
    MPI_Comm_size(comm, &size)  comm.Get_size()
    MPI_Send(buf, cnt, MPI_INT, dest, tag, comm)
                                comm.send(value, dest, tag)
    MPI_Recv(buf, cnt, MPI_INT, source, tag, comm, &status)
                                value = comm.recv(source, tag)
    MPI_Reduce(..., MPI_SUM, root, comm)     comm.reduce(value, op, root)
    MPI_Allreduce(..., MPI_SUM, comm)        comm.allreduce(value, op)
    MPI_Bcast/Scatter/Gather...              (Tutorial 3 territory)

Teaching extras (also via comm.*): naive_reduce / naive_allreduce /
tree_reduce / tree_allreduce / ring_allreduce.

Lifecycle note: each process owns ONE MPI session — worker.py calls
MPI.Init(server=...) once (which connects/joins the world and builds
MPI.COMM_WORLD), runs many collectives, then calls MPI.Finalize() once at
shutdown. Teacher (Rank 0) follows the same Init/COMM_WORLD/Finalize shape.
"""
def _col():
    """Lazy loader for the sibling `collectives` package (works whether
    minimpi is imported as a sub-package or as a top-level package)."""
    try:
        from .. import collectives as C
    except Exception:
        import collectives as C
    return C

# Pseudo datatypes (we encode as raw structs; names mirror MPI)
INT32 = "i32"
FLOAT64 = "f64"
INT = INT32          # commonly used alias
FLOAT = FLOAT64
RAW = "raw"

# Reduction operators
SUM = "sum"
XOR = "xor"

ANY_SOURCE = -1
ANY_TAG = -1

# Session state. COMM_WORLD is created by MPI.Init() (one MPI session per
# process) and invalidated by MPI.Finalize().
COMM_WORLD = None
_alive = False
_session = None     # {"rt": MiniRuntime, "control": control} on a worker


def Init(server=None):
    """Start ONE MPI session for this process.

    Worker: MPI.Init(server) performs the whole bootstrap — create runtime,
    connect to Rank 0, join the world, receive rank/size/peers — and then
    builds MPI.COMM_WORLD. Rank 0: MPI.Init() marks the session start; the
    Rank 0 COMM_WORLD is established once the coordinator is ready.
    """
    global _alive, COMM_WORLD, _session
    if _alive:
        return None
    _alive = True
    if server is not None:                     # worker side
        from .runtime import MiniRuntime
        rt = MiniRuntime(name="worker")
        rt.register_with_teacher(server)       # internal MPI.Init work
        comm = World(rt)
        rt.comm_world = comm
        COMM_WORLD = comm
        _session = {"rt": rt, "control": rt.control, "comm": comm}
    return None


def Finalize():
    """End the MPI session: close control/transport, drop COMM_WORLD."""
    global _alive, COMM_WORLD, _session
    if _session is not None:
        rt = _session["rt"]
        try:
            rt.close()
        except Exception:
            pass
        _session = None
    COMM_WORLD = None
    _alive = False
    return None


class World:
    """MPI-like communicator facade around one rank's runtime.

    All communication still goes through the runtime's transport; this class
    only gives it standard-MPI names, readable default metadata and the
    teaching local timeline (this rank's own clock, relative to Round Start):

        Send Completed / Receive Completed / Operation Completed /
        Local Work Completed   (recorded at the REAL send/recv/op/round end)
    """

    def __init__(self, rt):
        self._rt = rt
        self.rank = rt.comm.rank
        self.size = rt.comm.size
        self.fmt = "i32"
        self.algorithm = ""
        # current logical round / phase used to tag every send/recv event
        self._rnd = 0
        self._phase = ""
        # per-round local timeline (ns on THIS rank's own clock)
        self._round_start = 0.0
        self._send_finish = None
        self._recv_finish = None
        self._op_finish = None
        self._op_kind = ""          # "sum" | "copy"
        self._work_finish = None

    def configure(self, algorithm="", fmt="i32"):
        """Set metadata used to tag this demo's communication events."""
        self.algorithm = algorithm
        self.fmt = fmt

    # -- identity ----------------------------------------------------------
    def Get_rank(self):
        return self.rank

    def Get_size(self):
        return self.size

    # -- point-to-point (MPI-style) ---------------------------------------
    def send(self, value, dest, tag=0):
        """Blocking send (like MPI_Send). Returns bytes sent."""
        from .metrics import now_ns
        n = self._rt.comm.send(value, dest, tag=tag, fmt=self.fmt,
                               algo=self.algorithm, phase=self._phase,
                               rnd=self._rnd)
        # REAL send completion: the frame was handed to the transport
        self._send_finish = now_ns()
        return n

    def recv(self, source=ANY_SOURCE, tag=ANY_TAG, timeout=None):
        """Blocking receive matched by (source, tag)."""
        from .metrics import now_ns
        value = self._rt.comm.recv(source=source, tag=tag, fmt=self.fmt,
                                   algo=self.algorithm, phase=self._phase,
                                   rnd=self._rnd, timeout=timeout)
        # REAL receive completion: the message returned to the algorithm
        self._recv_finish = now_ns()
        return value

    # -- sync point (teaching) --------------------------------------------
    def begin_round(self, rnd, phase=""):
        """Mark the start of a logical round: Round Start on THIS rank's
        clock. Every Completed-At timestamp below is relative to here."""
        from .metrics import now_ns
        self._rnd = rnd
        self._phase = phase
        self._round_start = now_ns()
        self._send_finish = None
        self._recv_finish = None
        self._op_finish = None
        self._op_kind = ""
        self._work_finish = None

    # -- real-operation annotation (single line, teaching only) ------------
    def note_operation_complete(self, kind):
        """Called BY the collective right after a REAL operation finished
        (SUM after combine(), COPY after taking a received value). One tiny
        observability line; never changes the algorithm."""
        if getattr(self._rt, "mode", "performance") != "teaching":
            return
        from .metrics import now_ns
        self._op_finish = now_ns()
        self._op_kind = kind

    # -- timing (teaching local timeline) ----------------------------------
    def _ms(self, ns):
        if ns is None:
            return None
        return (ns - self._round_start) / 1e6

    def local_timings(self):
        """All Completed-At values on THIS rank's own clock, ms from Round
        Start. Fields are None when that event did not happen this round."""
        return {
            "send": self._ms(self._send_finish),
            "recv": self._ms(self._recv_finish),
            "op_kind": self._op_kind,
            "op": self._ms(self._op_finish),
            "work": self._ms(self._work_finish),
        }

    def snapshot_ms(self):
        """Back-compat: (send_ms, work_ms)."""
        t = self.local_timings()
        return t["send"], t["work"]

    def Barrier(self):
        """Start-of-RUN barrier (both modes). Internally an AllReduce-of-1:
        every rank contributes [1], rank 0 reduces to world_size and
        broadcasts it back — a MiniMPI teaching implementation of a barrier,
        not real-MPI's barrier algorithm."""
        from . import barrier as B
        # Rank 0 fires `on_start_gathered` as soon as EVERY rank has arrived
        # (gather complete) but BEFORE it broadcasts the release — that is the
        # true "all ranks ready" instant, so timing never starts late (a
        # released rank may already begin round 1 while rank 0 is still
        # sending releases).
        cb = None
        if self.rank == 0:
            cb = getattr(self._rt, "on_start_gathered", None)
        B.barrier(self._rt.comm, 0, on_root_gathered=cb)

    def sync_round(self, rnd):
        """End of logical round rnd.

        Teaching mode: everyone takes part in a data-plane allreduce-of-1
        barrier here; rank 0 prints the round view and waits for ENTER.
        Performance mode: no-op (round is only a label).

        Local Work Completed is fixed HERE — immediately when this rank
        finished all its real algorithm work for the round, before any UI
        printing or event upload (those run inside the barrier's on_arrived
        window and never enter the timing boundary).
        """
        if getattr(self._rt, "mode", "performance") == "teaching" \
                and self._work_finish is None:
            from .metrics import now_ns
            self._work_finish = now_ns()
        hook = getattr(self._rt, "on_local_work_done", None)
        if hook is not None:
            hook(rnd)
        self._rt.sync_round(rnd)

    # -- collectives (standard + teaching extras) -------------------------
    def reduce(self, value, op=SUM, root=0):
        """MPI_Reduce semantics: root gets the reduced value."""
        return _col().naive_reduce.naive_reduce(self, value, op, root=root)

    def allreduce(self, value, op=SUM):
        """MPI_Allreduce semantics: every rank gets the reduced value."""
        return _col().naive_allreduce.naive_allreduce(self, value, op)

    def naive_reduce(self, value, op=SUM, root=0):
        return _col().naive_reduce.naive_reduce(self, value, op, root=root)

    def naive_allreduce(self, value, op=SUM):
        return _col().naive_allreduce.naive_allreduce(self, value, op)

    def tree_reduce(self, value, op=SUM, root=0):
        return _col().tree_reduce.tree_reduce(self, value, op, root=root)

    def tree_allreduce(self, value, op=SUM):
        return _col().tree_allreduce.tree_allreduce(self, value, op)

    def ring_allreduce(self, value, op=SUM):
        return _col().ring_allreduce.ring_allreduce(self, value, op)
