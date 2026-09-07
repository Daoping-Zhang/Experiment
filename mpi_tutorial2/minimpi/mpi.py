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
    only gives it standard-MPI names and readable default metadata.
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
        return self._rt.comm.send(value, dest, tag=tag, fmt=self.fmt,
                                  algo=self.algorithm, phase=self._phase,
                                  rnd=self._rnd)

    def recv(self, source=ANY_SOURCE, tag=ANY_TAG, timeout=None):
        """Blocking receive matched by (source, tag)."""
        return self._rt.comm.recv(source=source, tag=tag, fmt=self.fmt,
                                  algo=self.algorithm, phase=self._phase,
                                  rnd=self._rnd, timeout=timeout)

    # -- sync point (teaching) --------------------------------------------
    def begin_round(self, rnd, phase=""):
        """Mark the start of a logical round (labels events; see README)."""
        self._rnd = rnd
        self._phase = phase

    def sync_round(self, rnd):
        """End of logical round rnd.

        Teaching mode: everyone takes part in a data-plane allreduce-of-1
        barrier here; rank 0 prints the round view and waits for ENTER.
        Performance mode: no-op (round is only a label).
        """
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
