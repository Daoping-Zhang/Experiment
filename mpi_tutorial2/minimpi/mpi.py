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

Lifecycle note: the classroom runtime performs MPI.Init() before a demo runs
and MPI.Finalize() when it finishes; student programs are `def run(comm, ...)`
functions that receive MPI.COMM_WORLD and only need Get_rank / Get_size /
send / recv / collectives.
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

# Bound while a demo is running (set by minimpi.collectives_dispatch.run).
COMM_WORLD = None
_alive = False


def Init(argv=None):
    """Start of the MPI lifecycle. The runtime calls this before each demo;
    a student program may also call it defensively (no-op if already up)."""
    global _alive
    _alive = True
    return None


def Finalize():
    """End of the MPI lifecycle (called by the runtime after each demo)."""
    global _alive
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
