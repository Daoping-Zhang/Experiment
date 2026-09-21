#!/usr/bin/env python3
"""worker.py — one student rank: a real MPI-style program.

Read the main flow top-down:

    MPI.Init(...)                 # 1  one MPI session for this process
    comm = MPI.COMM_WORLD         # 2  the world communicator
    rank = comm.Get_rank()        # 3  my identity
    size = comm.Get_size()

    while True:
        run = classroom.wait_for_run()    # 4  wait for the teacher's next RUN
        if run is None:                   #    (None == session shutdown)
            break
        value = read_one_integer(...)     # 5  I type one integer
        data = [value] * run.data_size    # 6  my local data vector
        comm.Barrier()                    # 7  Start Barrier: wait for everyone
        result = run_one_collective(...)  # 8  the real collective
        show_result(result)               # 9  my result

    MPI.Finalize()                # 10 end the MPI session

The classroom control plane (join / RUN / SHUTDOWN) is teaching
infrastructure, not fake MPI API — it stays inside ClassroomWorker.
"""
import argparse
import base64
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from minimpi import mpi as M               # noqa: E402
from minimpi import protocol as P          # noqa: E402
from minimpi import teaching as T          # noqa: E402
from minimpi.classroom_worker import ClassroomWorker  # noqa: E402
from minimpi.collectives_dispatch import make_benchmark_payload  # noqa: E402
from minimpi.runtime import JoinRefused, VersionMismatch  # noqa: E402


def _join_budget():
    """(attempts, delay) for the join retry loop. MINIMPI_JOIN_ATTEMPTS /
    MINIMPI_JOIN_RETRY_S exist for tests and automation (like the other
    MINIMPI_* hooks); the classroom uses the defaults."""
    try:
        attempts = int(os.environ.get("MINIMPI_JOIN_ATTEMPTS", "") or 60)
    except ValueError:
        attempts = 60
    try:
        delay = float(os.environ.get("MINIMPI_JOIN_RETRY_S", "") or 3.0)
    except ValueError:
        delay = 3.0
    return max(1, attempts), max(0.0, delay)


def join_world(MPI, server, attempts=None, delay=None):
    """Join the class, retrying while the teacher is busy.

    A student who starts the worker in the middle of a collective is refused
    ("a collective run is in progress"). Their copy is fine, so instead of
    exiting we wait and try again — by the time that run finishes they join as
    a normal latecomer and take part in the next RUN."""
    if attempts is None or delay is None:
        a, d = _join_budget()
        attempts = attempts or a
        delay = delay if delay is not None else d
    for i in range(1, attempts + 1):
        try:
            MPI.Init(server=server)
            if i > 1:
                print("\n[JOIN] reconnected on attempt %d." % i)
            return
        except (JoinRefused, OSError) as e:
            if isinstance(e, OSError):
                # the teacher is not listening (yet), or the network dropped:
                # a student must see words, not a traceback
                why = ("cannot reach the teacher at %s (%s) — is the class "
                       "running?" % (server, e))
            else:
                why = str(e)
            if i >= attempts:
                print("\n[JOIN REFUSED] %s" % why)
                print("Giving up after %d attempts — tell the teacher." % i)
                raise JoinRefused(why, getattr(e, "code", "unreachable"))
            if i == 1 or i % 10 == 0:
                print("\n[JOIN REFUSED] %s" % why)
                print("[JOIN] retrying in %.1f s (attempt %d/%d); Ctrl-C "
                      "stops." % (delay, i, attempts))
            else:
                print("[JOIN] still waiting for the teacher (attempt %d/%d)..."
                      % (i, attempts))
            time.sleep(delay)


def main():
    args = parse_args()

    MPI = M                                 # MPI.Init starts ONE session:
    print("MiniMPI Worker version %s (protocol %d)"
          % (P.MINIMPI_VERSION, P.PROTOCOL_VERSION))
    try:
        join_world(MPI, args.server)        # connects/joins + COMM_WORLD
    except VersionMismatch as e:
        print("\n[VERSION MISMATCH] %s" % e)
        print("Please update this student copy (git pull / re-download) "
              "and start the worker again.")
        sys.exit(2)
    except JoinRefused as e:
        # The copy is fine — the teacher stayed busy, or the world is full.
        print("\n[JOIN REFUSED] %s" % e)
        print("Start the worker again in a moment (or tell the teacher to "
              "raise --size).")
        sys.exit(3)
    try:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        print_banner(rank, size)
        run_classroom(comm)
    finally:
        MPI.Finalize()                      # every exit path ends the session


_RUNS = [0]           # how many RUNs this worker has entered (test hook)


def run_classroom(comm):
    """The student's run loop: wait -> input -> data -> barrier -> collective."""
    classroom = ClassroomWorker.attach_session()
    try:
        while True:
            run = classroom.wait_for_run()       # blocks until the teacher
            if run is None:                      # sends a RUN (None == end)
                print("[shutdown]")
                return

            # Teacher control messages that are NOT a collective run: roster
            # changes (somebody joined/left) and aborts.
            if run.kind == "roster":
                classroom.apply_roster(run.params.get("rank"),
                                       run.params.get("size"),
                                       run.params.get("peers", {}))
                print("\n[ROSTER] You are now Rank %s / %s  (world updated)"
                      % (run.params.get("rank"), run.params.get("size")))
                continue

            if run.kind == "kicked":
                # Teacher removed this rank (a stuck/duplicate student, or a
                # machine that is only half alive). Say so plainly and stop.
                print("\n[KICKED] %s" % run.params.get("why"))
                print("[KICKED] this worker left the class. Restart it to "
                      "join again.")
                return

            if run.kind == "abort":
                print("\n[ABORT] %s"
                      % (run.params.get("why") or "this RUN was cancelled"))
                classroom.clear_abort()
                continue

            # Performance Benchmark session setup: ONE input, all later cases
            # reuse that value — no per-case prompts, no zero Data Size.
            if run.kind == "benchmark_done":
                sm = run.params.get("summary")
                if sm:
                    print("\n" + sm)     # teacher's results, shown on this rank too
                print("\nBenchmark Complete.\n\nWaiting for Rank 0...")
                classroom.reset_benchmark()
                continue

            if run.kind == "benchmark_setup":
                print("\n========================================\n"
                      "Performance Benchmark Setup\n"
                      "========================================")
                print("\nThis value will be reused for all benchmark cases.")
                classroom.set_benchmark_value(read_one_integer(comm.Get_rank()))
                print("\nBenchmark value: %s\n" % classroom.benchmark_value())
                continue

            if run.kind == "benchmark_case":
                value = classroom.benchmark_value()
                if value is None:                # safety: never re-prompt
                    value = comm.Get_rank() + 1
                if not classroom.benchmark_announced():
                    print("\nPerformance Benchmark Running...\n\n"
                          "Local benchmark value: %s\n\nPlease wait." % value)
                    classroom.mark_benchmark_announced()
            else:
                value = read_one_integer(comm.Get_rank())

            if run.kind == "benchmark_case":
                # the value entered once at setup now REALLY builds this
                # rank's raw payload (byte-level benchmarking, xor combine)
                data = make_benchmark_payload(value, run.payload)
            elif run.payload:
                data = None
            else:
                data = [value] * run.data_size

            if run.kind != "benchmark_case":
                if run.data_size > 0:
                    print("\nAlgorithm: %s\nData Size: %d elements\n"
                          % (run.algorithm, run.data_size))
                else:
                    print("\nAlgorithm: %s\nLocal Payload per Rank: %d B\n"
                          % (run.algorithm, run.payload))
                if run.mode == "teaching":
                    print("Local data ready.\n\nEntering MPI Barrier...\n"
                          "Waiting for all ranks...")

            # A new RUN starts from a clean data plane: frames left over by a
            # previous aborted RUN would otherwise be matched by this run's
            # receives and corrupt the result.
            classroom.drop_stale_frames()

            try:
                comm.Barrier()           # Start Barrier (both modes): the
                                         # collective only starts when every
                                         # rank is ready
                result = run_one_collective(classroom, run, data)
            except Exception as e:  # noqa: BLE001
                # A RUN that could not start, or was aborted (a rank left,
                # the teacher cancelled), must never kill the worker: report
                # it and go back to waiting for the teacher's next command.
                why = classroom.runtime.abort_reason or str(e)
                print("\n[ABORT] this RUN ended early: %s" % why)
                classroom.clear_abort()
                continue
            if classroom.runtime.aborted:
                print("\n[ABORT] this RUN ended early: %s"
                      % classroom.runtime.abort_reason)
            classroom.clear_abort()
            if run.kind != "benchmark_case":
                show_result(comm.Get_rank(), run, result)
    finally:
        classroom.close()


def run_one_collective(classroom, run, data):
    """Run one RUN and ALWAYS report the outcome to the teacher.

    Anything that goes wrong — even while merely PREPARING the run, before the
    collective starts — is reported as a C_DONE error: a rank that stays silent
    would otherwise leave the whole class waiting until the stall watchdog.
    """
    rt = classroom.runtime
    control = rt.control
    try:
        return _run_one_collective(classroom, run, data)
    except Exception as e:  # noqa: BLE001
        msg = str(e)
        if getattr(rt, "aborted", False):
            msg = "ABORTED: %s (%s)" % (rt.abort_reason, msg)
        try:
            control.send({"t": P.C_DONE, "rank": rt.rank, "error": msg,
                          "final": None, "events": 0})
        except OSError:
            pass          # teacher is gone: the control reader handles that
        return None


def _run_one_collective(classroom, run, data):
    """Prepare one RUN and execute it on the session runtime.

    Returns the collective's real result (or None on error, which is reported
    to the teacher by the wrapper above). This is where teaching hooks are
    installed — the algorithm files stay untouched.
    """
    rt = classroom.runtime
    control = rt.control
    params = run.params

    # test-only hook (like MINIMPI_FAKE_VERSION): fail THIS run on one rank so
    # the classroom's error handling can be exercised end to end
    _RUNS[0] += 1
    fail_at = os.environ.get("MINIMPI_FAIL_RUN")
    if fail_at and _RUNS[0] == int(fail_at):
        raise RuntimeError("forced failure (test hook MINIMPI_FAIL_RUN=%s)"
                           % fail_at)

    # The welcome only contained the peers joined at that moment; the RUN
    # command carries the full peer table — apply it first.
    peers = params.get("peers")
    if peers:
        table = {int(k): (v["host"], int(v["port"]))
                 for k, v in peers.items() if int(k) != rt.rank}
        rt.transport.set_peers(rt.rank, table)

    rt.mode = run.mode
    rt.run_meta = {"algorithm": run.algorithm, "data_size": run.data_size,
                   "vector_len": params.get("vector_len", 0),
                   "fmt": params.get("fmt", "i32")}

    if run.mode == "teaching":
        from minimpi import barrier as BarrierMod
        rt.show_ui = True
        rt._report = None
        rt._on_round = None

        # Per-run local-semantics state (real events + real initial data).
        if run.data_size > 0 and run.fmt == "i32":
            rt._local_ctx = T.RoundCtx(run.algorithm, rt.size, rt.rank, data)
        else:
            rt._local_ctx = None

        def round_barrier(rnd):
            # ARRIVAL first: barrier() sends this rank's [1] to rank 0 the
            # moment its algorithm work finished. The student local view +
            # C_ROUND_DONE upload run inside the barrier's on_arrived window
            # (while waiting for the release), so UI/upload never enter the
            # local-work / ready timings.
            BarrierMod.barrier(rt.comm, rnd,
                               on_arrived=lambda _r: _show_and_report(
                                   rt, control, _r))
        rt._barrier = round_barrier
    else:
        rt.show_ui = False
        rt._report = None
        rt._barrier = None
        rt._on_round = None
        rt._local_ctx = None

    result = rt.run_algorithm(params, value=data, barrier=False)
    final = None if params.get("payload") else _encode(result)
    control.send({"t": P.C_DONE, "rank": rt.rank, "final": final,
                  "events": len(rt.events.events)})
    return result


# --------------------------------------------------------------------------
# helpers below — the run loop above is the student-facing program
# --------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="127.0.0.1:9000",
                    help="rank 0 ip:port")
    return ap.parse_args()


def print_banner(rank, size):
    print("\nMiniMPI Worker")
    print("Rank: %d / %d\n" % (rank, size))
    print("Waiting for Rank 0...")


def read_one_integer(rank):
    """Each student types one integer per RUN; vector = [n] * data_size.
    Headless (non-tty) fallback: rank + 1 so automation stays deterministic."""
    if os.environ.get("MINIMPI_INPUT_DELAY"):
        time.sleep(float(os.environ["MINIMPI_INPUT_DELAY"]))
    if sys.stdin.isatty():
        try:
            line = input("Input one integer:\n> ").strip()
            n = int(line)
        except (EOFError, ValueError):
            n = rank + 1
        return n
    return rank + 1


def show_result(rank, run, result):
    if result is None or run.params.get("payload"):
        return
    first = result[0] if isinstance(result, list) and result else result
    if run.mode == "teaching":
        _print_final(rank, run, result, first)
    else:
        print("\nResult: %s\n" % first)


def _print_final(rank, run, result, first):
    """Student-side 'Collective Complete' block (teaching mode)."""
    alg = run.algorithm
    allreduce = alg in ("naive_allreduce", "recursive_doubling_allreduce", "ring_allreduce")
    print("\n========================================")
    print("Collective Complete")
    print("========================================")
    print("\nResult:")
    if isinstance(result, list) and result:
        print(T.preview_vector(result))
        print("Summary: %s" % first)
        if allreduce:
            print("\nAll elements have the same value.")
            print("Every rank received the same reduced result.")
        elif rank == 0:
            print("\nRank 0 (root) owns the final reduced result.")
        else:
            print("\nThis rank is not the root.")
            print("Its local data is not the global Reduce result.")
    elif result is not None:
        print(str(result))
    print()


def _encode(value):
    if isinstance(value, (bytes, bytearray)):
        return {"raw": base64.b64encode(bytes(value)).decode()}
    if isinstance(value, list):
        return {"vec": value}
    return {"vec": [value]}


def _show_and_report(rt, control, rnd):
    """Student side of a teaching round sync — runs INSIDE the round
    barrier's on_arrived window: arrival already reported to rank 0 (so all
    round timings are fixed), now print the local view + upload events while
    waiting for the release."""
    _show_round(rt, rnd)
    control.send({"t": P.C_ROUND_DONE, "rnd": rnd,
                  "events": [e.light_dict()
                             for e in rt.comm.events.by_round(rnd)]})


def _show_round(rt, rnd):
    """Student local view for one finished logical round (teaching mode)."""
    # test-only: slow down the LOCAL UI after arrival (proves UI never
    # enters round timing; see tests/test_round_behavior F).
    if os.environ.get("MINIMPI_LOCAL_VIEW_DELAY"):
        time.sleep(float(os.environ["MINIMPI_LOCAL_VIEW_DELAY"]))
    evs = [e for e in rt.comm.events.by_round(rnd) if e.kind == "algorithm"]
    ctx = getattr(rt, "_local_ctx", None)
    if ctx is None:
        ds = int(rt.run_meta.get("data_size") or 0)
        base = getattr(rt, "typed_value", rt.rank + 1)
        ctx = T.RoundCtx(rt.run_meta.get("algorithm", ""), rt.size, rt.rank,
                         [base] * (ds or 1))
        rt._local_ctx = ctx
    view = T.describe_round(ctx, rnd, evs)
    lines = [T.local_view_text(view)]
    # Local Clock: every value on THIS rank's own clock, ms from Round Start
    lines.append("\nLOCAL TIMELINE")
    lines.extend(T.local_timeline_text(rt.comm_world.local_timings()))
    lines.append("\nWaiting at round synchronization...")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
