#!/usr/bin/env python3
"""test_teaching_semantics.py — Teaching View / UI-semantics tests.

G. Tree local semantics  : rank 0 BEFORE [1..] -> Receive [2..] -> AFTER [3..]
                           (real execution, teacher log)
H. Tree phase labels     : P=4 -> rounds 1-2 'Phase 1: Reduce',
                           rounds 3-4 'Phase 2: Broadcast' (in order)
I. Ring phase + chunk    : P=4 -> 3x Reduce-Scatter then 3x AllGather rounds;
                           per-message Chunk ids appear in the global view
J. Rank 0 is a real participant : naive allreduce final Result: 10 and the
                           Rank-0 LOCAL view shows real before/receive/after
K. Performance unaffected: teaching keywords absent from performance output

Also in-memory unit checks: phase_of / ring_chunk_index / total_rounds.

Run:  python3 tests/test_teaching_semantics.py
"""
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(os.path.dirname(HERE), "scripts")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(HERE))

from _proc import Runner  # noqa: E402

PASS = []
FAIL = []


def check(name, ok, extra=""):
    (PASS if ok else FAIL).append(name)
    print("[%s] %s%s" % ("PASS" if ok else "FAIL", name,
                         ("  (%s)" % extra) if extra else ""))


def _run_teaching_demo(demo, timeout=120):
    """Run teacher + 3 workers (values rank+1) in teaching mode, return
    (teacher_log, [worker logs rank1..3])."""
    r = Runner(4, timeout=timeout)
    t = r.teacher(["--auto", "--demo", demo, "--mode", "teaching",
                   "--data-size", "4"])
    time.sleep(2)
    ws = [subprocess.Popen(
        [sys.executable, os.path.join(os.path.dirname(HERE), "worker.py"),
         "--server", "127.0.0.1:%d" % r.port],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for _ in range(3)]
    t_out = t.communicate(timeout=timeout)[0]
    r.kill()
    w_out = [w.communicate(timeout=5)[0] for w in ws]
    return t_out, w_out


def _between(text, start_marker, end_marker):
    i = text.find(start_marker)
    if i < 0:
        return ""
    j = text.find(end_marker, i + len(start_marker))
    return text[i:j if j >= 0 else len(text)]


# --------------------------------------------------------------------------
def test_units():
    from minimpi import teaching as T
    ok1 = T.total_rounds("recursive_doubling_allreduce", 4) == 2  # log2(4)
    ok2 = T.phase_of("recursive_doubling_allreduce", 4, 1)[0] == \
        "Phase: Exchange + Reduce" and \
        T.phase_of("recursive_doubling_allreduce", 4, 2)[0] == \
        "Phase: Exchange + Reduce"
    ok3 = T.phase_of("ring_allreduce", 4, 2)[0] == "Phase 1: Reduce-Scatter" and \
          T.phase_of("ring_allreduce", 4, 4)[0] == "Phase 2: AllGather"
    ok4 = T.ring_chunk_index(0, 4, 1, "send") == 0 and \
          T.ring_chunk_index(0, 4, 1, "recv") == 3 and \
          T.ring_chunk_index(0, 4, 4, "send") == 1   # owned=(0+1)%4
    check("unit. phase/chunk mapping", ok1 and ok2 and ok3 and ok4)


def test_g_rd_local_semantics():
    t_out, _ = _run_teaching_demo("recursive_doubling_allreduce")
    # rank-0 local view, round 1: BEFORE [1,1,1,1], exchange with rank 1:
    # send + receive [2,2,2,2], SUM -> AFTER [3,3,3,3] (real execution).
    sec = _between(t_out, "RANK 0 LOCAL VIEW", "LOCAL TIMELINE")
    ok = ("My Role: Sender + Receiver + Reduce" in sec
          and "[1, 1, 1, 1]" in sec
          and "[2, 2, 2, 2]" in sec
          and "[3, 3, 3, 3]" in sec
          and "[1, 1, 1, 1] + [2, 2, 2, 2] = [3, 3, 3, 3]" in sec)
    check("G. recursive doubling rank0 exchange semantics (1<->2 -> 3)", ok)


def test_h_rd_two_rounds_no_broadcast():
    t_out, _ = _run_teaching_demo("recursive_doubling_allreduce")
    # P=4 -> exactly 2 exchange rounds; no Reduce/Broadcast two-phase labels
    first = t_out.find("Phase: Exchange + Reduce\nRound 1 / 2")
    second = t_out.find("Phase: Exchange + Reduce\nRound 2 / 2")
    ok = first >= 0 and second > first
    ok = ok and "Phase 1: Reduce" not in t_out and "Phase 2: Broadcast" \
        not in t_out and "My Role: Idle" not in t_out
    check("H. RD: 2 rounds, every round Exchange + Reduce, no idle/broadcast",
          ok)


def test_i_ring_phase_and_chunk():
    t_out, w_out = _run_teaching_demo("ring_allreduce", timeout=150)
    # 6 teaching rounds exist; the phase label appears twice per round
    # (round header + rank-0 local header), so assert presence + ORDER,
    # not exact counts.
    first_rs = t_out.find("Phase 1: Reduce-Scatter")
    first_ag = t_out.find("Phase 2: AllGather")
    ok_phase = (first_rs >= 0 and first_ag > first_rs
                and all(("Round %d / 6" % r) in t_out for r in range(1, 7)))
    ok_chunk = "Chunk 0" in t_out and "Chunk 3" in t_out
    # find the actual rank-3 worker log (join order may vary)
    r3 = next((o for o in w_out if "Rank: 3 / 4" in o), "")
    sec = _between(r3, "My Role: Sender + Receiver + Reduce", "TIMING")
    # rank 3 value = 4: round-1 reduce-scatter sends Chunk 3, receives
    # Chunk 2 from rank 2 (value 3) -> real SUM 4 + 3 = 7
    ok_student = bool(sec) and "[Chunk 3]" in sec and "[Chunk 2]" in sec and \
                 "Reduce Chunk 2" in sec and "[4] + [3] = [7]" in sec
    check("I. ring phases RS->AG in order, 6 rounds, chunk ids",
          ok_phase and ok_chunk,
          "" if ok_phase and ok_chunk else "phase/chunk")
    check("I2. ring student chunk send/recv + reduce arithmetic",
          ok_student)


def test_j_rank0_real_participant():
    t_out, _ = _run_teaching_demo("naive_allreduce")
    sec1 = _between(t_out, "RANK 0 LOCAL VIEW", "TIMING")
    # rank-0 REAL run: value0 = 1 default; students fallback 2,3,4 -> sum 10
    ok_before = "[1, 1, 1, 1]" in sec1
    ok_after_r1 = "[10, 10, 10, 10]" in sec1
    ok_result = bool(re.search(r"Result:\s*10", t_out)) and \
                "All ranks received the same reduced result." in t_out
    check("J. rank 0 real participant (before/receive/after/result)",
          ok_before and ok_after_r1 and ok_result,
          "" if ok_before and ok_after_r1 and ok_result else "naive view")


def test_k_performance_unaffected():
    r = Runner(4, timeout=90)
    t = r.teacher(["--auto", "--demo", "naive_allreduce", "--mode",
                   "performance", "--data-size", "16"])
    time.sleep(2)
    ws = [subprocess.Popen(
        [sys.executable, os.path.join(os.path.dirname(HERE), "worker.py"),
         "--server", "127.0.0.1:%d" % r.port],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for _ in range(3)]
    t_out = t.communicate(timeout=70)[0]
    r.kill()
    w_out = "".join(w.communicate(timeout=5)[0] for w in ws)
    combined = t_out + w_out
    bad = [tok for tok in ("BEFORE", "AFTER", "MY ROLE", "CHUNK",
                           "GLOBAL COMMUNICATION", "OPERATIONS",
                           "RANK 0 LOCAL VIEW", "Whole Round Finished",
                           "Local data ready", "Waiting for all ranks",
                           "All ranks ready") if tok in combined]
    check("K. performance output free of teaching state", not bad,
          "found: %s" % ", ".join(bad) if bad else "")


def main():
    test_units()
    test_g_rd_local_semantics()
    test_h_rd_two_rounds_no_broadcast()
    test_j_rank0_real_participant()
    test_i_ring_phase_and_chunk()
    test_k_performance_unaffected()
    print("\nTeaching-semantics tests: %d passed, %d failed"
          % (len(PASS), len(FAIL)))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
