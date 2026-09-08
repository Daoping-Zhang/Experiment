#!/usr/bin/env python3
"""check.py — Assignment 1 PUBLIC checker + the ONE parsing/checking logic.

The instructor grader uses these very same functions (single source of
truth): it only replaces the INPUT cases. See check_submission.py for the
one-click local test.

Usage:
    python3 check.py <input.txt> <output.txt>
    (as a module) from check import read_input, compute_expected,
    parse_output, check_output
"""
import re
import sys

RESULT_RE = re.compile(
    r"^(REDUCE|ALLREDUCE)\s+rank=(\d+):\s+(-?\d+(?:\s+-?\d+)*)\s*$")


# --------------------------------------------------------------------------
# Input / expected
# --------------------------------------------------------------------------
def read_input(path):
    """Parse the input file: 'P N' then P rows of N ints (one vector/rank)."""
    with open(path) as f:
        lines = [ln for ln in f.read().splitlines() if ln.strip()]
    if not lines:
        raise ValueError("empty input file")
    head = lines[0].split()
    P, N = int(head[0]), int(head[1])
    rows = []
    for i in range(1, P + 1):
        if i >= len(lines):
            raise ValueError("input: expected %d vector rows, found %d"
                             % (P, len(lines) - 1))
        row = [int(x) for x in lines[i].split()]
        if len(row) != N:
            raise ValueError("input: rank %d row has %d values (need %d)"
                             % (i - 1, len(row), N))
        rows.append(row)
    return P, N, rows


def compute_expected(rows):
    """Element-wise SUM over all rank vectors."""
    return [sum(v[k] for v in rows) for k in range(len(rows[0]))]


# --------------------------------------------------------------------------
# Output parsing  (order independent; extra lines ignored)
# --------------------------------------------------------------------------
def parse_output(text):
    """-> (results, malformed_msgs)

    results: {'REDUCE': {0: [..]}, 'ALLREDUCE': {rank: [..]}}
    malformed_msgs: list of problems (duplicate/missing required lines,
    non-rank-0 REDUCE, wrong token count, etc.). Free-form debug lines are
    ignored.
    """
    results = {"REDUCE": {}, "ALLREDUCE": {}}
    malformed = []
    for ln in text.splitlines():
        m = RESULT_RE.match(ln.strip())
        if not m:
            continue                      # free-form debug output: ignored
        op, rank_s, vec_s = m.group(1), m.group(2), m.group(3)
        rank = int(rank_s)
        vec = [int(x) for x in vec_s.split()]
        if op == "REDUCE" and rank != 0:
            malformed.append("REDUCE must be emitted by rank 0 only "
                             "(found rank=%d)" % rank)
            continue
        if rank in results[op]:
            malformed.append("duplicate %s rank=%d" % (op, rank))
            continue
        results[op][rank] = vec
    return results, malformed


# --------------------------------------------------------------------------
# Final comparison
# --------------------------------------------------------------------------
def check_output(P, N, rows, text, malformed_extra=None):
    """-> (ok, reasons, per_part)

    per_part: {'reduce': bool, 'allreduce': {rank: bool}}
    """
    expected = compute_expected(rows)
    results, malformed = parse_output(text)
    reasons = list(malformed or [])
    if malformed_extra:
        reasons.extend(malformed_extra)

    reduce_ok = True
    r_lines = results["REDUCE"]
    if r_lines.get(0) is None:
        reduce_ok = False
        reasons.append("REDUCE rank=0 missing")
    elif len(r_lines) != 1:
        reduce_ok = False
        reasons.append("REDUCE appears more than once")
    elif r_lines[0] != expected:
        reduce_ok = False
        reasons.append("REDUCE rank=0 wrong value: %s (expected %s)"
                       % (" ".join(map(str, r_lines[0])),
                          " ".join(map(str, expected))))

    ar = results["ALLREDUCE"]
    all_ok = {}
    for rank in range(P):
        v = ar.get(rank)
        if v is None:
            all_ok[rank] = False
            reasons.append("ALLREDUCE rank=%d missing" % rank)
        elif v != expected:
            all_ok[rank] = False
            reasons.append("ALLREDUCE rank=%d wrong value: %s (expected %s)"
                           % (rank, " ".join(map(str, v)),
                              " ".join(map(str, expected))))
        else:
            all_ok[rank] = True
    for rank in ar:
        if rank < 0 or rank >= P:
            reasons.append("ALLREDUCE rank=%d out of range (P=%d)" % (rank, P))

    overall = reduce_ok and all(all_ok.values()) and not reasons
    return overall, reasons, {"reduce": reduce_ok,
                              "allreduce": all_ok}


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def _fmt(vec):
    return " ".join(str(x) for x in vec)


def main(argv):
    if len(argv) != 3:
        print("usage: python3 check.py <input.txt> <output.txt>")
        return 2
    P, N, rows = read_input(argv[1])
    text = open(argv[2]).read() if argv[2] != "-" else sys.stdin.read()
    expected = compute_expected(rows)
    ok, reasons, parts = check_output(P, N, rows, text)
    print("Checking Assignment 1")
    print("\nExpected:")
    print("  REDUCE rank=0: %s" % _fmt(expected))
    for r in range(P):
        print("  ALLREDUCE rank=%d: %s" % (r, _fmt(expected)))
    print("\nREDUCE")
    print("  %s" % ("PASS" if parts["reduce"] else "FAIL"))
    print("\nALLREDUCE")
    for r in range(P):
        print("  Rank %d %s" % (r, "PASS" if parts["allreduce"].get(r)
                                else "FAIL"))
    print("\nOverall: %s" % ("PASS" if ok else "FAIL"))
    for msg in reasons:
        print("  ! %s" % msg)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
