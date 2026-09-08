#!/usr/bin/env python3
"""check_submission.py — one-click LOCAL test for Assignment 1.

Usage:
    python3 check_submission.py <submission_dir> [input.txt]

It:
    1. requires solution.c + build.sh + run.sh (all mandatory; C-only)
    2. runs build.sh, then `mpirun -n P ./run.sh <input>` in the submission dir
    3. checks stdout with the SAME logic as check.py
    4. prints REDUCE / ALLREDUCE / Overall PASS or FAIL

The instructor grader follows the exact same steps but replaces the input
with hidden cases and enforces a timeout.
"""
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from check import read_input, compute_expected, check_output  # noqa: E402
from check import _fmt  # noqa: E402


def _mpirun():
    env = os.environ.get("MPIRUN")
    if env:
        return env
    found = shutil.which("mpirun")
    if not found:
        sys.exit("mpirun not found — set $MPIRUN or add it to PATH")
    return found


def build_if_needed(sub_dir):
    """REQUIRED build.sh (its existence is checked before this call)."""
    build = os.path.join(sub_dir, "build.sh")
    r = subprocess.run(["bash", build], cwd=sub_dir, capture_output=True,
                       text=True, timeout=120)
    if r.returncode != 0:
        return False, "build.sh failed:\n" + (r.stdout + r.stderr)
    return True, "build.sh ok"


def run_submission(sub_dir, input_path, timeout=60):
    """Run `mpirun -n P ./run.sh <input>`.

    Returns (ok, reasons, parts, stdout):
      parts = {'reduce': bool, 'allreduce': {rank: bool}} when the output
      could be parsed (even if wrong), else None.
    """
    missing = [f for f in ("solution.c", "build.sh", "run.sh")
               if not os.path.exists(os.path.join(sub_dir, f))]
    if missing:
        return False, ["Required file missing: %s" % m for m in missing], None, ""
    for f in ("build.sh", "run.sh"):
        os.chmod(os.path.join(sub_dir, f), 0o755)
    build_ok, note = build_if_needed(sub_dir)
    if not build_ok:
        return False, [note], None, ""

    P, N, rows = read_input(input_path)
    try:
        proc = subprocess.Popen(
            [_mpirun(), "-n", str(P), "./run.sh", input_path],
            cwd=sub_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, start_new_session=True)
        out, _ = proc.communicate(timeout=timeout)
        ok = proc.returncode == 0
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), 9)   # kill whole mpirun group
        except Exception:  # noqa: BLE001
            pass
        proc.wait(timeout=10)
        return False, ["TIMEOUT after %ds — possible MPI deadlock.\n"
                       "(mpirun process group killed)" % timeout], None, ""
    if not ok:
        return False, ["program exited with %d\n%s"
                       % (proc.returncode, (out or "")[:2000])], None, out
    overall, reasons, parts = check_output(P, N, rows, out)
    return overall, reasons, parts, out


def main(argv):
    if len(argv) < 2:
        print("usage: python3 check_submission.py <submission_dir> "
              "[input.txt]")
        return 2
    sub_dir = os.path.abspath(argv[1])
    if not os.path.isdir(sub_dir):
        print("submission dir not found: %s" % sub_dir)
        return 2
    input_path = os.path.abspath(argv[2]) if len(argv) > 2 \
        else os.path.join(HERE, "demo_input.txt")

    P, N, rows = read_input(input_path)
    expected = compute_expected(rows)
    import shutil
    print("Checking Assignment 1")
    print("Submission      : %s" % sub_dir)
    print("Checker Python  : %s" % sys.executable)
    print("MPI Runtime     : %s"
          % (shutil.which("mpirun") or "mpirun not found on PATH"))
    print("MPI Compiler    : %s"
          % (shutil.which("mpicc") or "mpicc not found on PATH"))
    print("Input           : %s  (P=%d N=%d)"
          % (os.path.basename(input_path), P, N))

    ok, reasons, parts, _out = run_submission(sub_dir, input_path)
    print("\nExpected REDUCE rank=0: %s" % _fmt(expected))
    print("\nREDUCE")
    if parts is not None:
        print("  %s" % ("PASS" if parts["reduce"] else "FAIL"))
        print("\nALLREDUCE")
        for r in range(P):
            ok_r = parts["allreduce"].get(r)
            print("  Rank %d %s" % (r, "PASS" if ok_r else "FAIL"))
    else:
        print("  n/a (could not run / no output parsed)")
    print("\nOverall: %s" % ("PASS" if ok else "FAIL"))
    for msg in (reasons or []):
        print("  ! %s" % msg)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
