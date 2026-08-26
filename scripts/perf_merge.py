#!/usr/bin/env python3
"""Merge `perf stat` hardware counters into a benchmark's CSV row.

The benchmark binary prints exactly one CSV row to stdout (hardware columns are
"NA"). `perf stat` is run around it, writing counters to a sidecar file. This
script splices the measured counter values (and derived ratios such as IPC or
branch-miss-rate) into that row and appends the result to the target CSV.

Usage:
  perf_merge.py --row ROW --sidecar FILE --csv OUT.csv --header HDR \
      --map COL:event[,COL:event...] --derive COL:a/b[,COL:c/d...] \
      --format json|csv
"""
import argparse
import json
import os
import sys

KNOWN_EVENTS = [
    "cycles", "instructions", "branches", "branch-misses",
    "cache-references", "cache-misses", "LLC-loads", "LLC-load-misses",
    "LLC-load-misses", "LLC-store-misses",
]


def fmt(v):
    if v is None:
        return "NA"
    if isinstance(v, float):
        return format(v, ".6g")
    return str(v)


def parse_json_sidecar(path):
    values = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            event = obj.get("event")
            if event not in KNOWN_EVENTS:
                continue
            for key in ("counter-value", "value"):
                if key in obj:
                    try:
                        values[event] = float(obj[key])
                    except (TypeError, ValueError):
                        pass
                    break
    return values


def parse_csv_sidecar(path):
    """Flexible parser for `perf stat -x,` output (format varies by version)."""
    values = {}
    with open(path) as f:
        for line in f:
            parts = [p.strip() for p in line.split(",")]
            if not parts:
                continue
            event = None
            for p in parts:
                if p in KNOWN_EVENTS:
                    event = p
                    break
            if event is None:
                continue
            for p in parts:
                try:
                    values[event] = float(p)
                    break
                except ValueError:
                    continue
    return values


def lookup_val(name, lookup):
    name = name.strip()
    if name in lookup:
        return float(lookup[name])
    return float(name)


def eval_expr(expr, lookup):
    expr = expr.strip()
    if "/" in expr:
        a, b = expr.split("/", 1)
        return lookup_val(a, lookup) / lookup_val(b, lookup)
    return lookup_val(expr, lookup)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--row", required=True)
    ap.add_argument("--sidecar", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--header", required=True)
    ap.add_argument("--map", default="")
    ap.add_argument("--derive", default="")
    ap.add_argument("--format", default="csv")
    args = ap.parse_args()

    values = parse_json_sidecar(args.sidecar) if args.format == "json" else parse_csv_sidecar(args.sidecar)
    if not values:
        # No counters collected (e.g. permission denied): leave columns as NA.
        pass

    header = [h.strip() for h in args.header.split(",")]
    row = [r.strip() for r in args.row.split(",")]

    lookup = dict(values)
    if args.map:
        for item in args.map.split(","):
            item = item.strip()
            if not item or ":" not in item:
                continue
            col, ev = item.split(":", 1)
            col, ev = col.strip(), ev.strip()
            if col in header and ev in values:
                lookup[col] = values[ev]
                row[header.index(col)] = fmt(values[ev])
    if args.derive:
        for item in args.derive.split(","):
            item = item.strip()
            if not item or ":" not in item:
                continue
            col, expr = item.split(":", 1)
            col = col.strip()
            try:
                num = eval_expr(expr, lookup)
                if col in header:
                    row[header.index(col)] = fmt(num)
            except (ZeroDivisionError, ValueError, KeyError):
                pass

    write_header = not os.path.exists(args.csv) or os.path.getsize(args.csv) == 0
    with open(args.csv, "a") as f:
        if write_header:
            f.write(",".join(header) + "\n")
        f.write(",".join(row) + "\n")


if __name__ == "__main__":
    main()
