#!/usr/bin/env python3
"""Compute median epoch (UTC) from a collection of test data files.

Heuristics implemented:
- HDF5/.h5: look for a 1D dataset named 'time' or 'timestamp' (seconds since epoch)
- LLH (.LLH): parse ISO timestamps in text lines
- RINEX3 (.23O/.23P/.23B): look for epoch lines starting with '>' and parse

This is a best-effort helper used by the shrink wrapper.
"""
import argparse
import os
import re
import sys
from datetime import datetime, timezone

try:
    import h5py
except Exception:
    h5py = None

ISO_RE = re.compile(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})")


def parse_llh(path):
    times = []
    with open(path, "r", errors="ignore") as f:
        for line in f:
            m = ISO_RE.search(line)
            if m:
                try:
                    dt = datetime.fromisoformat(m.group(1)).replace(tzinfo=timezone.utc)
                    times.append(dt.timestamp())
                except Exception:
                    pass
    return times


def parse_rinex3(path):
    times = []
    with open(path, "r", errors="ignore") as f:
        for line in f:
            if line.startswith(">"):
                # Example: "> 2023 11 09 12 34 56.0000000  0  0"
                parts = line[1:].strip().split()
                if len(parts) >= 6:
                    try:
                        year, month, day, hour, minute = map(int, parts[:5])
                        sec = float(parts[5])
                        dt = datetime(year, month, day, hour, minute, int(sec), tzinfo=timezone.utc)
                        times.append(dt.timestamp())
                    except Exception:
                        pass
    return times


def parse_h5(path):
    times = []
    if h5py is None:
        return times
    try:
        with h5py.File(path, "r") as f:
            for name in ("time", "timestamp", "times"):
                if name in f:
                    ds = f[name]
                    try:
                        arr = ds[:]
                        # assume seconds since epoch
                        times.extend([float(x) for x in arr.flatten()])
                        break
                    except Exception:
                        pass
    except Exception:
        pass
    return times


def collect_times(root):
    times = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            path = os.path.join(dirpath, fn)
            ln = fn.lower()
            if ln.endswith(".llh"):
                times.extend(parse_llh(path))
            elif any(ln.endswith(ext) for ext in (".23o", ".23p", ".23b", ".o", ".p", ".b")):
                times.extend(parse_rinex3(path))
            elif ln.endswith(".h5"):
                times.extend(parse_h5(path))
    return times


def main():
    p = argparse.ArgumentParser(description="Compute median epoch from test data folder")
    p.add_argument("root", help="root data folder (e.g., tests/data/20231109)")
    args = p.parse_args()
    times = collect_times(args.root)
    if not times:
        print("No timestamps found", file=sys.stderr)
        sys.exit(2)
    times.sort()
    mid = times[len(times) // 2]
    dt = datetime.fromtimestamp(mid, tz=timezone.utc)
    print(dt.isoformat())


if __name__ == "__main__":
    main()
