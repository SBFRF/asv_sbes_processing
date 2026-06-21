#!/usr/bin/env python3
"""Shrink RINEX observation files to a time window.

Supports both RINEX 2.x (epochs start with ' YY MM DD') and RINEX 3.x (epochs start with '>').
Keeps header and filters epoch records by time.

Usage:
    shrink_rinex.py INPUT OUTPUT --center EPOCH --window SECONDS
"""
import argparse
import re
from datetime import datetime, timezone
import sys


# RINEX 3.x epoch line: "> YYYY MM DD HH MM SS.sssssss"
def parse_rinex3_epoch(line):
    parts = line[1:].strip().split()
    if len(parts) < 6:
        return None
    try:
        year, month, day, hour, minute = map(int, parts[:5])
        sec = float(parts[5])
        dt = datetime(year, month, day, hour, minute, int(sec), tzinfo=timezone.utc)
        return dt.timestamp()
    except Exception:
        return None


# RINEX 2.x epoch line: " YY MM DD HH MM SS.sssssss  F NN"
RINEX2_EPOCH = re.compile(r'^\s+(\d{2})\s+(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\s+(\d+\.\d+)\s+\d+\s+\d+')


def parse_rinex2_epoch(line):
    match = RINEX2_EPOCH.match(line)
    if match:
        yy, mo, dd, hh, mm, ss = match.groups()
        year = 2000 + int(yy) if int(yy) < 80 else 1900 + int(yy)
        try:
            dt = datetime(year, int(mo), int(dd), int(hh), int(mm), int(float(ss)),
                         tzinfo=timezone.utc)
            return dt.timestamp()
        except ValueError:
            return None
    return None


def is_epoch_line(line):
    """Check if line is an epoch line (RINEX 2 or 3)."""
    if line.startswith('>'):
        return True
    if RINEX2_EPOCH.match(line):
        return True
    return False


def parse_epoch_time(line):
    """Parse epoch time from either RINEX 2 or 3 format."""
    if line.startswith('>'):
        return parse_rinex3_epoch(line)
    return parse_rinex2_epoch(line)


def main():
    p = argparse.ArgumentParser(description='Shrink RINEX observation file by epoch window')
    p.add_argument('input')
    p.add_argument('output')
    p.add_argument('--center', required=True, type=float, help='center epoch (Unix timestamp)')
    p.add_argument('--window', type=float, default=300.0, help='window size in seconds')
    args = p.parse_args()

    half_window = args.window / 2
    start_ts = args.center - half_window
    end_ts = args.center + half_window

    print(f"Center: {datetime.fromtimestamp(args.center, tz=timezone.utc).isoformat()}")
    print(f"Window: {datetime.fromtimestamp(start_ts, tz=timezone.utc).isoformat()} to "
          f"{datetime.fromtimestamp(end_ts, tz=timezone.utc).isoformat()}")

    with open(args.input, 'r', errors='ignore') as f:
        lines = f.readlines()

    # Find end of header
    header_end = 0
    for i, line in enumerate(lines):
        if 'END OF HEADER' in line:
            header_end = i + 1
            break

    header = lines[:header_end]
    data_lines = lines[header_end:]

    # Filter epochs
    filtered = []
    keep_epoch = False
    epochs_kept = 0
    epochs_total = 0

    for line in data_lines:
        if is_epoch_line(line):
            ts = parse_epoch_time(line)
            epochs_total += 1
            if ts is not None and start_ts <= ts <= end_ts:
                keep_epoch = True
                epochs_kept += 1
            else:
                keep_epoch = False

        if keep_epoch:
            filtered.append(line)

    print(f"Epochs: {epochs_kept}/{epochs_total} kept")

    if epochs_kept == 0:
        print("ERROR: No epochs in window!", file=sys.stderr)
        sys.exit(1)

    with open(args.output, 'w') as f:
        f.writelines(header)
        f.writelines(filtered)

    print(f"Written to {args.output}")


if __name__ == '__main__':
    main()
