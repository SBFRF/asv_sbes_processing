#!/usr/bin/env python3
"""Shrink RTKlib POS file to a time window around a center epoch.

Usage:
    shrink_pos.py INPUT OUTPUT --center EPOCH --window SECONDS

The center is a Unix epoch timestamp. Window is total seconds (±half on each side).
"""
import argparse
import sys
from datetime import datetime, timezone, timedelta


def parse_pos_datetime(line):
    """Parse datetime from a POS data line. Returns None if not parseable."""
    parts = line.split()
    if len(parts) < 2:
        return None
    dt_str = parts[0] + " " + parts[1]
    try:
        return datetime.strptime(dt_str, "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def main():
    parser = argparse.ArgumentParser(description="Shrink POS file to time window")
    parser.add_argument("input", help="Input POS file")
    parser.add_argument("output", help="Output POS file")
    parser.add_argument("--center", type=float, required=True, help="Center epoch (Unix timestamp)")
    parser.add_argument("--window", type=float, default=300, help="Window size in seconds (default: 300)")
    args = parser.parse_args()

    center_dt = datetime.fromtimestamp(args.center, tz=timezone.utc)
    half_window = args.window / 2
    start_dt = center_dt - timedelta(seconds=half_window)
    end_dt = center_dt + timedelta(seconds=half_window)

    print(f"Center: {center_dt.isoformat()}")
    print(f"Window: {start_dt.isoformat()} to {end_dt.isoformat()}")

    with open(args.input, "r") as f:
        lines = f.readlines()

    # Separate header (lines starting with %) from data
    header_lines = []
    data_lines = []
    for line in lines:
        if line.startswith("%"):
            header_lines.append(line)
        elif line.strip():
            data_lines.append(line)

    # Filter data lines by time
    filtered_data = []
    filtered_times = []
    for line in data_lines:
        dt = parse_pos_datetime(line)
        if dt and start_dt <= dt <= end_dt:
            filtered_data.append(line)
            filtered_times.append(dt)

    if not filtered_data:
        print(f"ERROR: No data in window for {args.input}", file=sys.stderr)
        sys.exit(1)

    print(f"Filtered: {len(filtered_data)} rows (from {len(data_lines)})")

    # Update obs start/end in header
    new_start = filtered_times[0]
    new_end = filtered_times[-1]
    updated_header = []
    for line in header_lines:
        if line.startswith("% obs start"):
            # Format: % obs start : 2023/11/09 12:33:08.1 UTC (week2287 390806.1s)
            updated_header.append(f"% obs start : {new_start.strftime('%Y/%m/%d %H:%M:%S.%f')[:-5]} UTC (trimmed)\n")
        elif line.startswith("% obs end"):
            updated_header.append(f"% obs end   : {new_end.strftime('%Y/%m/%d %H:%M:%S.%f')[:-5]} UTC (trimmed)\n")
        else:
            updated_header.append(line)

    # Write output
    with open(args.output, "w") as f:
        f.writelines(updated_header)
        f.writelines(filtered_data)

    print(f"Written to {args.output}")


if __name__ == "__main__":
    main()
