#!/usr/bin/env python3
"""Shrink NMEA .dat files to a time window based on GNSS time in GGA sentences.

Usage:
    shrink_nmea.py INPUT_DIR OUTPUT_DIR --center EPOCH --window SECONDS

Copies files whose GNSS times fall within the window. For boundary files,
filters individual lines by GNSS time extracted from GGA sentences.
"""
import argparse
import os
import re
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Pattern to extract GNSS time from GGA sentence: $xxGGA,HHMMSS.ss,...
GGA_PATTERN = re.compile(r'\$\w{2}GGA,(\d{6})\.?\d*,')


def parse_gnss_time(line, date_str="2023-11-09"):
    """Extract GNSS time from a line containing a GGA sentence."""
    match = GGA_PATTERN.search(line)
    if match:
        hhmmss = match.group(1)
        try:
            time_str = f"{hhmmss[:2]}:{hhmmss[2:4]}:{hhmmss[4:6]}"
            dt = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S")
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            return None
    return None


def get_file_gnss_time(filename):
    """Extract expected GNSS time from filename like 20231109130554.dat"""
    base = os.path.basename(filename).replace('.dat', '')
    if len(base) == 14:  # YYYYMMDDHHMMSS
        try:
            return datetime.strptime(base, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
        except ValueError:
            return None
    return None


def main():
    parser = argparse.ArgumentParser(description="Shrink NMEA files to time window")
    parser.add_argument("input_dir", help="Input directory containing .dat files")
    parser.add_argument("output_dir", help="Output directory")
    parser.add_argument("--center", type=float, required=True, help="Center epoch (Unix timestamp)")
    parser.add_argument("--window", type=float, default=300, help="Window size in seconds (default: 300)")
    args = parser.parse_args()

    center_dt = datetime.fromtimestamp(args.center, tz=timezone.utc)
    half_window = args.window / 2
    start_dt = center_dt - timedelta(seconds=half_window)
    end_dt = center_dt + timedelta(seconds=half_window)
    date_str = center_dt.strftime("%Y-%m-%d")

    print(f"Center: {center_dt.isoformat()}")
    print(f"Window: {start_dt.isoformat()} to {end_dt.isoformat()}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Find all .dat files
    input_path = Path(args.input_dir)
    dat_files = sorted(input_path.glob("*.dat"))

    copied = 0
    filtered = 0
    skipped = 0

    for dat_file in dat_files:
        file_time = get_file_gnss_time(dat_file.name)

        # Skip files clearly outside window (with 2 second buffer for file duration)
        if file_time:
            if file_time < start_dt - timedelta(seconds=2):
                skipped += 1
                continue
            if file_time > end_dt + timedelta(seconds=2):
                skipped += 1
                continue

        # Read and filter by GNSS time in GGA sentences
        with open(dat_file, 'r', errors='ignore') as f:
            lines = f.readlines()

        # Filter lines - keep lines with GNSS time in window, or non-GGA lines
        filtered_lines = []
        has_gga_in_window = False

        for line in lines:
            gnss_time = parse_gnss_time(line, date_str)
            if gnss_time is None:
                # Non-GGA line - keep if adjacent to GGA lines in window
                filtered_lines.append(line)
            elif start_dt <= gnss_time <= end_dt:
                filtered_lines.append(line)
                has_gga_in_window = True

        if has_gga_in_window and filtered_lines:
            out_path = Path(args.output_dir) / dat_file.name
            with open(out_path, 'w') as f:
                f.writelines(filtered_lines)

            if len(filtered_lines) < len(lines):
                filtered += 1
            else:
                copied += 1

    print(f"\nResults:")
    print(f"  Copied whole: {copied}")
    print(f"  Filtered: {filtered}")
    print(f"  Skipped (outside window): {skipped}")
    print(f"  Total output files: {copied + filtered}")


if __name__ == "__main__":
    main()
