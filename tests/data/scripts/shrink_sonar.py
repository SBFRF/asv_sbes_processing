#!/usr/bin/env python3
"""Shrink S500 sonar binary .dat files to a time window.

Usage:
    shrink_sonar.py INPUT_DIR OUTPUT_DIR --center EPOCH --window SECONDS

Parses binary packets (BR marker + datestring) and keeps only packets
within the time window.
"""
import argparse
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path


def get_file_time(filename):
    """Extract time from filename like 20231109130613.dat"""
    base = os.path.basename(filename).replace('.dat', '')
    if len(base) == 14:
        try:
            return datetime.strptime(base, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
        except ValueError:
            return None
    return None


def parse_and_filter_sonar(filepath, start_dt, end_dt):
    """Parse sonar file and return filtered binary data."""
    with open(filepath, 'rb') as f:
        data = f.read()

    # Find all BR markers
    packets = []
    i = 0
    while i < len(data) - 2:
        if data[i:i+2] == b'BR':
            packets.append(i)
        i += 1

    if not packets:
        return None, 0, 0

    # For each packet, extract timestamp and determine if in window
    filtered_chunks = []
    kept = 0
    total = len(packets)

    for idx, pkt_start in enumerate(packets):
        # Determine packet end (start of next packet or end of file)
        if idx + 1 < len(packets):
            pkt_end = packets[idx + 1]
        else:
            pkt_end = len(data)

        # Extract datestring (26 bytes after BR marker)
        datestring_start = pkt_start + 2
        datestring_end = datestring_start + 26
        if datestring_end > len(data):
            continue

        datestring = data[datestring_start:datestring_end].decode('utf-8', 'replace')

        try:
            pkt_time = datetime.strptime(datestring, "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=timezone.utc)
        except ValueError:
            # Can't parse time, skip packet
            continue

        # Check if in window
        if start_dt <= pkt_time <= end_dt:
            filtered_chunks.append(data[pkt_start:pkt_end])
            kept += 1

    if not filtered_chunks:
        return None, kept, total

    return b''.join(filtered_chunks), kept, total


def main():
    parser = argparse.ArgumentParser(description="Shrink sonar files to time window")
    parser.add_argument("input_dir", help="Input directory containing .dat files")
    parser.add_argument("output_dir", help="Output directory")
    parser.add_argument("--center", type=float, required=True, help="Center epoch (Unix timestamp)")
    parser.add_argument("--window", type=float, default=300, help="Window size in seconds (default: 300)")
    args = parser.parse_args()

    center_dt = datetime.fromtimestamp(args.center, tz=timezone.utc)
    half_window = args.window / 2
    start_dt = center_dt - timedelta(seconds=half_window)
    end_dt = center_dt + timedelta(seconds=half_window)

    print(f"Center: {center_dt.isoformat()}")
    print(f"Window: {start_dt.isoformat()} to {end_dt.isoformat()}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Find all .dat files
    input_path = Path(args.input_dir)
    dat_files = sorted(input_path.glob("*.dat"))

    total_kept = 0
    total_packets = 0
    files_written = 0
    files_skipped = 0

    for dat_file in dat_files:
        file_time = get_file_time(dat_file.name)

        # Quick skip: if file is clearly outside window (with 2 min buffer)
        if file_time:
            if file_time < start_dt - timedelta(minutes=2):
                files_skipped += 1
                continue
            if file_time > end_dt + timedelta(minutes=2):
                files_skipped += 1
                continue

        # Parse and filter
        filtered_data, kept, total = parse_and_filter_sonar(dat_file, start_dt, end_dt)
        total_kept += kept
        total_packets += total

        if filtered_data:
            out_path = Path(args.output_dir) / dat_file.name
            with open(out_path, 'wb') as f:
                f.write(filtered_data)
            files_written += 1
            print(f"  {dat_file.name}: {kept}/{total} packets")

    print(f"\nResults:")
    print(f"  Files written: {files_written}")
    print(f"  Files skipped: {files_skipped}")
    print(f"  Total packets kept: {total_kept}/{total_packets}")


if __name__ == "__main__":
    main()
