#!/usr/bin/env python3
"""Test that all mini test data files overlap in time.

This script examines all data sources and reports their time ranges,
then verifies there's a common overlapping window.

Usage:
    python test_data_overlap.py [DATA_DIR]

    DATA_DIR defaults to tests/data/mini_files/20231109
"""
import os
import re
import sys
import struct
from datetime import datetime, timezone
from pathlib import Path


def parse_pos_times(filepath):
    """Extract time range from RTKlib POS file."""
    times = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('%'):
                continue
            parts = line.split()
            if len(parts) >= 2:
                dt_str = parts[0] + " " + parts[1]
                try:
                    dt = datetime.strptime(dt_str, "%Y/%m/%d %H:%M:%S.%f")
                    times.append(dt.replace(tzinfo=timezone.utc).timestamp())
                except ValueError:
                    pass
    return times


def parse_nmea_times(dirpath):
    """Extract time range from NMEA files using GNSS time in GGA sentences."""
    times = []
    gga_pattern = re.compile(r'\$\w{2}GGA,(\d{6})\.?\d*,')

    for dat_file in sorted(Path(dirpath).glob("*.dat")):
        # Get date from filename: 20231109HHMMSS.dat
        fname = dat_file.stem
        if len(fname) >= 8:
            date_str = fname[:8]  # YYYYMMDD
        else:
            continue

        with open(dat_file, 'r', errors='ignore') as f:
            for line in f:
                match = gga_pattern.search(line)
                if match:
                    hhmmss = match.group(1)
                    try:
                        time_str = f"{date_str} {hhmmss[:2]}:{hhmmss[2:4]}:{hhmmss[4:6]}"
                        dt = datetime.strptime(time_str, "%Y%m%d %H:%M:%S")
                        times.append(dt.replace(tzinfo=timezone.utc).timestamp())
                    except ValueError:
                        pass
    return times


def parse_sonar_times(dirpath):
    """Extract time range from S500 sonar binary files."""
    times = []

    for dat_file in sorted(Path(dirpath).glob("*.dat")):
        with open(dat_file, 'rb') as f:
            data = f.read()

        # Find BR markers and extract timestamps
        i = 0
        while i < len(data) - 30:
            if data[i:i+2] == b'BR':
                datestring = data[i+2:i+28].decode('utf-8', 'replace')
                try:
                    dt = datetime.strptime(datestring, "%Y-%m-%d %H:%M:%S.%f")
                    times.append(dt.replace(tzinfo=timezone.utc).timestamp())
                except ValueError:
                    pass
                i += 30
            else:
                i += 1
    return times


def parse_rinex_times(filepath):
    """Extract time range from RINEX observation file (2.x or 3.x)."""
    times = []

    # RINEX 2.x epoch pattern
    rinex2_pattern = re.compile(r'^\s+(\d{2})\s+(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\s+(\d+\.\d+)\s+\d+\s+\d+')

    with open(filepath, 'r', errors='ignore') as f:
        in_header = True
        for line in f:
            if 'END OF HEADER' in line:
                in_header = False
                continue
            if in_header:
                continue

            # Check RINEX 3.x format (starts with >)
            if line.startswith('>'):
                parts = line[1:].strip().split()
                if len(parts) >= 6:
                    try:
                        year, month, day, hour, minute = map(int, parts[:5])
                        sec = int(float(parts[5]))
                        dt = datetime(year, month, day, hour, minute, sec, tzinfo=timezone.utc)
                        times.append(dt.timestamp())
                    except (ValueError, IndexError):
                        pass
            else:
                # Check RINEX 2.x format
                match = rinex2_pattern.match(line)
                if match:
                    yy, mo, dd, hh, mm, ss = match.groups()
                    year = 2000 + int(yy) if int(yy) < 80 else 1900 + int(yy)
                    try:
                        dt = datetime(year, int(mo), int(dd), int(hh), int(mm), int(float(ss)), tzinfo=timezone.utc)
                        times.append(dt.timestamp())
                    except ValueError:
                        pass
    return times


def format_time(ts):
    """Format timestamp as ISO string."""
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%H:%M:%S")


def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "tests/data/mini_files/20231109"

    if not os.path.isdir(data_dir):
        print(f"ERROR: Directory not found: {data_dir}")
        sys.exit(1)

    print(f"Checking data overlap in: {data_dir}\n")

    results = {}

    # 1. POS file
    pos_files = list(Path(data_dir).rglob("*.pos"))
    pos_files = [f for f in pos_files if 'events' not in f.name]
    if pos_files:
        all_pos_times = []
        for pf in pos_files:
            times = parse_pos_times(pf)
            if times:
                all_pos_times.extend(times)
                print(f"POS ({pf.name}): {format_time(min(times))} - {format_time(max(times))} ({len(times)} records)")
        if all_pos_times:
            results['POS'] = (min(all_pos_times), max(all_pos_times), len(all_pos_times))

    # 2. NMEA files
    nmea_dir = Path(data_dir) / "nmeadata"
    if nmea_dir.exists():
        times = parse_nmea_times(nmea_dir)
        if times:
            results['NMEA'] = (min(times), max(times), len(times))
            print(f"NMEA: {format_time(min(times))} - {format_time(max(times))} ({len(times)} records)")

    # 3. Sonar files
    sonar_dir = Path(data_dir) / "s500"
    if sonar_dir.exists():
        times = parse_sonar_times(sonar_dir)
        if times:
            results['Sonar'] = (min(times), max(times), len(times))
            print(f"Sonar: {format_time(min(times))} - {format_time(max(times))} ({len(times)} records)")

    # 4. CORS RINEX observation file
    cors_obs = list(Path(data_dir).rglob("*.23o")) + list(Path(data_dir).rglob("*.??o"))
    if cors_obs:
        # Use the one in CORS/ directory (not ncdu313 subfolder to avoid duplicates)
        cors_file = [f for f in cors_obs if 'ncdu313' not in str(f.parent)]
        if cors_file:
            times = parse_rinex_times(cors_file[0])
            if times:
                results['CORS'] = (min(times), max(times), len(times))
                print(f"CORS ({cors_file[0].name}): {format_time(min(times))} - {format_time(max(times))} ({len(times)} epochs)")

    # 5. Rover RINEX observation file
    rover_obs = list(Path(data_dir).rglob("*.23O"))
    if rover_obs:
        times = parse_rinex_times(rover_obs[0])
        if times:
            results['Rover RINEX'] = (min(times), max(times), len(times))
            print(f"Rover RINEX ({rover_obs[0].name}): {format_time(min(times))} - {format_time(max(times))} ({len(times)} epochs)")

    # Calculate overlap
    print("\n" + "="*60)

    if len(results) < 2:
        print("ERROR: Not enough data sources found to check overlap")
        sys.exit(1)

    # Find intersection
    overlap_start = max(r[0] for r in results.values())
    overlap_end = min(r[1] for r in results.values())

    print(f"\nData sources found: {len(results)}")
    for name, (start, end, count) in results.items():
        print(f"  {name}: {format_time(start)} - {format_time(end)} ({count} records)")

    print(f"\nOverlap window:")
    if overlap_start <= overlap_end:
        duration = overlap_end - overlap_start
        print(f"  {format_time(overlap_start)} - {format_time(overlap_end)}")
        print(f"  Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
        print(f"  Center epoch: {(overlap_start + overlap_end) / 2:.0f}")
        print(f"\n✓ All data sources overlap!")

        # Show how much of each source is in the overlap
        print(f"\nRecords within overlap window:")
        for name, (start, end, count) in results.items():
            # Rough estimate based on uniform distribution
            if end > start:
                frac = min(1.0, max(0.0, (overlap_end - overlap_start) / (end - start)))
                est_records = int(count * frac)
                print(f"  {name}: ~{est_records} records")

        sys.exit(0)
    else:
        print(f"  NO OVERLAP!")
        print(f"  Gap: {format_time(overlap_end)} to {format_time(overlap_start)}")
        print(f"  Gap duration: {overlap_start - overlap_end:.1f} seconds")
        print(f"\n✗ Data sources do NOT overlap!")
        sys.exit(1)


if __name__ == "__main__":
    main()
