#!/usr/bin/env python3
"""
Extract minimal test data from median time point of full survey.

This script processes a full PPK survey file and extracts a 30-second window
centered on the median time point. The extracted data is anonymized by shifting
coordinates +1.0 degrees in latitude and longitude.

Usage:
    python create_minimal_test_data.py

Input:
    tests/data/full_survey.pos - Full PPK survey data (RTKlib format)

Output:
    tests/data/sample_survey_minimal/ppk/20231109.pos - Minimal test data (~300 lines)
"""

from pathlib import Path
from datetime import datetime, timedelta
import sys


def parse_ppk_timestamp(line):
    """
    Parse timestamp from PPK data line.

    Args:
        line: PPK data line in format "2023/11/09 12:38:40.123  ..."

    Returns:
        datetime object or None if parsing fails
    """
    parts = line.split()
    if len(parts) < 2:
        return None

    try:
        date_str = parts[0]  # 2023/11/09
        time_str = parts[1]  # 12:38:40.123
        dt = datetime.strptime(f"{date_str} {time_str}", "%Y/%m/%d %H:%M:%S.%f")
        return dt
    except (ValueError, IndexError):
        return None


def anonymize_ppk_line(line, lat_offset=1.0, lon_offset=1.0):
    """
    Anonymize a PPK data line by shifting coordinates.

    Args:
        line: Original PPK data line
        lat_offset: Latitude offset in degrees (default: +1.0)
        lon_offset: Longitude offset in degrees (default: +1.0)

    Returns:
        Anonymized line with shifted coordinates
    """
    parts = line.split()
    if len(parts) < 4:
        return line

    # PPK format: date time lat lon height Q ns sdn sde sdu sdne sdeu sdun age ratio
    # Indices:    0    1    2   3   4      5 6  7   8   9   10   11   12   13  14
    try:
        # Shift latitude (index 2) and longitude (index 3)
        parts[2] = f"{float(parts[2]) + lat_offset:13.9f}"
        parts[3] = f"{float(parts[3]) + lon_offset:14.9f}"

        # Reconstruct line with original spacing
        return f"{parts[0]} {parts[1]} {parts[2]} {parts[3]} {parts[4]:11s} {parts[5]:3s} {parts[6]:3s} {parts[7]:8s} {parts[8]:8s} {parts[9]:8s} {parts[10]:8s} {parts[11]:8s} {parts[12]:8s} {parts[13]:6s} {parts[14]}\n"
    except (ValueError, IndexError):
        return line


def extract_minimal_ppk(input_file, output_file, duration_seconds=30,
                       lat_offset=1.0, lon_offset=1.0):
    """
    Extract minimal test data from median time point of PPK survey.

    Args:
        input_file: Path to full survey PPK file
        output_file: Path to output minimal PPK file
        duration_seconds: Duration of extraction window (default: 30 seconds)
        lat_offset: Latitude anonymization offset (default: +1.0 degrees)
        lon_offset: Longitude anonymization offset (default: +1.0 degrees)

    Returns:
        tuple: (start_time, end_time) of extracted window
    """
    print(f"Reading {input_file}...")

    # Read file
    with open(input_file, 'r') as f:
        lines = f.readlines()

    # Separate header and data
    header = []
    data_lines = []
    timestamps = []

    for line in lines:
        if line.startswith('%'):
            header.append(line)
        else:
            # Skip empty lines
            if line.strip():
                ts = parse_ppk_timestamp(line)
                if ts:
                    data_lines.append(line)
                    timestamps.append(ts)

    if not timestamps:
        print("ERROR: No valid timestamps found in file")
        return None, None

    # Find median
    timestamps_sorted = sorted(timestamps)
    median_idx = len(timestamps_sorted) // 2
    median_time = timestamps_sorted[median_idx]

    # Survey info
    start_survey = timestamps_sorted[0]
    end_survey = timestamps_sorted[-1]
    duration = (end_survey - start_survey).total_seconds()

    print(f"\nSurvey Information:")
    print(f"  Start:    {start_survey}")
    print(f"  End:      {end_survey}")
    print(f"  Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
    print(f"  Points:   {len(timestamps)}")
    print(f"  Rate:     {len(timestamps)/duration:.1f} Hz")
    print(f"\nMedian Time: {median_time}")

    # Extract ±duration_seconds/2 around median
    half_duration = duration_seconds / 2.0
    start_time = median_time - timedelta(seconds=half_duration)
    end_time = median_time + timedelta(seconds=half_duration)

    print(f"\nExtracting window:")
    print(f"  Start: {start_time}")
    print(f"  End:   {end_time}")
    print(f"  Duration: {duration_seconds} seconds")

    # Extract and anonymize
    extracted = []
    for i, ts in enumerate(timestamps):
        if start_time <= ts <= end_time:
            line = data_lines[i]
            anonymized = anonymize_ppk_line(line, lat_offset, lon_offset)
            extracted.append(anonymized)

    print(f"  Extracted: {len(extracted)} points")

    # Create output directory
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Write output
    with open(output_file, 'w') as f:
        f.writelines(header)
        if extracted:
            f.writelines(extracted)

    print(f"\nSaved to: {output_file}")
    print(f"Coordinates anonymized: +{lat_offset}° lat, +{lon_offset}° lon")

    return start_time, end_time


def main():
    """Main entry point"""
    # File paths
    input_file = Path(__file__).parent / "full_survey.pos"
    output_file = Path(__file__).parent / "sample_survey_minimal" / "ppk" / "20231109.pos"

    # Check if input exists
    if not input_file.exists():
        print(f"ERROR: Input file not found: {input_file}")
        print("\nPlease ensure the full survey PPK file is at:")
        print(f"  {input_file.absolute()}")
        sys.exit(1)

    # Extract minimal data
    start_time, end_time = extract_minimal_ppk(
        input_file,
        output_file,
        duration_seconds=30,
        lat_offset=1.0,
        lon_offset=1.0
    )

    if start_time and end_time:
        print("\n" + "="*70)
        print("Next Steps:")
        print("="*70)
        print(f"\nTime window for extracting other sensor data:")
        print(f"  Start: {start_time}")
        print(f"  End:   {end_time}")
        print("\nPlease provide the following files for the same survey:")
        print("  1. NMEA GPS data (.dat file)")
        print("  2. Sonar bathymetry data (.dat file)")
        print("\nThese will be processed to extract matching time windows.")
    else:
        print("\nERROR: Extraction failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
