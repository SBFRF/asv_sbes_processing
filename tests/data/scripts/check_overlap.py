#!/usr/bin/env python3
"""Validate time overlap across a folder of test data files.

Usage: check_overlap.py [DATA_ROOT]

Scans files for timestamps (HDF5/netCDF/RINEX/LLH) and reports per-file
min/max times and the global intersection. Exits 0 if intersection non-empty,
2 otherwise.
"""
import sys
import os
import re
from datetime import datetime, timezone

try:
    import h5py
except Exception:
    h5py = None

try:
    import xarray as xr
except Exception:
    xr = None

ISO_SLASH = re.compile(r"(\d{4}/\d{1,2}/\d{1,2}\s+\d{2}:\d{2}:\d{2}(?:\.\d+)?)")
ISO_T = re.compile(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?)")
RINEX_EPOCH = re.compile(r"^>\s*(\d{4})\s+(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\s+([0-9.+-]+)")


def parse_llh(path):
    times = []
    with open(path, 'r', errors='ignore') as fh:
        for line in fh:
            m = ISO_SLASH.search(line)
            if m:
                s = m.group(1)
                try:
                    if '.' in s:
                        dt = datetime.strptime(s, '%Y/%m/%d %H:%M:%S.%f').replace(tzinfo=timezone.utc)
                    else:
                        dt = datetime.strptime(s, '%Y/%m/%d %H:%M:%S').replace(tzinfo=timezone.utc)
                    times.append(dt.timestamp())
                except Exception:
                    pass
    return times


def parse_rinex(path):
    times = []
    with open(path, 'r', errors='ignore') as fh:
        for line in fh:
            m = RINEX_EPOCH.match(line)
            if m:
                try:
                    year, month, day, hour, minute, sec = m.groups()
                    dt = datetime(int(year), int(month), int(day), int(hour), int(minute), int(float(sec)), tzinfo=timezone.utc)
                    times.append(dt.timestamp())
                except Exception:
                    pass
    return times


def parse_h5(path):
    if h5py is None:
        return []
    times = []
    try:
        with h5py.File(path, 'r') as f:
            # special-case ppk layout: epochTime stored in ppk/block1_values
            try:
                if 'ppk' in f:
                    g = f['ppk']
                    if 'block1_items' in g and 'block1_values' in g:
                        items = [x.decode() if isinstance(x, bytes) else str(x) for x in g['block1_items'][:]]
                        if 'epochTime' in items:
                            idx = items.index('epochTime')
                            arr = g['block1_values'][:, idx]
                            arr = arr.flatten()
                            for v in arr:
                                try:
                                    times.append(float(v))
                                except Exception:
                                    pass
                            return times
            except Exception:
                # ignore and try fallback locations
                pass

            # fallback: look for common top-level time arrays
            for name in ('gps_time', 'time', 'timestamp', 'times'):
                if name in f:
                    try:
                        arr = f[name][:]
                        arr = arr.flatten()
                        for v in arr:
                            try:
                                times.append(float(v))
                            except Exception:
                                pass
                        break
                    except Exception:
                        continue
    except Exception:
        pass
    return times


def parse_netcdf(path):
    times = []
    if xr is None:
        return times
    try:
        ds = xr.open_dataset(path)
        if 'time' in ds.coords:
            arr = ds['time'].values
            # convert to POSIX
            import numpy as np
            try:
                epoch_seconds = (arr.astype('datetime64[s]').astype(int))
                times = [int(x) for x in epoch_seconds]
            except Exception:
                pass
        ds.close()
    except Exception:
        pass
    return times


def scan_file(path):
    ln = path.lower()
    if ln.endswith('.h5'):
        return parse_h5(path)
    if ln.endswith('.nc'):
        return parse_netcdf(path)
    if ln.endswith('.llh'):
        return parse_llh(path)
    if any(ln.endswith(ext) for ext in ('.23o', '.23p', '.23b', '.o', '.p', '.b')):
        return parse_rinex(path)
    # try to find ISO T format inside text files
    try:
        with open(path, 'r', errors='ignore') as fh:
            times = []
            for line in fh:
                m = ISO_T.search(line)
                if m:
                    s = m.group(1)
                    try:
                        dt = datetime.fromisoformat(s).replace(tzinfo=timezone.utc)
                        times.append(dt.timestamp())
                    except Exception:
                        pass
            return times
    except Exception:
        return []


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else 'tests/data/minimal/20231109'
    files = []
    for dp, _, fs in os.walk(root):
        for f in fs:
            files.append(os.path.join(dp, f))
    results = []
    for f in sorted(files):
        times = scan_file(f)
        if times:
            lo = min(times); hi = max(times)
            results.append((f, lo, hi))
        else:
            results.append((f, None, None))

    valid = [(lo, hi) for (_, lo, hi) in results if lo is not None]
    print('Per-file ranges:')
    for f, lo, hi in results:
        if lo is None:
            print('-', os.path.relpath(f, root), ': no timestamps found')
        else:
            print('-', os.path.relpath(f, root), ':', datetime.fromtimestamp(lo, tz=timezone.utc).isoformat(), '->', datetime.fromtimestamp(hi, tz=timezone.utc).isoformat())

    if not valid:
        print('\nNo files with timestamps found under', root)
        sys.exit(2)
    inter_lo = max(lo for lo, hi in valid)
    inter_hi = min(hi for lo, hi in valid)
    print('\nIntersection:')
    if inter_lo <= inter_hi:
        print(datetime.fromtimestamp(inter_lo, tz=timezone.utc).isoformat(), '->', datetime.fromtimestamp(inter_hi, tz=timezone.utc).isoformat())
        sys.exit(0)
    else:
        print('EMPTY (no single overlapping time across files)')
        sys.exit(2)


if __name__ == '__main__':
    main()
