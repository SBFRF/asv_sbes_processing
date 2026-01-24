#!/usr/bin/env python3
"""Shrink a netCDF / xarray dataset by selecting a time window.

This uses xarray when available, falling back to netCDF4+numpy.
"""
import argparse
import sys
from datetime import datetime, timezone

try:
    import xarray as xr
    has_xr = True
except Exception:
    has_xr = False

try:
    import netCDF4
    import numpy as np
except Exception:
    netCDF4 = None


def shrink_xarray(inpath, outpath, center_iso, window):
    ds = xr.open_dataset(inpath)
    if 'time' not in ds.coords:
        ds.to_netcdf(outpath)
        return
    center = np.datetime64(center_iso)
    lo = center - np.timedelta64(int(window), 's')
    hi = center + np.timedelta64(int(window), 's')
    sub = ds.sel(time=slice(lo, hi))
    sub.to_netcdf(outpath)


def main():
    p = argparse.ArgumentParser(description='Shrink netCDF by time window')
    p.add_argument('input')
    p.add_argument('output')
    p.add_argument('--center', required=True, help='center time ISO (e.g., 2023-11-09T12:34:56)')
    p.add_argument('--window', type=int, default=300, help='seconds +/- around center')
    args = p.parse_args()
    if has_xr:
        shrink_xarray(args.input, args.output, args.center, args.window)
    else:
        print('xarray not available; please install xarray to shrink netCDF files', file=sys.stderr)
        sys.exit(2)


if __name__ == '__main__':
    main()
