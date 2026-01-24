#!/usr/bin/env python3
"""Shrink an HDF5 file by subsetting along a time-like first dimension.

Behavior:
- Finds a 1D dataset named 'time' or 'timestamp' at the file root.
- Determines indices within center +/- window_seconds and writes a new HDF5
  containing the same groups/datasets but with the first-dimension sliced
  for datasets that match the original time length.
"""
import argparse
import os
import sys

try:
    import h5py
    import numpy as np
except Exception as e:
    print("Missing dependency:", e, file=sys.stderr)
    sys.exit(2)


def find_time(ds, timevar=None):
    # Special-case: ppk layout stores labels and block values
    try:
        if 'ppk' in ds and 'ppk' in ds:
            g = ds['ppk']
            if 'block1_items' in g and 'block1_values' in g:
                try:
                    items = [x.decode() if isinstance(x, bytes) else str(x) for x in g['block1_items'][:]]
                    if 'epochTime' in items:
                        idx = items.index('epochTime')
                        vals = g['block1_values'][:, idx]
                        return 'ppk/block1_values[:,epochTime]', vals
                except Exception:
                    pass
    except Exception:
        pass

    names = ([timevar] if timevar else []) + ["time", "timestamp", "times", "gps_time", "pc_time_gga"]
    for name in names:
        if name and name in ds:
            try:
                return name, ds[name][:]
            except Exception:
                continue
    return None, None


def copy_subset(inpath, outpath, center_ts, window, timevar=None):
    import shutil
    with h5py.File(inpath, "r") as inf:
        time_name, time_arr = find_time(inf, timevar=timevar)
        if time_arr is None:
            # no time array found; fallback to copying the file
            shutil.copy2(inpath, outpath)
            return

        time_arr = np.asarray(time_arr).astype(float)
        lo = center_ts - window
        hi = center_ts + window
        mask = (time_arr >= lo) & (time_arr <= hi)
        if not mask.any():
            # nothing in window; pick the median index
            idx = len(time_arr) // 2
            idxs = np.array([idx])
        else:
            idxs = np.nonzero(mask)[0]

        n_time = len(time_arr)
        print(f"  Time variable: {time_name}, {n_time} total, {len(idxs)} in window")

        with h5py.File(outpath, "w") as outf:
            for key in inf:
                obj = inf[key]
                if isinstance(obj, h5py.Dataset):
                    try:
                        data = obj[:]
                    except Exception:
                        continue
                    shape = getattr(data, 'shape', ())
                    if shape and len(shape) >= 1:
                        # Check if first dimension matches time
                        if shape[0] == n_time:
                            new = data[idxs]
                            outf.create_dataset(key, data=new, dtype=new.dtype)
                        # Check if second dimension matches time (e.g., profile_data)
                        elif len(shape) >= 2 and shape[1] == n_time:
                            new = data[:, idxs]
                            outf.create_dataset(key, data=new, dtype=new.dtype)
                        else:
                            outf.create_dataset(key, data=data, dtype=data.dtype)
                    else:
                        outf.create_dataset(key, data=data, dtype=data.dtype)
                elif isinstance(obj, h5py.Group):
                    # Handle groups (like ppk/)
                    grp = outf.create_group(key)
                    for subkey in obj:
                        subobj = obj[subkey]
                        if isinstance(subobj, h5py.Dataset):
                            try:
                                data = subobj[:]
                            except Exception:
                                continue
                            shape = getattr(data, 'shape', ())
                            if shape and len(shape) >= 1 and shape[0] == n_time:
                                new = data[idxs]
                                grp.create_dataset(subkey, data=new, dtype=new.dtype)
                            else:
                                grp.create_dataset(subkey, data=data, dtype=data.dtype)


def main():
    p = argparse.ArgumentParser(description="Shrink HDF5 by time window")
    p.add_argument("input")
    p.add_argument("output")
    p.add_argument("--center", required=True, type=float, help="center epoch (seconds since epoch)")
    p.add_argument("--window", type=float, default=300.0, help="window seconds +/- around center")
    p.add_argument("--timevar", help="name of time variable inside HDF5 (e.g., gps_time)")
    args = p.parse_args()
    copy_subset(args.input, args.output, args.center, args.window, timevar=args.timevar)


if __name__ == "__main__":
    main()
