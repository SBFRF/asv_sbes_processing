# Test Data Directory

This directory contains test data files for unit and integration testing.

## Directory Structure

```
tests/data/
├── README.md                    # This file
├── mini_files.tar.gz            # Compressed archive of real survey data (~23 MB)
├── mini_files/                  # Extracted dataset (git-ignored, unpacked in CI/locally)
│   └── 20231109/
│       ├── CORS/                # CORS base-station RINEX files
│       ├── emlidRaw/            # Rover RINEX files from Emlid receiver
│       ├── nmeadata/            # NMEA GPS sentence files
│       ├── s500/                # Cerulean S500 sonar binary files
│       └── figures/             # Output figures from a previous run
├── create_minimal_test_data.py  # Script used to generate mini_files/ from full survey data
├── scripts/                     # Helper scripts for shrinking full survey data
│   ├── compute_median_time.py   # Compute median ISO timestamp from a folder
│   ├── shrink_h5.py             # Subset HDF5 files by time
│   ├── shrink_netcdf.py         # Subset netCDF files by time (requires xarray)
│   ├── shrink_nmea.py           # Subset NMEA text files by time window
│   ├── shrink_rinex.py          # Subset RINEX epoch blocks by time window
│   ├── shrink_pos.py            # Subset RTKlib .pos files by time window
│   ├── shrink_sonar.py          # Subset Cerulean S500 binary files by time
│   ├── make_minimal_dataset.sh  # Wrapper to run all shrinkers
│   ├── check_overlap.py         # Verify time overlap across all sensor files
│   └── test_data_overlap.py     # pytest checks for time-overlap in mini_files/
├── transect_global_attributes.yml  # Production YAML (unit tests)
└── transect_variables.yml          # Production YAML (unit tests)
```

## Test Data Types

### 1. YAML Templates (Production Files)
**Purpose:** Unit testing of YAML loading functions
**Status:** ✅ Complete
**Files:**
- `transect_global_attributes.yml` - Global netCDF attributes
- `transect_variables.yml` - Variable definitions and metadata

**Used by:**
- `test_py2netCDF.py::TestImportTemplateFile` - YAML loading tests

---

### 2. Mini Survey Dataset (`mini_files.tar.gz`)
**Purpose:** Integration testing of the full PPK workflow with real sensor data
**Status:** ✅ Complete
**Compressed size:** ~23 MB (tar.gz archive)
**Extracted size:** ~82 MB (raw sensor data)

This is a 5-minute window of real Yellowfin ASV field data recorded on 2023-11-09,
covering approximately 13:05–13:11 UTC. All four sensor sources overlap in time so
the complete processing pipeline (`workflow_ppk.py`) can run end-to-end in CI.

The dataset is stored as a compressed archive (`mini_files.tar.gz`) committed to git
to reduce clone/checkout time and repository size. The extracted `mini_files/`
directory is git-ignored.

**Contents (after extraction):**
| Sub-directory | Contents |
|---|---|
| `CORS/` | CORS base-station RINEX (*.23o, *.23n, *.23g, *.sp3) |
| `emlidRaw/` | Emlid rover RINEX + PPK position (*.23O, *.pos) |
| `nmeadata/` | NMEA GPS sentence text files (*.dat) |
| `s500/` | Cerulean S500 sonar binary files (*.dat) |

**Extraction:**

The archive is extracted automatically:
- **In CI:** The GitHub Actions workflow extracts `mini_files.tar.gz` before running
  tests (see `.github/workflows/tests.yml`).
- **Locally:** The `tests/conftest.py` auto-extracts the archive on first import if
  `tests/data/mini_files/` does not exist. You can also extract manually:
  ```bash
  tar -xzf tests/data/mini_files.tar.gz -C tests/data/
  ```

**Large-data strategy:** Only one compressed archive is committed to git.
Full survey days (hundreds of MB per day) should be stored externally (cloud bucket
/ artifact server) and fetched explicitly by jobs that need them. Do not commit
additional survey days to this repository—instead, add them as compressed archives
and update `.gitignore` accordingly.

---

## Generating or Regenerating `mini_files/`

Use `tests/data/create_minimal_test_data.py` together with the scripts in
`tests/data/scripts/` to extract a short time window from a full survey day.

```bash
# 1. Point the script at your full survey data directory
python tests/data/create_minimal_test_data.py --input /data/yellowfin/20231109 \
    --output tests/data/mini_files/20231109

# 2. Verify that all sensor files overlap in time
python tests/data/scripts/check_overlap.py tests/data/mini_files/20231109

# 3. Re-create the compressed archive
tar -czf tests/data/mini_files.tar.gz -C tests/data/ mini_files/
```

The script selects a window of ±300 seconds around the median PPK timestamp so
all sensors have stable, overlapping data. Override the window with the
`WINDOW_SECONDS` environment variable.

---

## Data Format Reference

### PPK File Format (RTKlib `.pos`)
```
% program   : rnx2rtkp ver.demo5 b34h
%  UTC                   latitude(deg) longitude(deg)  height(m)   Q  ns ...
2023/11/09 13:06:00.000   36.184162052  -75.751438333   -37.4834   1   6 ...
```

**Quality flags (Q):**
- 1 = RTK fix (cm-level accuracy)
- 2 = RTK float (dm-level accuracy)
- 5 = Single point positioning (m-level accuracy)

### NMEA File Format
Text files containing `$GNGGA` / `$GPGGA` GPS sentences, one sentence per line.

### Sonar File Format
Proprietary binary format from the Cerulean S500 single-beam echosounder
(packet ID 1308). Parsed by `yellowfinLib.loadSonar_s500_binary`.
