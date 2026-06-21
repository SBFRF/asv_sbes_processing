# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SWACSS (Shallow Water Coastal Survey System) - An ASV (Autonomous Surface Vehicle) based SBES (Single Beam Echo Sounder) processing pipeline. Processes raw sonar and navigation data from the Yellowfin ASV combined with CORS/RINEX base station files into netCDF format using PPK (Post-Processing Kinematic) positioning for bathymetric surveys.

## Common Commands

### Running the Pipeline
```bash
python workflow_ppk.py -d /path/to/data/YYYYMMDD
python workflow_ppk.py -d /data/yellowfin/20240626 -g ref/g2012bu0.bin -p --rtklib_executable ref/rnx2rtkp
```

### Testing
```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=. --cov-report=html

# Run specific test module
pytest tests/test_mission_yaml_files.py -v

# Run specific test class
pytest tests/test_yellowfinLib.py::TestButterLowpassFilter -v

# Run only fast tests
pytest tests/ -m "not slow"
```

Note: Several test files are ignored in CI (`test_py2netCDF.py`, `test_yellowfinLib.py`, `test_workflow_ppk.py`) because they require real field data files. See `pytest.ini` for current configuration.

### Linting
```bash
black . -l 100
pylint *.py --max-line-length=100
```

## Architecture

### Core Modules

1. **workflow_ppk.py** - CLI entry point that orchestrates the full PPK processing workflow
2. **yellowfinLib.py** - Core processing library containing:
   - Binary sonar parsing (`loadSonar_s500_binary`)
   - NMEA/GNSS parsing (`load_yellowfin_NMEA_files`, `read_emlid_pos`)
   - Signal processing (`butter_lowpass_filter`, `findTimeShiftCrossCorr`)
   - Coordinate transforms (`convertEllipsoid2NAVD88`, `is_local_to_FRF`)
   - Interactive plotting and transect selection
3. **py2netCDF.py** - NetCDF output generation from YAML templates
4. **mission_yaml_files.py** - Mission metadata generation (`make_summary_yaml`, `make_failure_yaml`)
5. **geoprocess.py** - Coordinate transformation utilities (FRF, State Plane, UTM)

### Data Flow
```
Input: CORS/ + emlidRaw/ + nmeadata/ + s500/
  → Parse binary sonar (loadSonar_s500_binary)
  → Parse GNSS timing (load_yellowfin_NMEA_files)
  → Load PPK positions (read_emlid_pos)
  → Time synchronization (findTimeShiftCrossCorr)
  → Filtering (butter_lowpass_filter)
  → Datum conversion (convertEllipsoid2NAVD88)
  → User validation (transectSelection)
  → Output: *.nc netCDF file
```

### Expected Data Directory Structure
```
YYYYMMDD/
├── CORS/       # RINEX base station files (*.zip with .*o, .*n, .*sp3)
├── emlidRaw/   # Rover RINEX files (*RINEX*.zip)
├── nmeadata/   # NMEA GPS text files (*.dat)
├── s500/       # Binary sonar data (*.dat)
└── figures/    # Generated output plots (created by script)
```

## Key Technical Details

- **Sonar format**: Cerulean S500 proprietary binary packets (packet_id 1308)
- **GNSS**: Emlid receivers with RTKlib for PPK processing
- **Coordinate systems**: FRF local, State Plane, lat/lon, UTM supported
- **Vertical datum**: NAVD88 via geoid file conversion from ellipsoid heights
- **Output format**: NetCDF with metadata from YAML templates (`yamlFile/`)

## Testing Infrastructure

- Uses pytest with fixtures defined in `tests/conftest.py`
- External dependencies mocked: `testbedutils`, `pygeodesy`
- Matplotlib uses `Agg` backend in CI (set `MPLBACKEND=Agg`)
- Test data extraction script: `tests/data/create_minimal_test_data.py`
- CI runs on Python 3.8-3.11

## Important Notes

- System clock timezone changed ~7/10 from ET to UTC - affects data prior to that date
- Interactive plots require TkAgg backend; use Agg for headless/CI environments
