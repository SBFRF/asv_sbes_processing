# ASV SBES Processing - Test Suite

This directory contains comprehensive unit tests for the ASV SBES processing pipeline.

## Test Coverage

The test suite covers the following modules:

- **yellowfinLib.py**: Core data loading, processing, and conversion functions
- **py2netCDF.py**: NetCDF file creation and manipulation utilities
- **mission_yaml_files.py**: YAML metadata file generation (87% coverage)
- **workflow_ppk.py**: Main workflow and command-line interface

## Running Tests

### Install Test Dependencies

```bash
pip install -r requirements-test.txt
```

### Run All Tests

```bash
pytest tests/
```

### Run Tests with Coverage

```bash
pytest tests/ --cov=. --cov-report=term-missing --cov-report=html
```

View detailed coverage report:
```bash
open htmlcov/index.html
```

### Run Specific Test File

```bash
pytest tests/test_yellowfinLib.py -v
```

### Run Specific Test Class or Function

```bash
pytest tests/test_mission_yaml_files.py::TestMakeSummaryYaml -v
pytest tests/test_py2netCDF.py::test_import_yaml_file -v
```

## Test Organization

### conftest.py
Contains shared fixtures and mocks for external dependencies:
- Sample data fixtures (sonar, NMEA, PPK data)
- Temporary directory management
- Mock external libraries (testbedutils, pygeodesy)

### test_yellowfinLib.py
Tests for core processing library:
- Data loading functions (PPK, sonar, NMEA)
- Signal processing (filtering, cross-correlation)
- Coordinate transformations
- H5 file operations

### test_py2netCDF.py
Tests for netCDF utilities:
- NetCDF file creation
- YAML template parsing
- Variable and dimension handling
- Metadata management

### test_mission_yaml_files.py
Tests for YAML metadata generation:
- Mission summary creation
- Failure report generation
- User input validation
- File overwrite handling

### test_workflow_ppk.py
Tests for main workflow:
- Command-line argument parsing
- Verbosity level configuration
- Sonar method validation
- Parameter validation

## Test Categories

Tests are marked with the following categories:

- `@pytest.mark.unit`: Fast unit tests
- `@pytest.mark.integration`: Integration tests (slower)
- `@pytest.mark.slow`: Long-running tests

Run only unit tests:
```bash
pytest tests/ -m unit
```

## Mocking Strategy

External dependencies that require installation or authentication are mocked:

- **testbedutils**: FRF coordinate transformations
- **pygeodesy**: Geoid height calculations
- **rasterio**: GeoTIFF handling
- **wget**: File downloads

## Coverage Goals

Target coverage levels:
- Critical functions (data loading, transformations): 90%+
- UI functions (plotting, user input): 70%+
- Overall codebase: 80%+

## Continuous Integration

Tests run automatically on:
- Push to any branch
- Pull request creation
- Scheduled daily builds

See `.github/workflows/` for CI configuration.

## Writing New Tests

When adding new functionality:

1. Create corresponding test in appropriate test file
2. Use existing fixtures from conftest.py when possible
3. Mock external dependencies
4. Add docstrings explaining what is being tested
5. Use descriptive test names: `test_<function>_<scenario>`

Example:
```python
def test_butter_lowpass_filter_basic():
    """Test basic filtering operation"""
    signal = np.random.randn(100)
    filtered = yellowfinLib.butter_lowpass_filter(signal, cutoff=10, fs=100, order=4)
    assert len(filtered) == len(signal)
```

## Known Test Limitations

Some functions are difficult to test without real data:
- Binary sonar file parsing (requires actual .dat files)
- RINEX processing (requires RTKlib and GNSS data)
- Argus imagery fetching (requires network access)

For these functions, we test:
- Error handling
- Input validation
- That mocked versions work correctly

### Interactive Matplotlib Features

**Important:** The production workflow (`workflow_ppk.py`) uses matplotlib's TkAgg backend for interactive features:
- User clicking points on plots (`ginput()`)
- Interactive transect selection
- Real-time QA/QC plots with user interaction

**In CI/Testing:** The code gracefully falls back to the Agg (non-interactive) backend when TkAgg is unavailable:
```python
# workflow_ppk.py
try:
    matplotlib.use('TkAgg')  # Required for interactive plotting
except (ImportError, ModuleNotFoundError):
    matplotlib.use('Agg')    # Fallback for CI/testing
```

Additionally, CI sets `MPLBACKEND=Agg` environment variable to ensure non-interactive backend.

## Improving Test Coverage

### Current Coverage Status

**Current: 18.5% overall coverage with 46 passing tests**

| Module | Current Coverage | Potential Coverage | What's Needed |
|--------|-----------------|-------------------|---------------|
| mission_yaml_files.py | 87% ✅ | 90% | Minor additions |
| py2netCDF.py | 10% | 60% | NetCDF test fixes |
| workflow_ppk.py | 17% | 50% | Sample data + integration tests |
| yellowfinLib.py | 10% | 70% | **Sample data files** |

### Why Coverage is Low

Many tests are currently **ignored** (see `pytest.ini`) because they require real data files:

```ini
# Currently ignored in CI:
--ignore=tests/test_py2netCDF.py      # 17 tests (NetCDF/HDF5 issues)
--ignore=tests/test_yellowfinLib.py   # 30 tests (need sample data)
--ignore=tests/test_workflow_ppk.py   # 32 tests (need sample data)
```

These tests exist but aren't running because they need actual field data to work properly.

### Adding Sample Data to Enable More Tests

To increase coverage from **18%** → **60-80%**, add anonymized sample data files:

#### 1. Create Test Data Directory

```bash
mkdir -p tests/data/{sonar,nmea,rinex,ppk}
```

#### 2. Add Sample Files

**Sonar Data** (`tests/data/sonar/`):
```
tests/data/sonar/
├── sample_sonar.dat          # Binary sonar file (any size, even 100 pings)
└── README.md                 # Notes about data source/date
```

**NMEA GPS Data** (`tests/data/nmea/`):
```
tests/data/nmea/
├── sample_nmea.dat           # NMEA strings from GPS
└── README.md                 # Notes about data
```

**RINEX Data** (`tests/data/rinex/`):
```
tests/data/rinex/
├── rover.obs                 # Rover observation file
├── base.obs                  # Base station observation
├── nav.nav                   # Navigation file
└── README.md
```

**PPK Position Files** (`tests/data/ppk/`):
```
tests/data/ppk/
├── sample.pos                # RTKlib output file
└── README.md
```

#### 3. Anonymize Data (Important!)

Before adding real mission data:

```python
# Example: Offset coordinates to protect survey locations
import pandas as pd
import numpy as np

# Load real data
data = pd.read_csv('real_mission.pos')

# Offset lat/lon by random amount
offset_lat = np.random.uniform(-1, 1)
offset_lon = np.random.uniform(-1, 1)
data['lat'] += offset_lat
data['lon'] += offset_lon

# Save anonymized version
data.to_csv('tests/data/ppk/sample.pos', index=False)
```

**OR** use publicly available test datasets from other oceanographic projects.

#### 4. Update Test Fixtures

Once sample data is in place, update `tests/conftest.py`:

```python
import os

# Point to actual sample data instead of mocks
SAMPLE_DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

@pytest.fixture
def real_sonar_file():
    """Use actual sonar file instead of generated mock"""
    return os.path.join(SAMPLE_DATA_DIR, 'sonar', 'sample_sonar.dat')

@pytest.fixture
def real_nmea_file():
    """Use actual NMEA file"""
    return os.path.join(SAMPLE_DATA_DIR, 'nmea', 'sample_nmea.dat')
```

#### 5. Enable Previously Ignored Tests

In `pytest.ini`, remove the ignore lines:

```ini
# Before (ignoring tests):
addopts =
    --ignore=tests/test_py2netCDF.py
    --ignore=tests/test_yellowfinLib.py
    --ignore=tests/test_workflow_ppk.py

# After (running all tests with sample data):
addopts =
    -v
    --strict-markers
    --cov=.
```

#### 6. Run Tests with Sample Data

```bash
# Run previously-ignored tests
pytest tests/test_yellowfinLib.py -v

# Expected result with sample data:
# ✅ 20+ additional tests passing
# ✅ Coverage increases to 60-70%
```

### Expected Coverage Improvements

With sample data in place:

| Scenario | Tests Running | Coverage | Status |
|----------|---------------|----------|--------|
| **Current (no sample data)** | 46 tests | 18.5% | ✅ CI passing |
| **+ Basic sample data** | 76 tests | 45% | Most data loading tests pass |
| **+ Complete sample data** | 98 tests | 70% | Comprehensive coverage |
| **+ Integration tests** | 110+ tests | 80%+ | Full pipeline testing |

### Sample Data Requirements

**Minimum for meaningful coverage increase:**
- 1 sonar `.dat` file (any size, even 10 seconds of data)
- 1 NMEA `.dat` file with GPS strings
- 1 `.pos` file from RTKlib

**These can be:**
- Extracted from old missions
- Generated from public datasets
- Anonymized versions of real data
- Minimal synthetic data that follows the correct format

### Benefits of Adding Sample Data

✅ **Increases coverage from 18% → 60-80%**
✅ **Validates data parsing logic with real formats**
✅ **Catches edge cases** (malformed data, missing fields)
✅ **Enables integration testing** of full workflow
✅ **Provides examples** for new developers
✅ **Documents expected file formats** through working examples

### Alternative: Mock-Based Testing

If sample data cannot be provided, improve coverage with more sophisticated mocks:

```python
# tests/test_yellowfinLib.py
def test_load_sonar_with_mock_binary_format():
    """Test sonar loading with carefully crafted mock binary"""
    # Create binary data matching exact S500 format
    mock_data = create_s500_format_mock()  # Helper function
    # ... test parsing logic
```

This approach is more work but doesn't require real data files.

## Troubleshooting

### Import Errors

If you see `ModuleNotFoundError`, install missing dependencies:
```bash
pip install numpy pandas h5py netCDF4 scipy matplotlib pyyaml
```

### H5py/NetCDF Errors

Some systems have HDF5 library conflicts. Try:
```bash
pip install --no-cache-dir h5py netCDF4
```

### Coverage Not Accurate

Ensure you're running from the repository root:
```bash
cd /path/to/asv_sbes_processing
pytest tests/ --cov=.
```
