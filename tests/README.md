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
