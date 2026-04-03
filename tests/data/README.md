# Test Data Directory

This directory contains test data files for unit and integration testing.

## Directory Structure

```
tests/data/
├── full_survey.pos              # Full PPK survey data (needs complete file)
├── create_minimal_test_data.py  # Extraction script
├── sample_survey_minimal/       # Minimal test data (auto-generated)
│   └── ppk/
│       └── 20231109.pos         # 1-minute PPK extract from median
├── transect_global_attributes.yml      # Production YAML (for unit tests)
└── transect_variables.yml              # Production YAML (for unit tests)
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

### 2. Minimal Survey Data (Integration Testing)
**Purpose:** Integration testing of full workflow with real survey data
**Status:** 🚧 In Progress
**Target size:** <1 MB total (manageable for git repository)

**Data extraction strategy:**
1. Find median timestamp in full survey
2. Extract 1-minute window (±30 seconds from median)
3. Preserve original coordinates (no anonymization)
4. Ensure time-synchronized data across all sensors (when added)

**Current files:**
- `sample_survey_minimal/ppk/20231109.pos` - PPK positions (RTK-processed GNSS)

**Future files (to be added):**
- `sample_survey_minimal/nmea/20231109.dat` - NMEA GPS sentences (TODO)
- `sample_survey_minimal/sonar/20231109.dat` - Sonar bathymetry (TODO)

---

## Current Status

### ✅ Completed
1. **Extraction script created** - `create_minimal_test_data.py`
2. **PPK extraction logic working** - Successfully extracts median time window
3. **Configuration updated** - 1-minute extraction window, no coordinate anonymization

### 🚧 In Progress
1. **Full PPK file needed** - Current `full_survey.pos` only has 3 data points
   - Should have: ~6,600 points (11 minutes at 10 Hz)
   - Currently has: 3 points (header + start/median/end)
   - Impact: Minimal file only has 1 point instead of ~600

2. **Target time window (to be updated with full file):**
   - Approximate median: `2023-11-09 12:38:38.900`
   - Expected extraction: ±30 seconds from median
   - Duration: 60 seconds
   - Expected output: ~600 data points

### 📋 Next Steps

1. **Provide complete PPK file** - Replace `full_survey.pos` with full survey data
2. **Run extraction** - Generate minimal PPK file with ~600 points
3. **Future:** Add NMEA and sonar data extraction (separate task)

---

## Extraction Process

### PPK Data Extraction

**Script:** `create_minimal_test_data.py`

**Process:**
1. Parse all timestamps from full survey
2. Sort timestamps and find median
3. Calculate 1-minute window around median (±30 seconds)
4. Extract all points within window
5. Write output with original header and coordinates intact

**Example output:**
```
Survey Information:
  Start:    2023-11-09 12:33:08.100000
  End:      2023-11-09 12:44:00.100000
  Duration: 652.0 seconds (10.9 minutes)
  Points:   6600 (estimated with full file)
  Rate:     10.1 Hz

Median Time: 2023-11-09 12:38:38.900000

Extracting window:
  Start: 2023-11-09 12:38:08.900000
  End:   2023-11-09 12:39:08.900000
  Extracted: ~600 points
```

**No anonymization:**
- Coordinates preserved at actual survey location
- Suitable for integration testing with real data

---

## Data Format Reference

### PPK File Format (RTKlib)
```
% program   : rnx2rtkp ver.demo5 b34h
% inp file  : [base station RINEX]
% inp file  : [rover RINEX]
% inp file  : [navigation file]
% inp file  : [precise ephemeris]
...
%  UTC                   latitude(deg) longitude(deg)  height(m)   Q  ns   sdn(m)   sde(m)   sdu(m)  sdne(m)  sdeu(m)  sdun(m) age(s)  ratio
2023/11/09 12:38:38.900   36.184162052  -75.751438333   -37.4834   1   6   0.0052   0.0037   0.0062  -0.0019   0.0017  -0.0032  -0.10   40.5
```

**Quality flags (Q):**
- 1 = RTK fix (cm-level accuracy)
- 2 = RTK float (dm-level accuracy)
- 5 = Single point positioning (m-level accuracy)

### NMEA File Format
TBD - Binary format with NMEA GPS sentences ($GNGGA, $GPGGA, etc.)

### Sonar File Format
TBD - Binary format from Cerulean S500 single-beam echosounder

---

## Time Synchronization Requirements

All sensor data must overlap in time to support workflow testing:

1. **PPK positions** - Define reference time window (base truth for positioning)
2. **NMEA GPS** - Must have data within same time window (raw GPS for comparison)
3. **Sonar data** - Must have data within same time window (bathymetry measurements)

The median extraction strategy ensures:
- Avoid survey startup/shutdown periods
- All sensors have stable data
- Representative of typical survey conditions
- Base station and rover had successful RTK convergence

---

## Usage

### Generate Minimal Test Data

```bash
# Run extraction script
python tests/data/create_minimal_test_data.py

# Output will be in sample_survey_minimal/
ls -lh tests/data/sample_survey_minimal/ppk/20231109.pos
```

### Expected Output Sizes

With complete files:
- PPK: ~60 KB (600 lines × ~100 bytes/line)
- NMEA: ~100 KB (estimated, 60 seconds of GPS data) - to be added
- Sonar: ~400 KB (estimated, 60 seconds of bathymetry) - to be added
- **Total: <600 KB** (well under 1 MB target)

---

## Integration Test Plan

Once minimal data is ready:

1. **Update `test_py2netCDF.py`:**
   - Remove mock fixtures
   - Use real data loaders
   - Re-enable integration tests
   - Test against actual survey workflow

2. **Create `test_workflow_integration.py`:**
   - Test full workflow from raw data to netCDF
   - Use minimal test data
   - Validate output format and content
   - Check coordinate transformations

3. **Update `pytest.ini`:**
   - Re-enable previously disabled tests
   - Add integration test markers
   - Configure for real data testing

4. **Achieve 80% code coverage:**
   - Current: ~60% (unit tests only)
   - Target: >80% (with integration tests)
