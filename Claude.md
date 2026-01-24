# Claude.md - AI Assistant Guide for SWACSS ASV Processing

## Project Overview

This repository contains the operational workflow for the SWACSS (Shallow Water Acoustic and Current Survey System) project, specifically for processing data from the Yellowfin ASV (Autonomous Surface Vehicle). The primary workflow processes raw observation/navigation data with RINEX base station files into netCDF format using Post Processing Kinematic (PPK) positioning.

## Key Components

### Main Workflow
- **workflow_ppk.py**: Primary processing script that handles PPK processing for yellowfin ASV data
- **py2netCDF.py**: Handles conversion of processed data to netCDF format
- **yellowfinLib.py**: Library of utility functions specific to yellowfin ASV operations
- **mission_yaml_files.py**: Configuration management for mission parameters

### Core Technologies
- RTKlib for PPK positioning
- RINEX format for GPS/GNSS data
- netCDF for output data format
- CORS (Continuously Operating Reference Stations) for base station data
- Emlid GNSS receivers
- Imagenex S500 sonar

## Data Structure

The workflow expects a specific directory structure with date-formatted folders (YYYYMMDD):

```
YYYYMMDD/
├── CORS/          - CORS station PPK observations (.zip, .*o, .*n, .*sp3)
├── emlidRaw/      - Emlid RINEX data (*RINEX*.zip containing *o files)
├── nmeadata/      - NMEA GPS data (.dat text files)
├── s500/          - Sonar data (.dat binary files)
├── teleLogs/      - Tele-operation logs (optional)
└── figures/       - Generated output figures (auto-created)
```

## Important Considerations

### Critical Timing Issue
There was a computer clock change from ET to UTC around 7/10. Data prior to this has a hardcoded fix to UTC. Always verify timezone handling when working with data from different time periods.

### Data Processing Flow
1. Read raw GNSS observation files from Emlid
2. Process CORS base station data
3. Run RTKlib for PPK positioning
4. Read and time-shift sonar data using NMEA timestamps
5. Convert ellipsoid heights to NAVD88 using geoid file
6. Output to netCDF format

## Command Line Usage

```bash
python workflow_ppk.py -d /data/yellowfin/20240626
```

### Key Arguments
- `-d, --data_dir`: [REQUIRED] Directory of data to process (YYYYMMDD format)
- `-g, --geoid_file`: Binary geoid file for ellipsoid to NAVD88 conversion
- `-p, --make_pos`: Create position file using RTKlib (vs. external)
- `-v, --verbosity`: Debug level (1=Debug, 2=Info, 3=Warning)
- `--sonar_method`: S500 depth reading method for time-shifting
- `--rtklib_executable`: Path to RTKlib executable

## Development Guidelines

### When Modifying Code
1. **Timezone Handling**: Be extremely careful with any time-related modifications due to the ET/UTC transition
2. **File Path Handling**: Maintain compatibility with the expected folder structure
3. **Data Validation**: The code expects specific file formats - validate inputs carefully
4. **Geoid Conversions**: Height conversions are critical for accurate bathymetry

### Testing Considerations
- Always test with data from both before and after the timezone change (7/10)
- Verify RINEX file parsing works with different CORS stations
- Check that sonar time-shifting produces reasonable results
- Validate netCDF output format compliance

### Common Modifications
- Adding new sonar processing methods
- Supporting additional GNSS receiver formats
- Implementing new filtering algorithms
- Adding visualization capabilities
- Extending CORS station support

## File Format Notes

### RINEX Files
- Observation files (.*o): Raw GNSS measurements
- Navigation files (.*n): Satellite ephemeris data
- SP3 files (.*sp3): Precise satellite orbits

### Sonar Data
- Binary .dat files from Imagenex S500
- Requires time-shifting based on NMEA timestamps
- System clock vs. GNSS time synchronization is critical

### NMEA Data
- Text .dat files containing GPS sentences
- Used for system clock to GNSS time correlation
- Critical for accurate sonar timestamp correction

## Known Issues

1. **Timezone Transition**: Hardcoded UTC fix for pre-7/10 data
2. **Memory Usage**: Avoid using `-uall` flag with git status on large repos
3. **Date Validation**: MM-dd-YYYY subfolders must match root directory date

## Recent Changes

- Added satellite imagery background to final lon/lat scatter plots (#35)
- Fixed GPS path plotting and organized imports (#37, #30, #14)
- Fixed final data product formatting (#34)
- Added external tracing capabilities (#29)
- Added timeout to plotPlanViewOnArgus (#28)

## Getting Help

For questions about the workflow or to report issues, refer to the main README.md or contact the SWACSS project team.
