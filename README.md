# SWACSS
This codebase is focused on the operational workflow development for the SWACSS project. This is starting with the yellowfin ASV. 
`workflow_ppk.py` will process raw observation/navigation data from the yellowfin, with CORS or other RINEX base station files 
into netCDF using post processing kinematic (PPK) positioning. 

## How to run yellowfin workflow: 
`python workflow_ppk.py [args]`

with example argument 

`python workflow_ppk.py -d /data/yellowfin/20240626`
### available arguments
``` 
usage: PPK processing for yellowfin (V0.2) [-h] -d True [-g]
                                           [-p | --make_pos | --no-make_pos]
                                           [-v] [--sonar_method SONAR_METHOD]
                                           [--rtklib_executable RTKLIB_EXECUTABLE]

options:
  -h, --help            show this help message and exit
  -d True, --data_dir True
                        [REQUIRED] directory of data that are to be processed
  -g , --geoid_file     binary geoid file, required for conversion of
                        ellipsoid height to NAVD88
  -p, --make_pos, --no-make_pos
                        make posfile (True) using RTKlib or provide one
                        through external environment (false)
  -v , --verbosity      sets verbosity for debug, 1=Debug (most), 2=Info
                        (normal), 3=Warning (least)
  --sonar_method SONAR_METHOD
                        which s500 depth reading to use for time-shifting and
                        bottom reporting
  --rtklib_executable RTKLIB_EXECUTABLE
                        path for RTK_lib executable (required if --make-pos
                        flag assigned)
```
### Assumptions:
The code expects a `data_dir` argument passed which should be a folder in the `YYYYMMDD` format
the code expects the following folders [`CORS`, `emlidRaw`, `nmeadata`, `s500`]. 
It will make `figures` folder.  It's good practice to save the tele operation files in the `teleLogs` folder, but this script does not use those files (yet). 

The below image shows an example of processed folder and what it looks like at the top, with a generic folder with required subfolders below
![folder structure](docs/yellowfin_expected_folder_structure.png)

each folder is expected to have files within it, specifically:

__CORS__ - this folder is expected to have a zip file with PPK observations from a CORS station (including `.*sp3` if available).
If observation and navigation files are manually generated, the script will look for `.*o` and `.*n` files in this directory


__emlidRaw__ - this folder is expected to have `*RINEX*.zip` files containing (`*o` - observation files) (multiple ok) 

      
__nmeadata__ - this folder is expected to have `.dat` files (text) in them with nmea data fom the GNSS. this file is mainly used 
for logging the system clock (time stamp for sonar) and the GNSS time.  The `.dat` files can be in a folder with date 
convention of `MM-dd-YYYY` though this is checked against the root data directory to ensure the dates are the same   


__s500__ - this folder is expected to have `*.dat` files (binary) in them with sonar data.  The `.dat` files can be in a folder with date 
convention of `MM-dd-YYYY` though this is checked against the root data directory to ensure the dates are the same


## Generating POS Files: 
__configFile__ - this folder is expected to have `rtkpost.conf`, which defines the processing parameters used by RTKLIB. They can be modified and adjusted as needed

You can read more about the configuration file [here (pg. 109)](https://www.rtklib.com/prog/manual_2.4.2.pdf). Additionally, you can use RTKPost to determine the configurations desired and export (save) the file. 

__ref__ - this folder is expected to have both `igs20.atx` and `ngs_abs.pcv` which provide antenna calibration and phase center variation data. It is referenced within the `rtkpost.conf` file to apply antenna offset corrections

The PCV was found [here](https://www.ngs.noaa.gov/ANTCAL/LoadFile?file=ngs20.003) and saved as **ngs_abs.pcv**. The ATX file was found [here](https://files.igs.org/pub/station/general/).

# WARNING!!!!!
Sometime around 7/10 we changed Pi computer clock was changed from ET to UTC.  We should be able to compare system clock time to Nick's Field notes for start log to confirm if 8/16 survey was in ET or UTC. It's important to note that there is a hard/stupid fix of data prior to that to UTC.



