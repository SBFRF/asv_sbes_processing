import os
import sys
import matplotlib
from scipy import interpolate, signal

import py2netCDF

matplotlib.use('TkAgg')
import yellowfinLib
import datetime as DT
from matplotlib import pyplot as plt
import numpy as np
import h5py
import pandas as pd
import glob
import zipfile
import tqdm
from testbedutils import geoprocess
import argparse, logging

from mission_yaml_files import make_summary_yaml, make_failure_yaml

sonar_methods = ['default', 'instant', 'smoothed', 'qaqc']

__version__ = 0.4
def parse_args(__version__):
    parser = argparse.ArgumentParser(f"PPK processing for yellowfin (V{__version__})", add_help=True)
    # datadir, geoid, makePos = True, verbose = 1
    # Command-Line Interface: (REQUIRED) Flags
    parser.add_argument('-d', '--data_dir', type=str, metavar=True,
                        help="[REQUIRED] directory of data that are to be processed",
                        required=True)



    # Command-Line Interface: (OPTIONAL) Flags
    parser.add_argument('-g', '--geoid_file', type=str, default='ref/g2012bu0.bin', metavar='',
                        help="binary geoid file, required for conversion of ellipsoid height to NAVD88")
    parser.add_argument('-p', '--make_pos', action=argparse.BooleanOptionalAction, type=bool, default=False,
                        help="make posfile (True) using RTKlib or provide one through external environment (false)")
    parser.add_argument('-v', '--verbosity', type=int, default=2, metavar='',
                        help='sets verbosity for debug, 1=Debug (most), 2=Info (normal), 3=Warning (least)')
    parser.add_argument('--sonar_method', type=str, default='default',
                        help="which s500 depth reading to use for time-shifting and bottom reporting, available "
                             f"are {sonar_methods}. default uses instant depth for time syncing and"
                             " smooth depths for final bathy out; 'smooth' uses smoothed values for both; 'instant' "
                             "uses instant values for both; 'qaqc' uses hand-traced values for both "
                             "(assumes sonar data h5 has been traced in sonar_qaqc tool externally)")
    parser.add_argument('--rtklib_executable', type=str, default='ref/rnx2rtkp',
                        help="path for RTK_lib executable (required if --make-pos flag assigned)")
    parser.add_argument("--ppk_quality_threshold",  type=int, default=1,
                        help="this is a quality threshold 1: Fixed, 2: Float, 4:DGPS, 5: single -- see appendix B for "
                             "more details: https://rtkexplorer.com/pdfs/manual_demo5.pdf  ")
    parser.add_argument("--instant_sonar_confidence", type=int, default=99,
                        help="This is a filter threshold for instantaneous confidence for each sonar ping")
    parser.add_argument("--smoothed_sonar_confidence", type=int, default=60,
                        help="This is a filter threshold for smoothed confidence from the sonar")

    return parser.parse_args()


def verbosity_conversion(verbose: int):
    if verbose == 1:
        logging.basicConfig(level=logging.DEBUG)
    elif verbose == 2:
        logging.basicConfig(level=logging.INFO)
    elif verbose == 3:
        logging.basicConfig(level=logging.WARN)
    else:
        raise EnvironmentError('logging verbosity is wrong!')

def main(datadir, geoid, makePos=True, verbose=2, sonar_method='default', rtklib_executable_path = 'ref/rnx2rtkp',
         ppk_quality_threshold=1, instant_sonar_confidence = 99, smoothed_sonar_confidence = 60):

    verbosity_conversion(verbose)
    antenna_offset = 0.25  # meters between the antenna phase center and sounder head - default for yellowfin

    #  date that Pi computer was changed to UTC time (will adjust timezone manually before this date)
    yellowfin_clock_reset_date = DT.datetime(2023, 7,10)  # do not adjust this date!
    if sonar_method == 'default':
        bathy_report = 'smoothed'
        time_sync = 'instant'
        sonar_confidence = smoothed_sonar_confidence
    elif sonar_method == 'instant':
        sonar_confidence = instant_sonar_confidence
        bathy_report = sonar_method
        time_sync = sonar_method
    elif sonar_method == 'smoothed':
        sonar_confidence = smoothed_sonar_confidence
        bathy_report = sonar_method
        time_sync = sonar_method
    elif sonar_method == 'qaqc':
        sonar_confidence = 100 # not used in filtering data, we assume 100% confidence in human tracing
        bathy_report = sonar_method
        time_sync = sonar_method
    else:
        raise ValueError(f'acceptable sonar methods include {sonar_methods}')

    logging.info(f"procesing prameters:  sonar time sync method {time_sync}")
    logging.info(f"procesing prameters:  bathy sonar method {bathy_report}")
    logging.info(f"input folder: {datadir}")
    logging.info(f"ppk_quality_threshold: {ppk_quality_threshold}")
    logging.info(f"sonar_confidence: {sonar_confidence} %")
    ####################################################################################################################

    if datadir.endswith('/'): datadir = datadir[:-1]
    ## Define all paths for the workflow
    timeString = os.path.basename(datadir)
    plotDir = os.path.join(datadir, 'figures')
    os.makedirs(plotDir, exist_ok=True)  # make folder structure if its not already made
    argusGeotiff = yellowfinLib.threadGetArgusImagery(DT.datetime.strptime(timeString, '%Y%m%d') +
                                                      DT.timedelta(hours=14),
                                                      ofName=os.path.join(plotDir, f'Argus_{timeString}.tif'),)

    # sonar data
    fpathSonar = os.path.join(datadir, 's500')  # reads sonar from here
    saveFnameSonar = os.path.join(datadir, f'{timeString}_sonarRaw.h5')  # saves sonar file here

    # NMEA data from sonar, this is not Post Processed Kinematic (PPK) data.  It is used for only cursory or
    # introductory look at the data
    fpathGNSS = os.path.join(datadir, 'nmeadata')  # load NMEA data from this location
    saveFnameGNSS = os.path.join(datadir, f'{timeString}_gnssRaw.h5')  # save nmea data to this location

    # RINEX data
    # look for all subfolders with RINEX in the folder name inside the "datadir" emlid ppk processor
    fpathEmlid = os.path.join(datadir, 'emlidRaw')
    saveFnamePPK = os.path.join(datadir, f'{timeString}_ppkRaw.h5')

    logging.debug(f"saving intermediate files for sonar here: {saveFnameSonar}")
    logging.debug(f"saving intermediate files for sonar here: {saveFnamePPK}")
    logging.debug(f"saving intermediate files for GNSS here: {saveFnameGNSS}")
    ## load files
    if not os.path.isfile(saveFnameSonar):
        yellowfinLib.loadSonar_s500_binary(fpathSonar, outfname=saveFnameSonar, verbose=verbose)
    else:
        logging.info(f'Skipping {saveFnameSonar}')
    # then load NMEA files
    if not os.path.isfile(saveFnameGNSS):  # if we've already processed the GNSS file
        yellowfinLib.load_yellowfin_NMEA_files(fpathGNSS, saveFname=saveFnameGNSS,
                                           plotfname=os.path.join(plotDir, 'GPSpath_fromNMEAfiles.png'),
                                           verbose=verbose)
    else:
        logging.info(f'Skipping {saveFnameGNSS}')

    if not os.path.isfile(saveFnamePPK):
        if makePos == True:
            # find folders with raw rinex
            rover_rinex_zip_files = glob.glob(os.path.join(fpathEmlid, '*RINEX*.zip'))
            # identify the nav/obs file
            base_zip_files = glob.glob(os.path.join(datadir, 'CORS', '*.zip'))

            if np.size(base_zip_files) == 1: # if there's zip file it's from CORS
                base_zip_files = base_zip_files[0]
                with zipfile.ZipFile(base_zip_files, 'r') as zip_ref:
                    zip_ref.extractall(path=base_zip_files[:-4])
                cors_search_path_obs = os.path.join(os.path.splitext(base_zip_files)[0], '*o')
                cors_search_path_nav = os.path.join(os.path.splitext(base_zip_files)[0], '*n')
                cors_search_path_sp3 = os.path.join(os.path.splitext(base_zip_files)[0], '*sp3')
            elif np.size(base_zip_files) >1: # if there's more than one zip file
                raise EnvironmentError('There are too many zip files in the CORS folder to extract')
            else:
                cors_search_path_obs = os.path.join(datadir, 'CORS', '*o')
                cors_search_path_nav = os.path.join(datadir, 'CORS', '*n')
                cors_search_path_sp3 = os.path.join(datadir, 'CORS', '*sp3')

            base_obs_fname = glob.glob(cors_search_path_obs)[0]
            base_nav_file = glob.glob(cors_search_path_nav)[0]
            base_sp3_list = glob.glob(cors_search_path_sp3)
            if np.size(base_sp3_list) == 1:
                sp3_fname = base_sp3_list[0]
            else:
                sp3_fname = ''

            # unzip all the rinex Files
            for ff in rover_rinex_zip_files:
                with zipfile.ZipFile(ff, 'r') as zip_ref:
                    zip_ref.extractall(path=ff[:-4])
                # identify and process rinex to Pos files
                flist_rinex = glob.glob(ff[:-4] + "/*")
                rover_obs_fname = flist_rinex[np.argwhere([i.endswith('O') for i in flist_rinex]).squeeze()]
                outfname = os.path.join(os.path.dirname(rover_obs_fname), os.path.basename(flist_rinex[0])[:-3] + "pos")
                # use below if the rover nav file is the right call
                yellowfinLib.makePOSfileFromRINEX(roverObservables=rover_obs_fname, baseObservables=base_obs_fname, navFile=base_nav_file,
                                                  outfname=outfname, executablePath=rtklib_executable_path, sp3=sp3_fname)

        # Now find all the folders that have ppk data in them (*.pos files in folders that have "raw" in them)
        # now identify the folders that have rinex in them
        fldrlistPPK = []  # initalize list for appending RINEX folder in
        [fldrlistPPK.append(os.path.join(fpathEmlid, fname)) for fname in os.listdir(fpathEmlid) if
         'raw' in fname and '.zip' not in fname]

        logging.warning('load PPK pos files ---- THESE ARE WGS84!!!!!!!!!!!!!!')
        try:
            T_ppk = yellowfinLib.loadPPKdata(fldrlistPPK)
            T_ppk.to_hdf(saveFnamePPK, 'ppk')  # now save the h5 intermediate file
        except KeyError:
            raise FileExistsError("the pos file hasn't been loaded, manually produce or turn on RTKlib processing")
    else:
        logging.info(f'Skipping {saveFnamePPK}')
        T_ppk = pd.read_hdf(saveFnamePPK)

    # 1. time in seconds to adjust to UTC from ET (varies depending on time of year!!!)
    if (T_ppk['datetime'].iloc[0].replace(tzinfo=None) < yellowfin_clock_reset_date) & (
            int(T_ppk['datetime'].iloc[0].day_of_year) > 71) & (int(T_ppk['datetime'].iloc[0].day_of_year) < 309):
        ET2UTC = 5 * 60 * 60
        logging.warning(" I'm using a 'dumb' conversion from ET to UTC")
    elif (T_ppk['datetime'].iloc[0].replace(tzinfo=None) < yellowfin_clock_reset_date) & (
            int(T_ppk['datetime'].iloc[0].day_of_year) < 71) & (int(T_ppk['datetime'].iloc[0].day_of_year) > 309):
        ET2UTC = 4 * 60 * 60
        logging.warning(" I'm using a 'dumb' conversion from ET to UTC")
    else:
        ET2UTC = 0  # time's already in UTC


    # 6.2: load all files we created in previous steps
    sonarData = yellowfinLib.load_h5_to_dictionary(saveFnameSonar)
    payloadGpsData = yellowfinLib.load_h5_to_dictionary(saveFnameGNSS)  # this is used for the pc time adjustement
    T_ppk = pd.read_hdf(saveFnamePPK)

    # Adjust GNSS time by the Leap Seconds https://www.cnmoc.usff.navy.mil/Our-Commands/United-States-Naval-Observatory/Precise-Time-Department/Global-Positioning-System/USNO-GPS-Time-Transfer/Leap-Seconds/
    # T_ppk['epochTime'] = T_ppk['epochTime'] - 18  # 18 is leap second adjustment
    # T_ppk['datetime'] = T_ppk['datetime'] - DT.timedelta(seconds=18)  # making sure both are equal
    # commented because the cross-correlation should account for this anyway (?)

    # convert raw ellipsoid values from satellite measurement to that of a vertical datum.  This uses NAVD88 [m] NAD83
    T_ppk['GNSS_elevation_NAVD88'] = yellowfinLib.convertEllipsoid2NAVD88(T_ppk['lat'], T_ppk['lon'], T_ppk['height'],
                                                                          geoidFile=geoid)
    # 6.3: now plot my time offset between GPS and sonar
    pc_time_off = payloadGpsData['pc_time_gga'] + ET2UTC - payloadGpsData['gps_time']

    ofname = os.path.join(plotDir, 'clockOffset.png')
    # TODO pull this figure out to a function
    yellowfinLib.qaqc_time_offset_determination(ofname, pc_time_off)
    # 6.4 Use the cerulean instantaneous bed detection since not sure about delay with smoothed
    # adjust time of the sonar time stamp with timezone shift (ET -> UTC) and the timeshift between the computer and GPS
    sonarData['time'] = sonarData['time'] + ET2UTC - np.median(pc_time_off)  # convert to UTC
    if sonar_method == 'default':
        sonar_range = sonarData['this_ping_depth_m']
        qualityLogic = sonarData['this_ping_depth_measurement_confidence'] > instant_sonar_confidence
    elif sonar_method == 'smooth':
        sonar_range = sonarData['smooth_depth_m']
        qualityLogic = sonarData['smoothed_depth_measurement_confidence'] > smoothed_sonar_confidence
    elif sonar_method == 'instant':
        sonar_range = sonarData['this_ping_depth_m']
        qualityLogic = sonarData['this_ping_depth_measurement_confidence'] > instant_sonar_confidence
    elif sonar_method == 'qaqc':
        sonar_range = sonarData['qaqc_depth_m']
        qualityLogic = sonarData['qaqc_depth_m'] >= 0
    else:
        raise ValueError(f'acceptable sonar methods include {sonar_methods}')
    # use the above to adjust whether you want smoothed/filtered data or raw ping depth values

    ofname = os.path.join(plotDir, 'SonarBackScatter.png')
    # 6.5 now plot sonar with time
    yellowfinLib.qaqc_sonar_profiles(ofname, sonarData)
    ofname = os.path.join(plotDir, 'AllData.png')

    yellowfinLib.qaqc_plot_all_data_in_time(ofname, sonarData, sonar_range, payloadGpsData, T_ppk)


    ofname = os.path.join(plotDir, 'subsetForCrossCorrelation.png')

    # 6.7 # plot sonar, select indices of interest, and then second subplot is time of interest
    sonarIndicies = yellowfinLib.sonar_pick_cross_correlation_time(ofname, sonar_range)
    # now identify corresponding times from ppk GPS to those times of sonar that we're interested in
    indsPPK = np.where((T_ppk['epochTime'] >= sonarData['time'][sonarIndicies[0]]) & (
            T_ppk['epochTime'] <= sonarData['time'][sonarIndicies[-1]]))[0]

    # 6.7 interpolate and calculate the phase offset between the signals

    ## now interpolate the lower sampled (sonar 3.33 hz) to the higher sampled data (gps 10 hz)
    # identify common timestamp to interpolate to at higher frequency
    commonTime = np.linspace(T_ppk['epochTime'][indsPPK[0]], T_ppk['epochTime'][indsPPK[-1]],
                             int((T_ppk['epochTime'][indsPPK[-1]] - T_ppk['epochTime'][indsPPK[0]]) / .1),
                             endpoint=True)

    # always use instant ping for time offset calculation
    f = interpolate.interp1d(sonarData['time'], sonarData['this_ping_depth_m'])
    sonar_range_i = f(commonTime)
    f = interpolate.interp1d(T_ppk['epochTime'], T_ppk['height'])
    ppkHeight_i = f(commonTime)
    # now i have both signals at the same time stamps
    phaseLagInSamps, phaseLaginTime = yellowfinLib.findTimeShiftCrossCorr(signal.detrend(ppkHeight_i),
                                                                          signal.detrend(sonar_range_i),
                                                                          sampleFreq=np.median(np.diff(commonTime)))

    ofname = os.path.join(plotDir, 'subsetAfterCrossCorrelation.png')
    yellowfinLib.qaqc_post_sonar_time_shift(ofname, T_ppk, indsPPK, commonTime, ppkHeight_i, sonar_range_i,
                                            phaseLaginTime, sonarData, sonarIndicies, sonar_range)

    print(f"sonar data adjusted by {phaseLaginTime:.3f} seconds")

    ## now process all data for saving to file
    sonar_time_out = sonarData['time'] + phaseLaginTime

    ## ok now put the sonar data on the GNSS timestamps which are decimal seconds.  We can do this with sonar_time_out,
    # because we just adjusted by the phase lag to make sure they are time synced.
    timeOutInterpStart = np.ceil(sonar_time_out.min() * 10) / 10  # round to nearest 0.1
    timeOutInterpEnd = np.floor(sonar_time_out.max() * 10) / 10  # round to nearest 0.1
    # create a timestamp for data to be output and in phase with that of the ppk gps data which are on the 0.1 s
    time_out = np.linspace(timeOutInterpStart, timeOutInterpEnd, int((timeOutInterpEnd - timeOutInterpStart) / 0.1),
                           endpoint=True)

    print("TODO: here's where some better filtering could be done, probably worth saving an intermediate product here "
          "for future revisit")

    logging.info(f"saving/logging values that have a GNSS fix quality of {ppk_quality_threshold} and a "
                 f"sonar confidence > {smoothed_sonar_confidence}")

    # now put relevant GNSS and sonar on output timestamps
    # initalize out variables
    sonar_smooth_depth_out, sonar_smooth_confidence_out = np.zeros_like(time_out) * np.nan, np.zeros_like(
        time_out) * np.nan
    sonar_instant_depth_out, sonar_instant_confidence_out = np.zeros_like(time_out) * np.nan, np.zeros_like(
        time_out) * np.nan
    sonar_backscatter_out = np.zeros((time_out.shape[0], sonarData['range_m'].shape[0])) * np.nan
    bad_lat_out, bad_lon_out, lat_out, lon_out = np.zeros_like(time_out) * np.nan, np.zeros_like(
        time_out) * np.nan, np.zeros_like(time_out) * np.nan, np.zeros_like(time_out) * np.nan
    elevation_out, fix_quality = np.zeros_like(time_out) * np.nan, np.zeros_like(time_out) * np.nan
    gnss_out, sonar_out = np.zeros_like(bad_lat_out) * np.nan, np.zeros_like(time_out) * np.nan
    # loop through my common time (.1 s increment) and find associated sonar and gnss values; this might be slow
    for tidx, tt in tqdm.tqdm(enumerate(time_out), desc="Filter & Time Match"):
        idxTimeMatchGNSS, idxTimeMatchGNSS = None, None

        # first find if there is a time match for sonar
        sonarlogic = np.abs(np.ceil(tt * 10) / 10 - np.ceil(sonar_time_out * 10) / 10)
        if sonarlogic.min() <= 0.2:  # 0.2  with a sampling of <0-2, it should identify the nearest sample (at 0.3s interval)
            idxTimeMatchSonar = np.argmin(sonarlogic)
        # then find comparable time match for ppk
        ppklogic = np.abs(np.ceil(tt * 10) / 10 - np.ceil(T_ppk['epochTime'].array * 10) / 10)
        if ppklogic.min() <= 0.101:  # .101 handles numerics
            idxTimeMatchGNSS = np.argmin(ppklogic)

        # if we have both sonar and GNSS for this time step, then we log the data
        if idxTimeMatchGNSS is not None and idxTimeMatchSonar is not None:  # we have matching data
            # if it passes quality thresholds
            if T_ppk['Q'][idxTimeMatchGNSS] <= ppk_quality_threshold and qualityLogic[idxTimeMatchSonar]:
                # log matching data that meets quality metrics
                sonar_smooth_depth_out[tidx] = sonarData['smooth_depth_m'][idxTimeMatchSonar]
                sonar_instant_depth_out[tidx] = sonarData['this_ping_depth_m'][idxTimeMatchSonar]
                sonar_smooth_confidence_out[tidx] = sonarData['smoothed_depth_measurement_confidence'][
                    idxTimeMatchSonar]
                sonar_instant_confidence_out[tidx] = sonarData['this_ping_depth_measurement_confidence'][
                    idxTimeMatchSonar]
                sonar_backscatter_out[tidx] = sonarData['profile_data'][:, idxTimeMatchSonar]
                lat_out[tidx] = T_ppk['lat'][idxTimeMatchGNSS]
                lon_out[tidx] = T_ppk['lon'][idxTimeMatchGNSS]
                gnss_out[tidx] = T_ppk['GNSS_elevation_NAVD88'][idxTimeMatchGNSS]
                fix_quality[tidx] = T_ppk['Q'][idxTimeMatchGNSS]
                # now log elevation outs depending on which sonar i want to log
                if sonar_method == 'default':
                    elevation_out[tidx] = T_ppk['GNSS_elevation_NAVD88'][idxTimeMatchGNSS] - antenna_offset - \
                                          sonarData['smooth_depth_m'][idxTimeMatchSonar]
                    sonar_out[tidx] = sonarData['smooth_depth_m'][idxTimeMatchSonar]

                elif sonar_method == 'smooth':
                    elevation_out[tidx] = T_ppk['GNSS_elevation_NAVD88'][idxTimeMatchGNSS] - antenna_offset - \
                                          sonarData['smooth_depth_m'][idxTimeMatchSonar]
                    sonar_out[tidx] = sonarData['smooth_depth_m'][idxTimeMatchSonar]

                elif sonar_method == 'instant':
                    elevation_out[tidx] = T_ppk['GNSS_elevation_NAVD88'][idxTimeMatchGNSS] - antenna_offset - \
                                          sonarData['this_ping_depth_m'][idxTimeMatchSonar]
                    sonar_out[tidx] = sonarData['this_ping_depth_m'][idxTimeMatchSonar]
                elif sonar_method == 'qaqc':
                    elevation_out[tidx] = T_ppk['GNSS_elevation_NAVD88'][idxTimeMatchGNSS] - antenna_offset - \
                                          sonarData['qaqc_depth_m'][idxTimeMatchSonar]
                    sonar_out[tidx] = sonarData['qaqc_depth_m'][idxTimeMatchSonar]
                else:
                    raise ValueError(f'acceptable sonar methods include {sonar_methods}')

            # now log bad locations for quality plotting
            if T_ppk['Q'][idxTimeMatchGNSS] <= ppk_quality_threshold and not qualityLogic[idxTimeMatchSonar]:
                bad_lat_out[tidx] = T_ppk['lat'][idxTimeMatchGNSS]
                bad_lon_out[tidx] = T_ppk['lon'][idxTimeMatchGNSS]
    # identify data that are not nan's to save
    idxDataToSave = np.argwhere(~np.isnan(sonar_smooth_depth_out)).squeeze()  # identify data that are not NaNs

    # convert the lon/lat data we care about to FRF coords
    coords = geoprocess.FRFcoord(lon_out[idxDataToSave], lat_out[idxDataToSave], coordType='LL')

    FRF = yellowfinLib.is_local_to_FRF(coords)
    if not FRF:
        logging.info("identified data as NOT Local to the FRF")
    ofname = os.path.join(plotDir, 'FinalDataProduct.png')
    yellowfinLib.plot_planview_lonlat(ofname, T_ppk, bad_lon_out, bad_lat_out, elevation_out, lat_out,
                                       lon_out, timeString, idxDataToSave, FRF)


    #now make data packat to save
    data = {'time': time_out[idxDataToSave], 'date': DT.datetime.strptime(timeString, "%Y%m%d").timestamp(),
            'Latitude': lat_out[idxDataToSave], 'Longitude': lon_out[idxDataToSave],
            'Northing': coords['StateplaneN'], 'Easting': coords['StateplaneE'],  'Elevation': elevation_out[idxDataToSave],
            'Ellipsoid': np.ones_like(elevation_out[idxDataToSave]) * -999}
    if FRF:
        data['xFRF'] = coords['xFRF']
        data['yFRF'] = coords['yFRF']
        data['Profile_number'] = np.ones_like(elevation_out[idxDataToSave]) * -999
        data['Survey_number'] =  np.ones_like(elevation_out[idxDataToSave]) * -999
        yellowfinLib.plotPlanViewOnArgus(data, argusGeotiff, ofName=os.path.join(plotDir, 'yellowfinDepthsOnArgus.png'))

        ofname = os.path.join(plotDir, 'singleProfile.png')
        yellowfinLib.plot_planview_FRF(ofname, coords, gnss_out, antenna_offset, elevation_out, sonar_instant_depth_out, sonar_smooth_depth_out, idxDataToSave)

        data['UNIX_timestamp'] = data['time']
        data = yellowfinLib.transectSelection(pd.DataFrame.from_dict(data), outputDir=plotDir) # bombs out on non-frf data
        data['Profile_number'] = data.pop('profileNumber')
        data['Profile_number'].iloc[np.argwhere(data['Profile_number'].isnull()).squeeze()] = -999

    ## now make netCDF files
    ofname = os.path.join(datadir, f'FRF_geomorphology_elevationTransects_survey_{timeString}.nc')
    py2netCDF.makenc_generic(ofname, globalYaml='yamlFile/transect_global.yml',
                             varYaml='yamlFile/transect_variables.yml', data=data)


    outputfile = os.path.join(datadir, f'{timeString}_totalCombinedRawData.h5')
    with h5py.File(outputfile, 'w') as hf:
        hf.create_dataset('time', data=time_out[idxDataToSave])
        hf.create_dataset('longitude', data=lon_out[idxDataToSave])
        hf.create_dataset('latitude', data=lat_out[idxDataToSave])
        hf.create_dataset('elevation', data=elevation_out[idxDataToSave])
        hf.create_dataset('fix_quality_GNSS', data=fix_quality[idxDataToSave])
        hf.create_dataset('gnss_elevation_navd_m', data=gnss_out)
        hf.create_dataset('sonar_smooth_depth', data=sonar_smooth_depth_out[idxDataToSave])
        hf.create_dataset('sonar_smooth_confidence', data=sonar_smooth_confidence_out[idxDataToSave])
        hf.create_dataset('sonar_instant_depth', data=sonar_instant_depth_out[idxDataToSave])
        hf.create_dataset('sonar_instant_confidence', data=sonar_instant_confidence_out[idxDataToSave])
        hf.create_dataset('sonar_backscatter_out', data=sonar_backscatter_out[idxDataToSave])
        hf.create_dataset('bad_lat', data=bad_lat_out)
        hf.create_dataset('bad_lon', data=bad_lon_out)
        hf.create_dataset('sonar_depth_bin', data=sonarData['range_m'])
        if FRF:
            hf.create_dataset('xFRF', data=coords['xFRF'])
            hf.create_dataset('yFRF', data=coords['yFRF'])
            hf.create_dataset('Profile_number', data=data['Profile_number'])


    # Make mission summary YAML based on user prompted inputs, write to datadir
    make_summary_yaml(datadir)

    # Make mission failure YAML based on user prompted inputs, write to datadir
    make_failure_yaml(datadir)


if __name__ == "__main__":
    # filepath = '/data/yellowfin/20231109'  # 327'  # 04' #623' #705'
    args = parse_args(__version__)
    assert os.path.isdir(args.data_dir), "check your input filepath, code doesn't see the folder"

    main(args.data_dir, geoid=args.geoid_file, makePos=args.make_pos, verbose=args.verbosity,
         rtklib_executable_path=args.rtklib_executable, sonar_method=args.sonar_method,
         ppk_quality_threshold = args.ppk_quality_threshold,
    smoothed_sonar_confidence = args.smoothed_sonar_confidence, instant_sonar_confidence = args.instant_sonar_confidence)
    logging.info(f"success processing {args.data_dir}")

