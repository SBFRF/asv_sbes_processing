import os
import matplotlib
from bottomTracerGUI import run_sonar_tracer_gui
# matplotlib.use("TkAgg")
from scipy import interpolate, signal
import py2netCDF
import yellowfinLib
import datetime as DT
import numpy as np
import h5py
import pandas as pd
import glob
import zipfile
import tqdm
from testbedutils import geoprocess
import argparse, logging, yaml
from mission_yaml_files import make_summary_yaml, make_failure_yaml

__version__ = 0.5


def deconflict_args(args, yaml_config):
    """
    Update a dict with values from a Namespace if they differ.

    Parameters:
        args (argparse.Namespace): source of truth for values.
        yaml_config (dict):        dict to update.

    Returns:
        dict: the updated mapping (in-place).
    """
    for key, ns_value in vars(args).items():
        # print(key, ns_value)
        # only overwrite when there's a difference (or missing key)
        if yaml_config.get(key) != ns_value and yaml_config.get(key) != None:
            print(f"updated {key} from {yaml_config[key]} to {ns_value}")
            yaml_config[key] = ns_value

    return yaml_config


def parse_args(__version__):
    parser = argparse.ArgumentParser(f"PPK processing for yellowfin (V{__version__})", add_help=True)
    # datadir, geoid, makePos = True, verbose = 1
    # Command-Line Interface: (REQUIRED) Flags
    parser.add_argument(
        "-d",
        "--data_dir",
        type=str,
        metavar=True,
        help="[REQUIRED] directory of data that are to be processed",
        required=True,
    )

    parser.add_argument(
        "--config",
        type=str,
        required=False,
        default=None,
        help="YAML config file, will overwrite all other arguments in CLI",
    )

    # Command-Line Interface: (OPTIONAL) Flags
    parser.add_argument(
        "-g",
        "--geoid_file",
        type=str,
        default="ref/g2012bu0.bin",
        metavar="",
        help="binary geoid file, required for conversion of ellipsoid height to NAVD88",
    )
    parser.add_argument(
        "-p",
        "--make_pos",
        action=argparse.BooleanOptionalAction,
        type=bool,
        default=False,
        help="make posfile (True) using RTKlib or provide one through external environment (false)",
    )
    parser.add_argument(
        "-v",
        "--verbosity",
        type=int,
        default=2,
        metavar="",
        help="sets verbosity for debug, 1=Debug (most), 2=Info (normal), 3=Warning (least)",
    )
    parser.add_argument(
        "--sonar_method",
        type=str,
        default="default",
        help="which s500 depth reading to use for time-shifting and bottom reporting, avialable "
        "are ['default', 'smooth', 'instant']. default uses instant depth for time syncing and"
        " smooth depths for final bathy out; 'smooth' uses smoothed values for both, 'instant' "
        "uses instant values for both",
    )
    parser.add_argument(
        "--rtklib_executable",
        type=str,
        default="ref/rnx2rtkp",
        help="path for RTK_lib executable (required if --make-pos flag assigned)",
    )
    parser.add_argument(
        "--ppk_quality_threshold",
        type=int,
        default=1,
        help="this is a quality threshold 1: Fixed, 2: Float, 4:DGPS, 5: single -- see appendix B for "
        "more details: https://rtkexplorer.com/pdfs/manual_demo5.pdf  ",
    )

    return parser.parse_args()


def parse_config_yaml(fname):
    """Load configuration from YAML file."""
    with open(fname, "r") as file:
        return yaml.safe_load(file)


def update_namespace_from_dict(namespace, updates):
    """
    Update an argparse.Namespace with values from a dictionary if they differ.

    Parameters:
        namespace (argparse.Namespace): The namespace to update.
        updates (dict): A dictionary of key-value pairs to update in the namespace.

    Returns:
        argparse.Namespace: The updated namespace.
    """
    for key, new_value in updates.items():
        # If the key exists, compare values; if not, add it.
        if hasattr(namespace, key):
            current_value = getattr(namespace, key)
            if current_value != new_value:
                setattr(namespace, key, new_value)
        else:
            setattr(namespace, key, new_value)
    return namespace


def verbosity_conversion(verbose: int):
    if verbose == 1:
        logging.basicConfig(level=logging.DEBUG)
    elif verbose == 2:
        logging.basicConfig(level=logging.INFO)
    elif verbose == 3:
        logging.basicConfig(level=logging.WARN)
    else:
        raise EnvironmentError("logging verbosity is wrong!")


def main(
    datadir,
    geoid,
    makePos=True,
    rtklib_executable_path="ref/rnx2rtkp",
    instant_sonar_confidence=99,
    smoothed_sonar_confidence=60,
    yaml_config=None,
):
    """This function is the main function for processing ppk GNSS and sonar data for MURG."""
    acceptable_time_sync = ["default", "instant", "smooth", "native"]
    verbose = yaml_config['processing'].get('verbosity', 2) # overwrite hard argument with yaml
    verbosity_conversion(verbose)
    # unpack yaml configuration
    antenna_offset = yaml_config.get(
        "gnss_antenna_offset_m", 0.25
    )  # meters between the antenna phase center and sounder head - default for yellowfin
    time_sync_method = yaml_config["processing"].get("time_sync_method", "default").lower()
    sonar_model = yaml_config["sonar"].get("sonar_model", False).lower()
    ppk_quality_threshold = yaml_config["processing"].get("ppk_quality_threshold", 1)

    #  date that Pi computer was changed to UTC time (will adjust timezone manually before this date)
    yellowfin_clock_reset_date = DT.datetime(2023, 7, 10)  # do not adjust this date!

    logging.info(f"input folder: {datadir}")
    logging.info(f"ppk_quality_threshold: {ppk_quality_threshold}")
    ####################################################################################################################

    if datadir.endswith("/"):
        datadir = datadir[:-1]
    ## Define all paths for the workflow
    timeString = os.path.basename(datadir)
    plotDir = os.path.join(datadir, "figures")
    os.makedirs(plotDir, exist_ok=True)  # make folder structure if its not already made

    # sonar data
    fpathSonar = os.path.join(datadir, sonar_model)  # reads sonar from here
    saveFnameSonar = os.path.join(datadir, f"{timeString}_sonarRaw.h5")  # saves sonar file here
    traced_fname_sonar = os.path.join(datadir, f"{timeString}_sonarRaw_bottomTraced_wholeRecord.h5")
    # NMEA data from sonar, this is not Post Processed Kinematic (PPK) data.  It is used for only cursory or
    # introductory look at the data
    fpathGNSS = os.path.join(datadir, "nmeadata")  # load NMEA data from this location
    save_fname_gnss = os.path.join(datadir, f"{timeString}_gnssRaw.h5")  # save nmea data to this location

    # RINEX data
    # look for all subfolders with RINEX in the folder name inside the "datadir" emlid ppk processor
    fpath_pos_files = os.path.join(datadir, "pos_files")
    saveFnamePPK = os.path.join(datadir, f"{timeString}_ppkRaw.h5")

    logging.debug(f"saving intermediate files for sonar here: {saveFnameSonar}")
    logging.debug(f"saving intermediate files for sonar here: {saveFnamePPK}")
    logging.debug(f"saving intermediate files for GNSS here: {save_fname_gnss}")
    if sonar_model in ["s500"] and not os.path.isfile(saveFnameSonar):
        if time_sync_method == "default":
            bathy_report = "smoothed"
            sonar_confidence = smoothed_sonar_confidence
        elif time_sync_method == "instant":
            sonar_confidence = instant_sonar_confidence
            bathy_report = time_sync_method

        elif time_sync_method == "smoothed":
            sonar_confidence = smoothed_sonar_confidence
            bathy_report = time_sync_method

        logging.info(f"procesing prameters:  sonar time sync method {time_sync_method}")
        logging.info(f"procesing prameters:  bathy sonar method {bathy_report}")
        logging.info(f"sonar_confidence: {sonar_confidence} %")
        ## load files
        if not os.path.isfile(saveFnameSonar):
            yellowfinLib.loadSonar_s500_binary(fpathSonar, h5_ofname=saveFnameSonar, verbose=verbose)
        else:
            logging.info(f"Skipping {saveFnameSonar}")
    elif sonar_model.lower() in ["d032", "ect-d032"]:
        high_low = yellowfinLib.is_high_low_dual_freq(saveFnameSonar)
        timeString = timeString + "_low_"
        of_plot = os.path.join(plotDir, f"{timeString}_raw_sonar-ect-d032.png")
        saveFnameSonar = os.path.join(datadir, f"{timeString}_sonarRaw.h5")  # saves sonar file here
        yellowfinLib.loadSonar_ectd032_ascii(
            fpathSonar,
            h5_ofname=saveFnameSonar,
            verbose=verbose,
            of_plot=of_plot,
            high_low=high_low,
        )

    elif not os.path.isfile(saveFnameSonar):
        raise NotImplementedError("sonar option not implemented")

    # then load NMEA files
    if not os.path.isfile(save_fname_gnss) and time_sync_method != "native":
        yellowfinLib.load_yellowfin_NMEA_files(
            fpathGNSS,
            saveFname=save_fname_gnss,
            #plotfname=os.path.join(plotDir, "GPSpath_fromNMEAfiles.png"), # confusing plot
            verbose=verbose,
        )
    else:  # we've already generated this fpathGNSS file
        logging.info(f"Skipping {save_fname_gnss}")

    if not os.path.isfile(saveFnamePPK):
        if makePos == True:
            # find folders with raw rinex
            rover_rinex_zip_files = glob.glob(os.path.join(fpath_pos_files, "*RINEX*.zip"))
            # identify the nav/obs file
            base_zip_files = glob.glob(os.path.join(datadir, "CORS", "*.zip"))

            if np.size(base_zip_files) == 1:  # if there's zip file it's from CORS
                base_zip_files = base_zip_files[0]
                with zipfile.ZipFile(base_zip_files, "r") as zip_ref:
                    zip_ref.extractall(path=base_zip_files[:-4])
                cors_search_path_obs = os.path.join(os.path.splitext(base_zip_files)[0], "*o")
                cors_search_path_nav = os.path.join(os.path.splitext(base_zip_files)[0], "*n")
                cors_search_path_sp3 = os.path.join(os.path.splitext(base_zip_files)[0], "*sp3")
            elif np.size(base_zip_files) > 1:  # if there's more than one zip file
                raise EnvironmentError("There are too many zip files in the CORS folder to extract")
            else:
                cors_search_path_obs = os.path.join(datadir, "CORS", "*o")
                cors_search_path_nav = os.path.join(datadir, "CORS", "*n")
                cors_search_path_sp3 = os.path.join(datadir, "CORS", "*sp3")

            base_obs_fname = glob.glob(cors_search_path_obs)[0]
            base_nav_file = glob.glob(cors_search_path_nav)[0]
            base_sp3_list = glob.glob(cors_search_path_sp3)
            if np.size(base_sp3_list) == 1:
                sp3_fname = base_sp3_list[0]
            else:
                sp3_fname = ""

            # unzip all the rinex Files
            for ff in rover_rinex_zip_files:
                with zipfile.ZipFile(ff, "r") as zip_ref:
                    zip_ref.extractall(path=ff[:-4])
                # identify and process rinex to Pos files
                flist_rinex = glob.glob(ff[:-4] + "/*")
                rover_obs_fname = flist_rinex[np.argwhere([i.endswith("O") for i in flist_rinex]).squeeze()]
                outfname = os.path.join(
                    os.path.dirname(rover_obs_fname),
                    os.path.basename(flist_rinex[0])[:-3] + "pos",
                )
                # use below if the rover nav file is the right call
                yellowfinLib.makePOSfileFromRINEX(
                    roverObservables=rover_obs_fname,
                    baseObservables=base_obs_fname,
                    navFile=base_nav_file,
                    outfname=outfname,
                    executablePath=rtklib_executable_path,
                    sp3=sp3_fname,
                )

        # Now find all the folders that have ppk data in them (*.pos files in folders that have "raw" in them)
        # now identify the folders that have rinex in them
        flist_pos = sorted(glob.glob(os.path.join(fpath_pos_files, "*.pos")))
        if len(flist_pos) < 1:
            raise NotImplementedError("need to put pos files in pos folder")
            fldrlistPPK = []  # initalize list for appending RINEX folder in
            [
                fldrlistPPK.append(os.path.join(fpath_pos_files, fname))
                for fname in os.listdir(fpath_pos_files)
                if "raw" in fname and ".zip" not in fname
            ]

        logging.warning("load PPK pos files ---- THESE ARE WGS84 (EPSG:4326) !!!!!!!!!!!!!!")
        try:
            T_ppk = yellowfinLib.load_ppk_fils_list(flist_ppk=flist_pos)
            T_ppk.to_hdf(path_or_buf=saveFnamePPK, key="ppk")  # now save the h5 intermediate file
        except KeyError:
            raise FileExistsError("the pos file hasn't been loaded, manually produce or turn on RTKlib processing")
    else:
        logging.info(f"Skipping {saveFnamePPK}")
        T_ppk = pd.read_hdf(saveFnamePPK)

    # 1. time in seconds to adjust to UTC from ET (varies depending on time of year!!!)
    if (
        (T_ppk["datetime"].iloc[0].replace(tzinfo=None) < yellowfin_clock_reset_date)
        & (int(T_ppk["datetime"].iloc[0].day_of_year) > 71)
        & (int(T_ppk["datetime"].iloc[0].day_of_year) < 309)
    ):
        ET2UTC = 5 * 60 * 60
        logging.warning(" I'm using a 'dumb' conversion from ET to UTC")
    elif (
        (T_ppk["datetime"].iloc[0].replace(tzinfo=None) < yellowfin_clock_reset_date)
        & (int(T_ppk["datetime"].iloc[0].day_of_year) < 71)
        & (int(T_ppk["datetime"].iloc[0].day_of_year) > 309)
    ):
        ET2UTC = 4 * 60 * 60
        logging.warning(" I'm using a 'dumb' conversion from ET to UTC")
    else:
        ET2UTC = 0  # time's already in UTC

    ##################################### above is loading/making intermediate files ################################3
    # 6.2: load all files we created in previous steps
    sonarData = yellowfinLib.load_h5_to_dictionary(saveFnameSonar)
    trace_bottom_chunk = yaml_config['processing'].get('trace_bottom_chunk_size', 250)
    if not os.path.exists(traced_fname_sonar):  # if the traced bottom doesn't exist, go into the gui
        traced_bottom = run_sonar_tracer_gui(saveFnameSonar, trace_bottom_chunk)
    else:
        traced_bottom = yellowfinLib.load_h5_to_dictionary(traced_fname_sonar)
    # now fuse bottom traced to the correct bathy location
    sonarData = yellowfinLib.swap_human_traced_line(sonarData, traced_bottom)
    T_ppk = pd.read_hdf(saveFnamePPK)
    if time_sync_method != "native":
        payload_gps_data = yellowfinLib.load_h5_to_dictionary(
            save_fname_gnss
        )  # this is used for the pc time adjustment
    else:
        payload_gps_data = None
    # Adjust GNSS time by the Leap Seconds https://www.cnmoc.usff.navy.mil/Our-Commands/United-States-Naval-Observatory/Precise-Time-Department/Global-Positioning-System/USNO-GPS-Time-Transfer/Leap-Seconds/
    # T_ppk['epochTime'] = T_ppk['epochTime'] - 18  # 18 is leap second adjustment
    # T_ppk['datetime'] = T_ppk['datetime'] - DT.timedelta(seconds=18)  # making sure both are equal
    # commented because the cross-correlation should account for this anyway (?)

    # convert raw ellipsoid values from satellite measurement to that of a vertical datum.  This uses NAVD88 [m] NAD83
    T_ppk["GNSS_elevation_NAVD88"] = yellowfinLib.convertEllipsoid2NAVD88(
        T_ppk["lat"], T_ppk["lon"], T_ppk["height"], geoidFile=geoid
    )
    # 6.3: now plot my time offset between GPS and sonar
    if time_sync_method != "native":
        pc_time_off = payload_gps_data["pc_time_gga"] + ET2UTC - payload_gps_data["gps_time"]
        ofname = os.path.join(plotDir, "clock_offset.png")
        yellowfinLib.plot_qaqc_time_offset_determination(ofname, pc_time_off)
    else:
        pc_time_off = np.array(0)  # in this case GNSS time is native time
    # 6.4 Use the cerulean instantaneous bed detection since not sure about delay with smoothed
    # adjust time of the sonar time stamp with timezone shift (ET -> UTC) and the timeshift between the computer and GPS
    sonarData["time"] = sonarData["time"] + ET2UTC - np.median(pc_time_off)  # convert to UTC
    if time_sync_method in acceptable_time_sync and time_sync_method == "default":
        sonar_bottom_algorithm_m = sonarData["this_ping_depth_m"]
        qualityLogic = sonarData["this_ping_depth_measurement_confidence"] > instant_sonar_confidence
    elif time_sync_method in acceptable_time_sync and time_sync_method == "smooth":
        sonar_bottom_algorithm_m = sonarData["smooth_depth_m"]
        qualityLogic = sonarData["smoothed_depth_measurement_confidence"] > smoothed_sonar_confidence
    elif time_sync_method in acceptable_time_sync and time_sync_method == "instant":
        sonar_bottom_algorithm_m = sonarData["this_ping_depth_m"]
        qualityLogic = sonarData["this_ping_depth_measurement_confidence"] > instant_sonar_confidence
    elif time_sync_method in acceptable_time_sync and time_sync_method == "native":
        sonar_bottom_algorithm_m = sonarData["this_ping_depth_m"]
        qualityLogic = np.ones_like(sonar_bottom_algorithm_m, dtype=bool)
    elif time_sync_method not in acceptable_time_sync:
        raise ValueError(f"acceptable sonar methods include {acceptable_time_sync}")

    # 6.5 now plot sonar with time
    ofname = os.path.join(plotDir, f"{timeString}_SonarBackScatter.png")
    yellowfinLib.plot_qaqc_sonar_profiles(ofname, sonarData)

    ofname = os.path.join(plotDir, f"{timeString}_AllData.png")
    yellowfinLib.plot_qaqc_all_data_in_time(ofname, sonarData, sonar_bottom_algorithm_m, payload_gps_data, T_ppk)

    if time_sync_method == "native":
        sonar_time_out = sonarData["time"]
    else:
        # 6.7 # plot sonar, select indices of interest, and then second subplot is time of interest
        ofname = os.path.join(plotDir, f"{timeString}_subsetForCrossCorrelation.png")
        sonarIndicies = yellowfinLib.plot_sonar_pick_cross_correlation_time(ofname, sonar_bottom_algorithm_m)
        # now identify corresponding times from ppk GPS to those times of sonar that we're interested in
        indsPPK = np.where(
            (T_ppk["epochTime"] >= sonarData["time"][sonarIndicies[0]])
            & (T_ppk["epochTime"] <= sonarData["time"][sonarIndicies[-1]])
        )[0]

        # 6.7 interpolate and calculate the phase offset between the signals

        ## now interpolate the lower sampled (sonar 3.33 hz) to the higher sampled data (gps 10 hz)
        # identify common timestamp to interpolate to at higher frequency
        commonTime = np.linspace(
            T_ppk["epochTime"][indsPPK[0]],
            T_ppk["epochTime"][indsPPK[-1]],
            int((T_ppk["epochTime"][indsPPK[-1]] - T_ppk["epochTime"][indsPPK[0]]) / 0.1),
            endpoint=True,
        )

        # always use instant ping for time offset calculation
        f = interpolate.interp1d(sonarData["time"], sonarData["this_ping_depth_m"])
        sonar_range_i = f(commonTime)
        f = interpolate.interp1d(T_ppk["epochTime"], T_ppk["height"])
        ppkHeight_i = f(commonTime)
        # now i have both signals at the same time stamps
        phaseLagInSamps, phaseLaginTime = yellowfinLib.findTimeShiftCrossCorr(
            signal.detrend(ppkHeight_i),
            signal.detrend(sonar_range_i),
            sampleFreq=np.median(np.diff(commonTime)),
        )

        ofname = os.path.join(plotDir, f"{timeString}_subsetAfterCrossCorrelation.png")
        yellowfinLib.plot_qaqc_post_sonar_time_shift(
            ofname,
            T_ppk,
            indsPPK,
            commonTime,
            ppkHeight_i,
            sonar_range_i,
            phaseLaginTime,
            sonarData,
            sonarIndicies,
            sonar_bottom_algorithm_m,
        )

        print(f"sonar data adjusted by {phaseLaginTime:.3f} seconds")

        ## now process all data for saving to file
        sonar_time_out = sonarData["time"] + phaseLaginTime

    ## ok now put the sonar data on the GNSS timestamps which are decimal seconds.  We can do this with sonar_time_out,
    # because we just adjusted by the phase lag to make sure they are time synced.
    timeOutInterpStart = np.ceil(sonar_time_out.min() * 10) / 10  # round to nearest 0.1
    timeOutInterpEnd = np.floor(sonar_time_out.max() * 10) / 10  # round to nearest 0.1
    # create a timestamp for data to be output and in phase with that of the ppk gps data which are on the 0.1 s
    dt = (
        np.round(
            min(
                np.median(np.diff(T_ppk["epochTime"])),
                np.median(np.diff(sonarData["time"])),
            )
            * 10
        )
        / 10
    )
    time_out = np.linspace(
        timeOutInterpStart,
        timeOutInterpEnd,
        int((timeOutInterpEnd - timeOutInterpStart) / dt),
        endpoint=True,
    )

    logging.info(
        "TODO: here's where some better filtering could be done, probably worth saving an intermediate product here "
        "for future revisit"
    )

    logging.info(
        f"saving/logging values that have a GNSS fix quality of {ppk_quality_threshold} and a "
        f"sonar confidence > {smoothed_sonar_confidence}"
    )

    # now pair relevant GNSS and sonar on output newly generated time_out
    # initalize out variables
    sonar_smooth_depth_out, sonar_smooth_confidence_out = (
        np.zeros_like(time_out) * np.nan,
        np.zeros_like(time_out) * np.nan,
    )
    sonar_instant_depth_out, sonar_instant_confidence_out = (
        np.zeros_like(time_out) * np.nan,
        np.zeros_like(time_out) * np.nan,
    )
    sonar_backscatter_out = np.zeros((time_out.shape[0], sonarData["range_m"].shape[0])) * np.nan
    bad_lat_out, bad_lon_out, lat_out, lon_out = (
        np.zeros_like(time_out) * np.nan,
        np.zeros_like(time_out) * np.nan,
        np.zeros_like(time_out) * np.nan,
        np.zeros_like(time_out) * np.nan,
    )
    elevation_out, fix_quality = (
        np.zeros_like(time_out) * np.nan,
        np.zeros_like(time_out) * np.nan,
    )
    gnss_out, sonar_out = (
        np.zeros_like(bad_lat_out) * np.nan,
        np.zeros_like(time_out) * np.nan,
    )
    # loop through my common time (.1 s increment) and find associated sonar and gnss values; this might be slow
    for tidx, tt in tqdm.tqdm(enumerate(time_out), desc="Filter & Time Match"):
        idxTimeMatchGNSS, idxTimeMatchGNSS = None, None

        # first find if there is a time match for sonar
        sonarlogic = np.abs(np.ceil(tt * 10) / 10 - np.ceil(sonar_time_out * 10) / 10)
        if (
            sonarlogic.min() <= 0.2
        ):  # 0.2  with a sampling of <0-2, it should identify the nearest sample (at 0.3s interval)
            idxTimeMatchSonar = np.argmin(sonarlogic)
        # then find comparable time match for ppk
        ppklogic = np.abs(np.ceil(tt * 10) / 10 - np.ceil(T_ppk["epochTime"].array * 10) / 10)
        if ppklogic.min() <= 0.101:  # .101 handles numerics
            idxTimeMatchGNSS = np.argmin(ppklogic)

        # if we have both sonar and GNSS for this time step, then we log the data as matched
        if idxTimeMatchGNSS is not None and idxTimeMatchSonar is not None:  # we have matching data
            # if it passes quality thresholds
            if T_ppk["Q"][idxTimeMatchGNSS] <= ppk_quality_threshold and qualityLogic[idxTimeMatchSonar]:
                # log matching data that meets quality metrics
                sonar_smooth_depth_out[tidx] = sonarData["smooth_depth_m"][idxTimeMatchSonar]
                sonar_instant_depth_out[tidx] = sonarData["this_ping_depth_m"][idxTimeMatchSonar]
                sonar_smooth_confidence_out[tidx] = sonarData["smoothed_depth_measurement_confidence"][
                    idxTimeMatchSonar
                ]
                sonar_instant_confidence_out[tidx] = sonarData["this_ping_depth_measurement_confidence"][
                    idxTimeMatchSonar
                ]
                sonar_backscatter_out[tidx] = sonarData["profile_data"][:, idxTimeMatchSonar]
                lat_out[tidx] = T_ppk["lat"][idxTimeMatchGNSS]
                lon_out[tidx] = T_ppk["lon"][idxTimeMatchGNSS]
                gnss_out[tidx] = T_ppk["GNSS_elevation_NAVD88"][idxTimeMatchGNSS]
                fix_quality[tidx] = T_ppk["Q"][idxTimeMatchGNSS]
                # now log elevation outs depending on which sonar i want to log
                if time_sync_method == "default":
                    elevation_out[tidx] = (
                        T_ppk["GNSS_elevation_NAVD88"][idxTimeMatchGNSS]
                        - antenna_offset
                        - sonarData["smooth_depth_m"][idxTimeMatchSonar]
                    )
                    sonar_out[tidx] = sonarData["smooth_depth_m"][idxTimeMatchSonar]

                elif time_sync_method == "smooth":
                    elevation_out[tidx] = (
                        T_ppk["GNSS_elevation_NAVD88"][idxTimeMatchGNSS]
                        - antenna_offset
                        - sonarData["smooth_depth_m"][idxTimeMatchSonar]
                    )
                    sonar_out[tidx] = sonarData["smooth_depth_m"][idxTimeMatchSonar]

                elif time_sync_method == "instant":
                    elevation_out[tidx] = (
                        T_ppk["GNSS_elevation_NAVD88"][idxTimeMatchGNSS]
                        - antenna_offset
                        - sonarData["this_ping_depth_m"][idxTimeMatchSonar]
                    )
                    sonar_out[tidx] = sonarData["this_ping_depth_m"][idxTimeMatchSonar]
                elif time_sync_method == "native":
                    elevation_out[tidx] = (
                        T_ppk["GNSS_elevation_NAVD88"][idxTimeMatchGNSS]
                        - antenna_offset
                        - sonarData["this_ping_depth_m"][idxTimeMatchSonar]
                    )
                    sonar_out[tidx] = sonarData["this_ping_depth_m"][idxTimeMatchSonar]
                else:
                    raise ValueError('acceptable sonar methods include ["default", "instant", "smooth"]')

            # now log bad locations for quality plotting
            if T_ppk["Q"][idxTimeMatchGNSS] <= ppk_quality_threshold and not qualityLogic[idxTimeMatchSonar]:
                bad_lat_out[tidx] = T_ppk["lat"][idxTimeMatchGNSS]
                bad_lon_out[tidx] = T_ppk["lon"][idxTimeMatchGNSS]
    # identify data that are not nan's to save
    idxDataToSave = np.argwhere(~np.isnan(sonar_smooth_depth_out)).squeeze()  # identify data that are not NaNs

    # convert the lon/lat data we care about to FRF coords
    coords = geoprocess.FRFcoord(lon_out[idxDataToSave], lat_out[idxDataToSave], coordType="LL")

    # identify if data are local to the FRF, will be used later to process FRF specific data quantities
    is_local_FRF = yellowfinLib.is_local_to_FRF(coords)

    if not is_local_FRF:
        logging.info("identified data as NOT Local to the FRF")
    else:  # start argus download as soon as we know its FRF data
        glob_argus_result = glob.glob(os.path.join(plotDir, "*rgus*.tif"))
        if len(glob_argus_result) == 1:
            argusGeotiff = glob_argus_result[0]
        else:
            argusGeotiff = None
            # below was throwing errors
            # argusGeotiff = yellowfinLib.threadGetArgusImagery(DT.datetime.strptime(timeString, '%Y%m%d') +
            #                                                   DT.timedelta(hours=14),
            #                                                   ofName=os.path.join(plotDir, f'Argus_{timeString}.tif'),
            #                                                   imageType='timex')

    ofname = os.path.join(plotDir, f"{timeString}_FinalDataProduct.png")
    yellowfinLib.plot_planview_lonlat(
        ofname=ofname,
        T_ppk=T_ppk,
        bad_lon_out=bad_lon_out,
        bad_lat_out=bad_lat_out,
        elevation_out=elevation_out,
        lat_out=lat_out,
        lon_out=lon_out,
        timeString=timeString,
        idxDataToSave=idxDataToSave,
        FRF=is_local_FRF,
    )

    # now make a data packet to save
    data_product = {
        "time": time_out[idxDataToSave],
        "date": np.ones_like(time_out[idxDataToSave]) * DT.datetime.strptime(timeString[:8], "%Y%m%d").timestamp(),
        "Latitude": lat_out[idxDataToSave],
        "Longitude": lon_out[idxDataToSave],
        "Northing": coords["StateplaneN"],
        "Easting": coords["StateplaneE"],
        "Elevation": elevation_out[idxDataToSave],
        "Ellipsoid": np.ones_like(elevation_out[idxDataToSave]) * -999,
    }

    if is_local_FRF == True:
        data_product["xFRF"] = coords["xFRF"]
        data_product["yFRF"] = coords["yFRF"]
        data_product["Profile_number"] = np.ones_like(elevation_out[idxDataToSave]) * -999
        data_product["Survey_number"] = np.ones_like(elevation_out[idxDataToSave]) * -999
        yellowfinLib.plot_plan_view_on_argus(
            data_product,
            argusGeotiff,
            ofName=os.path.join(plotDir, f"{timeString}_yellowfinDepthsOnArgus.png"),
        )

        ofname = os.path.join(plotDir, f"{timeString}_singleProfile.png")
        yellowfinLib.plot_planview_FRF_with_profile(
            ofname,
            coords,
            instant_depths=gnss_out[idxDataToSave] - antenna_offset - sonar_instant_depth_out[idxDataToSave],
            smoothed_depths=gnss_out[idxDataToSave] - antenna_offset - sonar_smooth_depth_out[idxDataToSave],
            processed_depths=elevation_out[idxDataToSave],
        )

        data_product["UNIX_timestamp"] = data_product["time"]
        # if np.size(data_product['date']) == 1:
        #     data_product['date'] = np.ones_like(data_product['time']) * data_product['date']
        # save = data_product.pop("date")
        data_product = yellowfinLib.transect_selection_tool(pd.DataFrame.from_dict(data_product), outputDir=plotDir)
        data_product["Profile_number"] = data_product["profileNumber"]
        mask = data_product["Profile_number"].isnull()
        data_product.loc[mask, "Profile_number"] = -999  # assign -999's instead of NaN's before write

        ## now make netCDF files
        ofname = os.path.join(datadir, f"FRF_geomorphology_elevationTransects_survey_{timeString}.nc")
    else:
        # below bombs out on non-FRF data
        # data = yellowfinLib.transect_selection_tool(pd.DataFrame.from_dict(data), outputDir=plotDir)
        ofname = os.path.join(datadir, f"{'output_data'}_geomorphology_elevationTransects_survey_{timeString}.nc")
    py2netCDF.makenc_generic(
        ofname,
        globalYaml="yamlFile/transect_global.yml",
        varYaml="yamlFile/transect_variables.yml",
        data=data_product,
    )

    outputfile = os.path.join(datadir, f"{timeString}_totalCombinedRawData.h5")
    with h5py.File(outputfile, "w") as hf:
        hf.create_dataset("time", data=time_out[idxDataToSave])
        hf.create_dataset("longitude", data=lon_out[idxDataToSave])
        hf.create_dataset("latitude", data=lat_out[idxDataToSave])
        hf.create_dataset("elevation", data=elevation_out[idxDataToSave])
        hf.create_dataset("fix_quality_GNSS", data=fix_quality[idxDataToSave])
        hf.create_dataset("gnss_elevation_navd_m", data=gnss_out)
        hf.create_dataset("sonar_smooth_depth", data=sonar_smooth_depth_out[idxDataToSave])
        hf.create_dataset("sonar_smooth_confidence", data=sonar_smooth_confidence_out[idxDataToSave])
        hf.create_dataset("sonar_instant_depth", data=sonar_instant_depth_out[idxDataToSave])
        hf.create_dataset("sonar_instant_confidence", data=sonar_instant_confidence_out[idxDataToSave])
        hf.create_dataset("sonar_backscatter_out", data=sonar_backscatter_out[idxDataToSave])
        hf.create_dataset("bad_lat", data=bad_lat_out)
        hf.create_dataset("bad_lon", data=bad_lon_out)
        hf.create_dataset("sonar_depth_bin", data=sonarData["range_m"])
        if is_local_FRF is True:
            hf.create_dataset("xFRF", data=coords["xFRF"])
            hf.create_dataset("yFRF", data=coords["yFRF"])
            hf.create_dataset("Profile_number", data=data_product["Profile_number"])

    # Make mission summary YAML based on user prompted inputs, write to datadir
    make_summary_yaml(datadir)

    # Make mission failure YAML based on user prompted inputs, write to datadir
    make_failure_yaml(datadir)


if __name__ == "__main__":
    # filepath = '/data/yellowfin/20231109'  # 327'  # 04' #623' #705'
    args = parse_args(__version__)
    assert os.path.isdir(args.data_dir), "check your input filepath, code doesn't see the folder"
    extra_args = None
    if args.config is not None and args.config.endswith(".yaml"):
        yaml_config = parse_config_yaml(args.config)
        # args = update_namespace_from_dict(args, yaml_config)
        yaml_config = deconflict_args(args, yaml_config)
    elif args.config is not None:
        raise AttributeError("config file must end with .yaml")

    logging.warning("The arg parsing here is kinda stupid, please fix before merge")
    main(
        args.data_dir,
        geoid=args.geoid_file,
        makePos=args.make_pos,
        rtklib_executable_path=args.rtklib_executable,
        yaml_config=yaml_config,
    )
    logging.info(f"success processing {args.data_dir}")
