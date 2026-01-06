import datetime as DT
import glob
import logging
import os
import shutil
import struct
import threading
import time
import logging

import h5py
import netCDF4 as nc
import numpy as np
import pandas as pd
import rasterio
import tqdm
import wget
from matplotlib import pyplot as plt
from matplotlib.pyplot import ticklabel_format
from rasterio import plot as rplt
from testbedutils import geoprocess
from scipy import signal

def read_emlid_pos(fldrlistPPK, plot=False, saveFname=None):
    """read and parse multiple pos files in multiple folders provided

    :param fldrlistPPK: list of folders to provide
    :param plot: if a path name will save a QA/QC plots (default=False)
    :param saveFname: will save file as h5
    :return: dataframe with loaded data from pos file
    """
    T_ppk = pd.DataFrame()
    for fldr in sorted(fldrlistPPK):
        # this is before ppk processing so should agree with nmea strings
        fn = glob.glob(os.path.join(fldr, "*.pos"))[0]
        try:
            colNames = [
                "datetime",
                "lat",
                "lon",
                "height",
                "Q",
                "ns",
                "sdn(m)",
                "sde(m)",
                "sdu(m)",
                "sdne(m)",
                "sdeu(m)",
                "sdun(m)",
                "age(s)",
                "ratio",
            ]
            Tpos = pd.read_csv(fn, delimiter=r"\s+ ", header=10, names=colNames, engine="python")
            print(f"loaded {fn}")
            if all(Tpos.iloc[-1]):  # if theres nan's in the last row
                Tpos = Tpos.iloc[:-1]  # remove last row
            T_ppk = pd.concat([T_ppk, Tpos])  # merge multiple files to single dataframe

        except:
            continue
    T_ppk["datetime"] = pd.to_datetime(T_ppk["datetime"], format="%Y/%m/%d %H:%M:%S.%f", utc=True)

    # now make plot of both files
    # first llh file
    # plt.plot(T_LLH['lon'], T_LLH['lat'], '.-m', label = 'LLH file')
    # plt.xlabel('longitude')
    # plt.ylabel('latitude')
    if plot is not False:
        plt.plot(T_ppk["lon"], T_ppk["lat"], ".-g", label="PPK file")
        plt.xlabel("longitude")
        plt.ylabel("latitude")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot + "Lat_Lon")
        plt.close()

        fig = plt.figure()
        plt.plot(T_ppk["datetime"], T_ppk["height"], label="elevation")
        plt.plot(T_ppk["datetime"], 10 * T_ppk["Q"], ".", label="quality factor")
        plt.plot(
            T_ppk["datetime"],
            10000 * (T_ppk["lat"] - T_ppk["lat"].iloc[0]),
            label="lat from original lat",
        )
        plt.xlabel("time")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot + "elev_Q")
        plt.close()

    return T_ppk


def loadSonar_s500_binary(dataPath, outfname=None, verbose=False):
    """Loads and concatenates all of the binary files (*.dat) located in the dataPath location

    :param dataPath: search path for sonar data files
    :param outfname: string to save h5 file. If None it will skip this process. (Default =None)
    :param verbose: turn on print statement for file names as loading (1 is little print, 2 is detailed print)
    :return: pandas data frame of sonar data
    """
    # find dat files for sonar
    dd = sorted(glob.glob(os.path.join(dataPath, "*.dat")))
    if len(dd) == 0:  # if i didn't find it first, maybe i need to unpack
        # try to move dat files out of a folder with the same name as the base folder  in the s500 folder
        try:
            fldInterest = [i for i in os.listdir(dataPath) if ".dat" not in i][0]
            parts_interst = fldInterest.split("-")
            flist = glob.glob(
                os.path.join(
                    dataPath,
                    fldInterest,
                    f"{parts_interst[-1] + parts_interst[0] + parts_interst[1]}*.dat",
                )
            )
            if len(flist) < 1:  # this is the new file name (file name updated late August 2024)
                flist = glob.glob(os.path.join(dataPath, fldInterest, ''.join(parts_interst)+'*.dat'))
            toDir = "/" + os.path.join(*flist[0].split(os.sep)[:-2])
            [shutil.move(l, toDir) for l in flist]
            # os.rmdir(os.path.join(dataPath, fldInterest)) # remove folder data came from
            dd = sorted(glob.glob(os.path.join(dataPath, "*.dat")))

        except:
            raise EnvironmentError(
                "The sounder date doesn't match folder date, or there is no data in that folder"
            )

    # https://docs.ceruleansonar.com/c/v/s-500-sounder/appendix-f-programming-api
    ij, i3 = 0, 0
    allocateSize = 45000  # some ridiculously large number that memory can still hold.
    # initialize variables for loop
    distance, confidence, transmit_duration = (
        np.zeros(allocateSize),
        np.zeros(allocateSize),
        np.zeros(allocateSize),
    )  # [],
    ping_number, scan_start, scan_length = (
        np.zeros(allocateSize),
        np.zeros(allocateSize),
        np.zeros(allocateSize),
    )
    end_ping_hz, adc_sample_hz, timestamp_msec, spare2 = (
        np.zeros(allocateSize),
        np.zeros(allocateSize),
        np.zeros(allocateSize),
        np.zeros(allocateSize),
    )
    start_mm, length_mm, start_ping_hz = (
        np.zeros(allocateSize),
        np.zeros(allocateSize),
        np.zeros(allocateSize),
    )
    (
        ping_duration_sec,
        analog_gain,
        profile_data_length,
    ) = (
        np.zeros(allocateSize),
        np.zeros(allocateSize),
        np.zeros(allocateSize),
    )

    min_pwr, step_db, smooth_depth_m, fspare2 = (
        np.zeros(allocateSize),
        np.zeros(allocateSize),
        np.zeros(allocateSize),
        np.zeros(allocateSize),
    )
    is_db, gain_index, power_results = (
        np.zeros(allocateSize),
        np.zeros(allocateSize),
        np.zeros(allocateSize),
    )
    max_pwr, num_results = np.zeros(allocateSize), np.zeros(allocateSize, dtype=int)
    gain_setting, decimation, reserved = (
        np.zeros(allocateSize),
        np.zeros(allocateSize),
        np.zeros(allocateSize),
    )
    # these are complicated preallocations
    txt, dt_profile, dt_txt, dt = (
        np.zeros(allocateSize, dtype=object),
        np.zeros(allocateSize, dtype=object),
        np.zeros(allocateSize, dtype=object),
        np.zeros(allocateSize, dtype=object),
    )
    rangev = np.zeros((allocateSize * 2, allocateSize))  # arbitrary large value for time

    profile_data = np.zeros((allocateSize * 2, allocateSize))
    if verbose == 1:
        print(f"processing {len(dd)} s500 files data files")

    for fi in tqdm.tqdm(range(len(dd))):
        with open(dd[fi], "rb") as fid:
            fname = dd[fi]
            logging.debug(f"processing {fname}")
            xx = fid.read()
            st = [i + 1 for i in range(len(xx)) if xx[i : i + 2] == b"BR"]
            # initalize variables for loop
            packet_len, packet_id = np.zeros(len(st)), np.zeros(len(st))
            for ii in range(len(st) - 1):
                fid.seek(st[ii] + 1, os.SEEK_SET)
                datestring = fid.read(26).decode(
                    "utf-8", "replace"
                )  # 'replace' causes a replacement marker (such as '?')
                # to be inserted where there is malformed data.
                try:
                    dt[ii] = DT.datetime.strptime(datestring, "%Y-%m-%d %H:%M:%S.%f")
                except:
                    continue
                packet_len[ii] = struct.unpack("<H", fid.read(2))[0]
                packet_id[ii] = struct.unpack("<H", fid.read(2))[0]
                r1 = struct.unpack("<B", fid.read(1))[0]
                r2 = struct.unpack("<B", fid.read(1))[0]
                if packet_id[ii] == 1300:  # these are i believe ping sonar values
                    distance[ij] = struct.unpack("<I", fid.read(4))[0]  # mm
                    confidence[ij] = struct.unpack("<H", fid.read(2))[0]  # mm
                    transmit_duration[ij] = struct.unpack("<H", fid.read(2))[0]  # us
                    ping_number[ij] = struct.unpack("<I", fid.read(4))[0]  # #
                    scan_start[ij] = struct.unpack("<I", fid.read(4))[0]  # mm
                    scan_length[ij] = struct.unpack("<I", fid.read(4))[0]  # mm
                    gain_setting[ij] = struct.unpack("<I", fid.read(4))[0]
                    profile_data_length[ij] = struct.unpack("<I", fid.read(4))[0]
                    for jj in range(200):
                        tmp = struct.unpack("<B", fid.read(1))[0]
                        if tmp:
                            profile_data[ij, jj] = tmp
                    ij += 1

                if packet_id[ii] == 3:
                    txt[ij] = fid.read(int(packet_len[ii])).decode("utf-8")
                    dt_txt[ij] = dt
                if packet_id[ii] == 1308:  # these are s500 protocols
                    dtp = dt
                    # https://docs.ceruleansonar.com/c/v/s-500-sounder/appendix-f-programming-api#ping-response-packets
                    ping_number[ij] = struct.unpack("<I", fid.read(4))[0]  # mm
                    start_mm[ij] = struct.unpack("<I", fid.read(4))[0]  # mm
                    length_mm[ij] = struct.unpack("<I", fid.read(4))[0]  # mm
                    start_ping_hz[ij] = struct.unpack("<I", fid.read(4))[0]  # us
                    end_ping_hz[ij] = struct.unpack("<I", fid.read(4))[0]  # #
                    adc_sample_hz[ij] = struct.unpack("<I", fid.read(4))[0]  # mm
                    timestamp_msec[ij] = struct.unpack("I", fid.read(4))[0]
                    spare2[ij] = struct.unpack("I", fid.read(4))[0]

                    ping_duration_sec[ij] = struct.unpack("f", fid.read(4))[0]
                    analog_gain[ij] = struct.unpack("f", fid.read(4))[0]
                    max_pwr[ij] = struct.unpack("f", fid.read(4))[0]
                    min_pwr[ij] = struct.unpack("f", fid.read(4))[0]
                    step_db[ij] = struct.unpack("f", fid.read(4))[0]
                    smooth_depth_m[ij] = struct.unpack("f", fid.read(4))[0]
                    fspare2[ij] = struct.unpack("f", fid.read(4))[0]

                    is_db[ij] = struct.unpack("B", fid.read(1))[0]
                    gain_index[ij] = struct.unpack("B", fid.read(1))[0]
                    decimation[ij] = struct.unpack("B", fid.read(1))[0]
                    reserved[ij] = struct.unpack("B", fid.read(1))[0]
                    num_results[ij] = struct.unpack("H", fid.read(2))[0]
                    power_results[ij] = struct.unpack("H", fid.read(2))[0]
                    rangev[ij, 0 : num_results[ij]] = np.linspace(
                        start_mm[ij], start_mm[ij] + length_mm[ij], num_results[ij]
                    )
                    dt_profile[ij] = dt[ii]  # assign datetime from data written
                    # profile_data_single = [] #= np.empty((num_results[-1], ), dtype=np.uint16)
                    for jj in range(num_results[ij]):
                        # print(jj)
                        read = fid.read(2)
                        if read:
                            try:  # data should be unsigned short
                                tmp = struct.unpack("<H", read)[0]
                            except:  # when it's unsigned character
                                tmp = struct.unpack("B", read)[0]
                            if tmp:
                                profile_data[ij, jj] = tmp
                    ij += 1

    # clean up array's from over allocation to free up memory and data
    idxShort = (
        num_results != 0
    ).sum()  # np.argwhere(num_results != 0).max()  # identify index for end of data to keep
    num_results = np.median(num_results[:idxShort]).astype(int)  # num_results[:idxShort][0]

    # make data frame for output
    smooth_depth_m = smooth_depth_m[:idxShort]
    reserved = reserved[:idxShort]
    start_mm = start_mm[:idxShort]
    length_mm = length_mm[:idxShort]
    start_ping_hz = start_ping_hz[:idxShort]
    end_ping_hz = end_ping_hz[:idxShort]
    adc_sample_hz = adc_sample_hz[:idxShort]
    timestamp_msec = timestamp_msec[:idxShort]
    spare2 = spare2[:idxShort]
    ping_duration_sec = ping_duration_sec[:idxShort]
    analog_gain = analog_gain[:idxShort]
    max_pwr = max_pwr[:idxShort]
    min_pwr = min_pwr[:idxShort]
    step_db = step_db[:idxShort]
    fspare2 = fspare2[:idxShort]
    is_db = is_db[:idxShort]
    gain_index = gain_index[:idxShort]
    decimation = decimation[:idxShort]
    dt_profile = dt_profile[:idxShort]

    # rangev,  profile_data need to be handled separately
    rangev = rangev[0, :num_results]
    profile_data = profile_data[:idxShort, :num_results].T

    # now save output file (can't save as pandas because of multi-dimensional sonar data)
    if outfname is not None:
        with h5py.File(outfname, "w") as hf:
            hf.create_dataset("min_pwr", data=min_pwr)
            hf.create_dataset("ping_duration", data=ping_duration_sec)
            hf.create_dataset(
                "time", data=nc.date2num(dt_profile, "seconds since 1970-01-01")
            )  # TODO: confirm tz
            hf.create_dataset("smooth_depth_m", data=smooth_depth_m)
            hf.create_dataset("profile_data", data=profile_data)  # putting time as first axis
            hf.create_dataset("num_results", data=num_results)
            hf.create_dataset("start_mm", data=start_mm)
            hf.create_dataset("length_mm", data=length_mm)
            hf.create_dataset("start_ping_hz", data=start_ping_hz)
            hf.create_dataset("end_ping_hz", data=end_ping_hz)
            hf.create_dataset("adc_sample_hz", data=adc_sample_hz)
            hf.create_dataset("timestamp_msec", data=timestamp_msec)
            hf.create_dataset("analog_gain", data=analog_gain)
            hf.create_dataset("max_pwr", data=max_pwr)
            hf.create_dataset("this_ping_depth_m", data=step_db)
            hf.create_dataset("this_ping_depth_measurement_confidence", data=is_db)
            hf.create_dataset("smoothed_depth_measurement_confidence", data=reserved)
            hf.create_dataset("gain_index", data=gain_index)
            hf.create_dataset("decimation", data=decimation)
            hf.create_dataset("range_m", data=rangev / 1000)


def load_h5_to_dictionary(fname):
    """Loads already created H5 file from the sonar data."""
    hf = h5py.File(fname, "r")
    dataOut = {}
    for key in hf.keys():
        dataOut[key] = np.array(hf.get(key))
    hf.close()
    return dataOut


def makePOSfileFromRINEX(
    roverObservables, baseObservables, navFile, outfname, executablePath="rnx2rtkp", **kwargs
):
    """uses RTKLIB rnx2rtkp to post process
    Args:
        roverObservables: the RINEX format observables of the rover
        baseObservables: the RINEX format observables from the base station
        navFile: "at least one RINEX NAV/GNAV/HNAV file shall be included in input files."
        executablePath: path for rnx2rtkp (Default rnx2rtkp in local directory)

    Kwargs:
        'freq': 1 L1, 2 L1+L2; 3: L1+L2+L3 (Default=3)

    Assumes UTC time and yyyy/mm/dd hh:mm:ss.ss output time format

    References:
        https://rtkexplorer.com/pdfs/manual_demo5.pdf
    """
    # arguments for command here: https://rtkexplorer.com/pdfs/manual_demo5.pdf
    sp3 = kwargs.get("sp3", "")
    freq = kwargs.get("freq", 3)

    logging.debug(
        f"converting {os.path.basename(roverObservables)} using RTKLIB: Q=1:fix,2:float,3:sbas,4:dgps,5:single,6:ppp"
    )
    os.system(
        f"./{executablePath} -o {outfname} -t -u -f {freq} {roverObservables} {baseObservables} {navFile} {sp3}"
    )


def plot_single_backscatterProfile(
    fname, time, sonarRange, profile_data, this_ping_depth_m, smooth_depth_m, index
):
    """Create's a plot that shows full backscatter and individual profile  with identified depths

    :param fname:
    :param time:
    :param sonarRange:
    :param profile_data:
    :param this_ping_depth_m:
    :param smooth_depth_m:
    :param index:
    :return:
    """
    plt.figure(figsize=(12, 8))
    ax1 = plt.subplot2grid((4, 4), (0, 1), colspan=3, rowspan=4)
    backscatter = ax1.pcolormesh(time, sonarRange[0], profile_data.T, shading="auto")
    ax1.plot(time, this_ping_depth_m, color="black", ms=0.1, label="instant depth", alpha=0.5)
    # ax1.plot(time, smooth_depth_m, 'black', ms=1, label='Smooth Depth')
    ax1.plot(time[index], smooth_depth_m[index], ms=15, marker="X", color="red")
    cbar = plt.colorbar(mappable=backscatter, ax=ax1)
    cbar.set_label("backscatter value")
    ax1.set_ylim([0, 5])

    ax2 = plt.subplot2grid((4, 4), (0, 0), rowspan=4, sharey=ax1)
    ax2.plot(profile_data[index], sonarRange[0], alpha=1)
    ax2.plot(
        profile_data[index, np.argmin(np.abs(sonarRange[index] - this_ping_depth_m[index]))],
        this_ping_depth_m[index],
        "grey",
        marker="X",
        ms=10,
        label="this ping",
    )
    ax2.plot(
        profile_data[index, np.argmin(np.abs(sonarRange[index] - smooth_depth_m[index]))],
        smooth_depth_m[index],
        "black",
        marker="X",
        ms=10,
        label="smoothed bottom",
    )
    ax2.legend()

    for ii in range(5):
        ax2.plot(profile_data[index - ii], sonarRange[0], alpha=0.4 - ii * 0.07, color="k")
    ax2.set_ylabel("depth [m]")
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()


def mLabDatetime_to_epoch(dt):
    """Convert matlab datetime to unix Epoch time"""
    epoch = DT.datetime(1970, 1, 1)
    delta = dt - epoch
    return delta.total_seconds()


def convertEllipsoid2NAVD88(lats, lons, ellipsoids, geoidFile="g2012bu8.bin"):
    """converts elipsoid values to NAVD88's

     NOTE: if this is the first time you're using this, you'll likely have to go get the geoid bin file.  Code was
          developed using the uncompressed bin file.  It is unclear if the pygeodesy library requires the bin file to
          be uncompressed.  https://geodesy.noaa.gov/GEOID/GEOID12B/GEOID12B_CONUS.shtml

    :param lats:
    :param lons:
    :param ellipsoids: raw sattelite elipsoid values.
    :param geoidFile: pull from https://geodesy.noaa.gov/GEOID/GEOID12B/GEOID12B_CONUS.shtml
    :return: NAVD88 values
    """
    from pygeodesy import geoids

    assert (
        len(lons) == len(lats) == len(ellipsoids)
    ), "lons/lats/elipsoids need to be of same length"
    try:
        instance = geoids.GeoidG2012B(geoidFile)
    except ImportError:
        print(
            "if this is the first time you're using this, you'll likely have to go get the geoid bin file.  Code was "
            "developed using the uncompressed bin file.  It is unclear if the pygeodesy library requires the bin file to"
            " be uncompressed.  https://geodesy.noaa.gov/GEOID/GEOID12B/GEOID12B_CONUS.shtml"
        )
        import wget

        wget.download("[w]")
    geoidHeight = instance.height(lats, lons)
    return ellipsoids - geoidHeight


def load_yellowfin_NMEA_files(fpath:str, saveFname: str, plotfname: str = False, verbose: int=0) -> None:
    """loads and possibly plots NMEA data from Emlid Reach M2 on yellowin

    :param fpath: location to search for NMEA data files
    :param saveFname: where to save the Hdf5 file
    :param plotfname: where to save plot showing path of yellowfin, if False, will not plot (default=False)
    :param verbose: will print more output when processing if True (default=0), 0-warn, 1-info, 2-debug
    :return:
    """
    level=logging.WARN
    if verbose == 1:
        level=logging.INFO
    elif verbose == 2:
        level=logging.DEBUG
    logging.basicConfig(level=level)
    flist = glob.glob(os.path.join(fpath, "*.dat"))
    dd = sorted(
        [flist[os.path.getsize(i) > 0] for i in flist]
    )  # remove files of size zero from processing list
    if len(dd) == 0:  # if i didn't find it first, maybe i need to unpack
        try:  # try to move dat files out of a folder with the same name as the base folder  in the s500 folder
            fldInterest = [i for i in os.listdir(fpath) if ".dat" not in i][0]
            flist = glob.glob(
                os.path.join(
                    fpath,
                    fldInterest,
                    f"{fldInterest.split('-')[-1] + fldInterest.split('-')[0] + fldInterest.split('-')[1]}*.dat",
                )
            )
            if len(flist) < 1:  # to accomodate the august 2024 filename convention change
                flist = glob.glob(os.path.join(fpath, fldInterest, ''.join(fldInterest.split('-'))+"*.dat"))

            toDir = "/" + os.path.join(*flist[0].split(os.sep)[:-2])
            [shutil.move(l, toDir) for l in flist]
            # os.rmdir(os.path.join(dataPath, fldInterest)) # remove folder data came from
            dd = glob.glob(os.path.join(fpath, "*.dat"))
        except:
            raise EnvironmentError("The GNSS data date doesn't match folder date")

    logging.info(f"processing {len(dd)} GPS data files")


    ji = 0
    gps_time, lat, lon, altWGS84, altMSL, pc_time_gga = [], [], [], [], [], []
    lat, latHemi, lon, lonHemi, fixQuality, satCount, HDOP = [], [], [], [], [], [], []
    elevationMSL, eleUnits, geoSep, geoSepUnits, ageDiffGPS, diffRefStation = [], [], [], [], [], []
    for fi in tqdm.tqdm(range(1, len(dd))):
        fname = dd[fi]

        logging.debug(f" loading: {fname}")
        with open(fname, "r") as f:
            lns = f.readlines()

        for ln in lns:
            if ln.strip():
                try:
                    ss = ln.split("$")
                    datestring = ss[0].strip("#")
                    stringNMEA = ss[1].split(",")
                except IndexError:
                    continue

                try:
                    dt = DT.datetime.strptime(datestring.strip(), "%Y-%m-%d %H:%M:%S.%f")  # valueError
                except ValueError:
                    dt = DT.datetime.strptime(datestring.strip(), "%Y-%m-%d %H:%M:%S")  # valueError
                nmcode = stringNMEA[0]

                if nmcode == "GNGGA" and len(stringNMEA[2]) > 1:
                    # Sentence Identifier: This field identifies the type of NMEA sentence and is represented by "$GPGGA" for the GGA sentence.
                    # 1. UTC Time: This field provides the time in hours, minutes, and seconds in UTC.
                    # 2. Latitude: This field represents the latitude of the GPS fix in degrees and minutes, in the format of
                    #     ddmm.mmmm, where dd denotes degrees and mm.mmmm denotes minutes.
                    # 3. Latitude Hemisphere: This field indicates the hemisphere of the latitude, either "N" for North or "S" for South.
                    # 4. Longitude: This field represents the longitude of the GPS fix in degrees and minutes, in the format
                    # of dddmm.mmmm, where ddd denotes degrees and mm.mmmm denotes minutes.
                    # 5. Longitude Hemisphere: This field indicates the hemisphere of the longitude, either "E" for East or "W" for West.
                    # 6. GPS Fix Quality: This field provides information about the quality of the GPS fix, represented by a
                    # numeric value. Common values include 0 for no fix, 1 for GPS fix, and 2 for Differential GPS (DGPS) fix.
                    # 7. Number of Satellites in Use: This field indicates the number of satellites used in the GPS fix represented by a numeric value.
                    # 8. Horizontal Dilution of Precision (HDOP): This field represents the HDOP, which is a measure of the
                    # horizontal accuracy of the GPS fix, represented by a numeric value.
                    # 9 Altitude: This field provides the altitude above mean sea level (MSL) in meters, represented by a numeric value.
                    # 10 Altitude Units: This field indicates the units used for altitude, typically "M" for meters.
                    # 11 Geoidal Separation: This field represents the geoidal separation, which is the difference between
                    # the WGS84 ellipsoid and mean sea level, in meters, represented by a numeric value.
                    # 12Geoidal Separation Units: This field indicates the units used for geoidal separation, typically "M" for meters.
                    # 13 Age of Differential GPS Data: This field provides the age of the DGPS data used in the GPS fix, represented by a numeric value.
                    # 14 Differential Reference Station ID: This field indicates the identification number of the DGPS
                    # reference station used in the GPS fix, represented by a numeric value.
                    #
                    # parse the individual string, add to list
                    pc_time_gga.append(dt)
                    gps_time.append(float(stringNMEA[1]))
                    lat.append(float(stringNMEA[2][:2]) + float(stringNMEA[2][2:]) / 60)
                    latHemi.append(stringNMEA[3])
                    lona = float(stringNMEA[4][:3]) + float(stringNMEA[4][2:]) / 60
                    lonHemi.append(stringNMEA[5])
                    if lonHemi == "W":
                        lona = -lona
                    lon.append(lona)
                    fixQuality.append(
                        int(stringNMEA[6])
                    )  # GPS Fix Quality: represented by anumeric value. Common values
                    # include 0 for no fix, 1 for GPS fix, and 2 for Differential GPS (DGPS) fix.
                    satCount.append(int(stringNMEA[7]))
                    HDOP.append(
                        float(stringNMEA[8])
                    )  # measure of the horizontal accuracy of the GPS fix, represented
                    # by a numeric value.
                    altMSL.append(float(stringNMEA[9]))
                    eleUnits.append(stringNMEA[10])
                    geoSep.append(float(stringNMEA[11]))
                    geoSepUnits.append(stringNMEA[12])
                    ageDiffGPS.append(float(stringNMEA[13]))
                    diffRefStation.append(stringNMEA[14].strip())

    lat = np.array(lat)
    lon = np.array(lon)
    lat[lat == 0] = np.nan
    lon[lon == 0] = np.nan
    # convert datetimes to epochs for file writing.
    gpstimeobjs = [
        DT.time(int(str(ii)[:2]), int(str(ii)[2:4]), int(str(ii)[4:6]), int(str(ii)[7:] + "00000"))
        for ii in gps_time
    ]
    aa = [
        DT.datetime.combine(pc_time_gga[ii].date(), gpstimeobjs[ii])
        for ii in range(len(gpstimeobjs))
    ]
    # now save output file
    with h5py.File(saveFname, "w") as hf:
        hf.create_dataset("lat", data=lat)
        # hf.create_dataset('latHemi', data=latHemi)
        hf.create_dataset("lon", data=lon)
        # hf.create_dataset('lonHemi',data=lonHemi)
        hf.create_dataset("fixQuality", data=fixQuality)
        hf.create_dataset("satCount", data=satCount)
        hf.create_dataset("HDOP", data=HDOP)
        hf.create_dataset(
            "pc_time_gga",
            data=[mLabDatetime_to_epoch(pc_time_gga[ii]) for ii in range(len(pc_time_gga))],
        )
        hf.create_dataset("gps_time", data=[mLabDatetime_to_epoch(aa[i]) for i in range(len(aa))])
        hf.create_dataset("altMSL", data=altMSL)
        # hf.create_dataset('eleUnits', data=eleUnits) # putting time as first axis
        # hf.create_dataset('geoSepUnits', data=geoSepUnits)
        hf.create_dataset("ageDiffGPS", data=ageDiffGPS)
    # now plot data
    if plotfname is not False:
        plt.figure(figsize=(12, 4))
        plt.subplot(121)
        plt.plot(lon, lat, "-.")
        plt.subplot(122)
        # plt.plot(pc_time_gga, altWGS84, '.-')
        # plt.plot(pc_time_gga, geoSep, label='geoSep')
        plt.plot(pc_time_gga, altMSL, ".-", label="altMSL")
        plt.savefig(plotfname)
        plt.close()


def findTimeShiftCrossCorr(signal1, signal2, sampleFreq=1):
    """Finds time shift between two signals.

    :param signal1: a signal of same length of signal 2
    :param signal2: a signal of same length of signal 1
    :param sampleFreq: sampling frequency, in HZ
    :return: phase lag in samples, phase lag in time
    """
    import numpy as np
    from scipy.signal import correlate

    assert len(signal1) == len(signal2), "signals need to be the same lenth"
    # load your time series data into two separate arrays, let's call them signal1 and signal2.
    # compute the cross-correlation between the two signals using the correlate function:
    cross_corr = correlate(signal1, signal2)
    # Identify the index of the maximum value in the cross-correlation:
    max_index = np.argmax(np.abs(cross_corr))
    # Compute the phase lag in terms of the sample offset:
    phase_lag_samples = max_index - (len(signal1) - 1)
    # # If desired, convert the phase lag to time units (e.g., seconds):
    phase_lag_seconds = phase_lag_samples * sampleFreq
    return phase_lag_samples, phase_lag_seconds


def loadLLHfiles(flderlistLLH):
    # first load the LLH quick processed data
    T_LLH = pd.DataFrame()
    for fldr in sorted(flderlistLLH):
        # this is before ppk processing so should agree with nmea strings
        fn = glob.glob(os.path.join(fldr, "*"))[0]
        try:
            T = pd.read_csv(fn, delimiter="  ", header=None, engine="python")
            print(f"loaded {fn}")
            if all(T.iloc[-1]):  # if theres nan's in the last row
                T = T.iloc[:-1]  # remove last row
            T_LLH = pd.concat([T_LLH, T])  # merge multiple files to single dataframe

        except:
            continue

    T_LLH["datetime"] = pd.to_datetime(T_LLH[0], format="%Y/%m/%d %H:%M:%S.%f", utc=True)
    T_LLH["epochTime"] = T_LLH["datetime"].apply(lambda x: x.timestamp())

    T_LLH["lat"] = T_LLH[1]
    T_LLH["lon"] = T_LLH[2]
    return T_LLH


def butter_lowpass_filter(data, cutoff, fs, order):
    from scipy import signal

    b, a = signal.butter(order, cutoff / fs / 2, "low", analog=False)
    output = signal.filtfilt(b, a, data)

    # ormal_cutoff = cutoff / nyq
    # Get the filter coefficients
    # b, a = butter(order, normal_cutoff, btype='low', analog=False)
    # y = filtfilt(b, a, data)
    return output


def loadPPKdata(fldrlistPPK):
    """This function loads a single *.pos file per folder from a list of folders.  Each pos file in the folder has to be
    named the same as the folder name + .pos

    :param fldrlistPPK: a list of folders with ind
    :return: a data frame with loaded ppk data
    """

    T_ppk = pd.DataFrame()
    for fldr in sorted(fldrlistPPK):
        # this is before ppk processing so should agree with nmea strings
        # fn = glob.glob(os.path.join(fldr, "*.pos"))
        fn = os.path.join(fldr, os.path.basename(fldr).split("_R")[0] + ".pos")
        try:
            # colNames = [
            #     "datetime",
            #     "lat",
            #     "lon",
            #     "height",
            #     "Q",
            #     "ns",
            #     "sdn(m)",
            #     "sde(m)",
            #     "sdu(m)",
            #     "sdne(m)",
            #     "sdeu(m)",
            #     "sdun(m)",
            #     "age(s)",
            #     "ratio",
            # ]
            # col_widths=[(0,23), (26, 39), (40, 55), (56, 65), (66, 68), (70, 73)]
            # try:
            #     Tpos = pd.read_csv(fn, sep="\s{2,}", header=10, names=colNames, engine="python")
            # except ValueError:
            #     Tpos = pd.read_csv(fn, sep="\s{2,}", header=12, names=colNames, engine="python")
            colNames = ["date", "time", "lat", "lon", "height", "Q", "ns", "sdn(m)", "sde(m)", "sdu(m)", "sdne(m)",
                 "sdeu(m)", "sdun(m)", "age(s)", "ratio"]
            Tpos = pd.read_fwf(fn, skiprows=12, infer_nrows=1000, names=colNames) # fixed width reader
            logging.info(f"loaded {fn}")
            if all(Tpos.iloc[-1]):  # if theres nan's in the last row
                Tpos = Tpos.iloc[:-1]  # remove last row
            # Tpos["datetime"] = pd.to_datetime(Tpos['date'] + Tpos['time'], format="%Y/%m/%d%H:%M:%S.%f", utc=True)
            T_ppk = pd.concat([T_ppk, Tpos], ignore_index=True) # merge multiple files to single dataframe

        except:  # this is in the event there is no data in the pos files
            continue
    T_ppk["datetime"] = pd.to_datetime(T_ppk['date'] + T_ppk['time'], format="%Y/%m/%d%H:%M:%S.%f", utc=True)
    T_ppk["epochTime"] = T_ppk["datetime"].apply(lambda x: x.timestamp())
    return T_ppk


def unpackYellowfinCombinedRaw(fname):
    data = {}
    with h5py.File(fname, "r") as hf:
        for var in [
            "time",
            "longitude",
            "latitude",
            "elevation",
            "fix_quality_GNSS",
            "sonar_smooth_depth",
            "sonar_smooth_confidence",
            "sonar_instant_depth",
            "sonar_instant_depth",
            "sonar_instant_confidence",
            "sonar_backscatter_out",
            "bad_lat",
            "bad_lon",
            "xFRF",
            "yFRF",
            "Profile_number",
        ]:
            data[var] = hf.get(var)[:]
    return data


def plotPlanViewOnArgus(data, geoTifName, ofName=None, argus_time_out_s=120):
    """plots a survey path over a geotiff at the FRF (assumes NC stateplane)
    Args:
        data: this is a dictionary of data loaded with keys of 'longitude', 'latitude', 'elevation'
        geoTifName: this is a filenamepath of a geotiff file over which elevation and path data are to be overlayed
        ofname: this is the plot output save name/location (default=None)
        argus_time_out_s: time to wait to pull argus imagery before timing out, in seconds (default=120)

    References
        https://pratiman-91.github.io/2020/06/30/Plotting-GeoTIFF-in-python.html

    """
    coords = geoprocess.FRFcoord(data["Longitude"], data["Latitude"])
    tt = 0
    while not os.path.isfile(
        geoTifName
    ):  # this is waiting for the file to show up, if the download is threaded
        time.sleep(30)
        tt += 30
        print(f"waited for {tt} seconds for {geoTifName}")
        if tt >= argus_time_out_s:
            print(f"Timed out after {tt} seconds waiting for {geoTifName}. Image not found.")
            return

    timex = rasterio.open(geoTifName)
    # array = timex.read()  # for reference, this pulls the image data out of the geotiff object
    ## now make plot
    plt.figure(figsize=(14, 10))
    ax1 = plt.subplot()
    aa = rplt.show(timex, ax=ax1)
    a = ax1.scatter(coords["StateplaneE"], coords["StateplaneN"], c=data["Elevation"], vmin=-8)
    cbar = plt.colorbar(a)
    cbar.set_label("depths")
    ax1.set_xlabel("NC stateplane Easting")
    ax1.set_ylabel("NC stateplane Northing")
    if ofName is None:
        ofName = os.path.join(os.getcwd(), "Overview_on_Argus.png")
    plt.savefig(ofName)
    plt.close()


def getArgusImagery(dateOfInterest, ofName=None, imageType="timex", verbose=True):
    if verbose: logging.basicConfig(level=logging.INFO)
    # client = Minio("coastalimaging.erdc.dren.mil")
    # ## now lets find what files are around
    # objects = client.list_objects('FrfTower', prefix="Processed/alignedObliques/c1", recursive=True,)
    baseURL = "https://coastalimaging.erdc.dren.mil/FrfTower/Processed/Orthophotos/cxgeo/"
    fldr = dateOfInterest.strftime("%Y_%m_%d")
    fname = f'{dateOfInterest.strftime("%Y%m%dT%H%M%SZ")}.FrfTower.cxgeo.{imageType}.tif'

    logging.info(f"retreiving {baseURL + fldr + fname}")
    wgetURL = os.path.join(baseURL, fldr, fname)
    if ofName is None:
        ofName = os.path.join(os.getcwd(), os.path.basename(wgetURL))
    try:
        wget.download(wgetURL, ofName)
    except:
        pass

    logging.debug(f"retrieved {ofName}")
    return ofName


def threadGetArgusImagery(dateOfInterest, ofName=None, imageType="timex", verbose=True):
    if ofName is None:
        ofName = os.path.join(
            os.getcwd(), f'Argus_{imageType}_{dateOfInterest.strftime("%Y%m%dT%H%M%SZ")}.tif'
        )
    t = threading.Thread(
        target=getArgusImagery, args=[dateOfInterest, ofName, imageType, verbose], daemon=True
    )
    t.start()
    return ofName


def transectSelection(data, **kwargs):
    """
    Args:
        data: dataframe containing crawler transect data, to be modified with isTransect and profileNumber columns

    Keyword Args:
        'outputDir': this is the output directory for the file name save
    Returns:
        data: input dataframe with columns isTransect and profileNumber added, isTransect is a boolean denoting
        whether a point is part of a transect, profileNumber is a float to designate the transect a point is a part of
        typically the mean FRFy coordinate of the transect, points not part of a transect are assigned a profileNumber
        of Nan
    """
    outputDir = kwargs.get("outputDir", os.getcwd())
    plotting = kwargs.get("savePlots", True)
    # For date string in saved plot file name, convert UNIX timestamp → datetime → YYYYMMDD string
    ts = DT.datetime.fromtimestamp(float(data['date'][0]), tz=DT.timezone.utc)
    ts_yyyymmdd = ts.strftime("%Y%m%d")  # e.g., 20240716
    # added columns for isTransect boolean and profileNumber float to data dataframe
    data["isTransect"] = [False] * data.shape[0]
    data["profileNumber"] = [float("nan")] * data.shape[0]
    # create copy of data for display, points are removed from the frame once identified as part of a transect
    dispData = data.copy(deep=True)
    # create copies of data and display data to hold previous version for undo function
    prevDisp = data.copy(deep=True)
    prevData = data.copy(deep=True)
    # main loop for identifying transects, continues to allow for selections while user inputs y/Y
    transectIdentify = input("Do you want to select a transect? (Y/N):")
    while transectIdentify.lower() == "y" or transectIdentify.lower() == "u":
        if transectIdentify.lower() == "y":
            pointsValid = True
            print(
                "To identify a transect, please place a single point at the start and end of the transect with left click"
            )
            print(
                "Right click to erase the most recently selected point. Middle click (press the scroll wheel) to save."
            )
            print(
                "Points have saved when they no longer appear on the graph, close the graph window to proceed."
            )
            print("Remember to remove points used in zooming and panning with right click.")
            print("If more or less than 2 points are selected, no changes will be made")
            print(
                "Each graph's colorscale represents the y axis of the other graph, i.e. the colorscale of the xy graph is time, and vice versa"
            )
            print("Select the transect using only 1 graph at a time")
            # displays plots of two subplots, one with x vs y colored in time and one with time vs y colored in x
            fig = plt.figure()
            fig.suptitle("Transects xFRF (top) and time (bottom) vs yFRF ")
            shape = (4, 6)

            ax0 = plt.subplot2grid(shape, (0, 0), colspan=2, rowspan=5)
            ax0.scatter(dispData["xFRF"], dispData["yFRF"], c=dispData["time"], cmap="hsv", s=1)
            ax0.set(xlabel="xFRF [m]", ylabel="yFRF [m]")

            ax1 = plt.subplot2grid(shape, (0, 2), colspan=6, rowspan=2)
            ax1.scatter(
                dispData["UNIX_timestamp"], dispData["xFRF"], c=dispData["yFRF"], cmap="hsv", s=1
            )
            ax1.set(xlabel="UNIX Timestamp (seconds)", ylabel="xFRF (m)")
            ax2 = plt.subplot2grid(shape, (2, 2), colspan=6, rowspan=2, sharex=ax1)
            ax2.scatter(
                dispData["UNIX_timestamp"], dispData["yFRF"], c=dispData["yFRF"], cmap="hsv", s=1
            )
            ax2.set(xlabel="UNIX Timestamp (seconds)", ylabel="yFRF (m)")
            plt.tight_layout()
            nodes = plt.ginput(-1, 0)
            print("Selected Points: ")
            print(nodes)
            plt.close()
            # ginput returns list of tuples of selected coordinates, each is in its graph's proper scale
            if len(nodes) == 2:
                prevDisp = dispData.copy(deep=True)
                prevData = data.copy(deep=True)
                # false means ycoord is yFRF, true means UNIX Timestamp
                isTime = [False, False]
                isTime[0] = nodes[0][0] > 1500
                isTime[1] = nodes[1][0] > 1500
                if isTime[0] == isTime[1]:
                    endpts = []
                    # each node is matched to the closest point in the dispData dataframe
                    for x in range(len(nodes)):
                        curr = nodes[x]
                        prevDist = float("inf")
                        closest = tuple()
                        for y in range(dispData.shape[0]):
                            if isTime[x]:
                                dist = np.sqrt(
                                    (dispData["UNIX_timestamp"][y] - curr[0]) ** 2
                                    + (dispData["yFRF"][y] - curr[1]) ** 2
                                )
                            else:
                                dist = np.sqrt(
                                    (dispData["yFRF"][y] - curr[1]) ** 2
                                    + (dispData["xFRF"][y] - curr[0]) ** 2
                                )
                            if dist < prevDist:
                                prevDist = dist
                                closest = (dispData["xFRF"][y], dispData["yFRF"][y])
                        endpts.append(closest)

                    # identify endpoints within dispdata frame
                    isEndPt = []
                    for x in range(dispData.shape[0]):
                        if (dispData["xFRF"][x], dispData["yFRF"][x]) in endpts:
                            isEndPt.append(True)
                        else:
                            isEndPt.append(False)
                    dispData["endPt"] = isEndPt
                    # endPt column identifies where each transect starts and stops

                    # identify transect within dispdata
                    isTransect = []
                    betweenNodes = False
                    for x in range(dispData.shape[0]):
                        if dispData["endPt"][x] and not betweenNodes:
                            # first node in time of transect
                            betweenNodes = True
                            isTransect.append(True)
                        elif dispData["endPt"][x] and betweenNodes:
                            # last node in time of transect
                            betweenNodes = False
                            isTransect.append(True)
                        else:
                            isTransect.append(betweenNodes)
                    dispData["isTransect"] = isTransect

                    # assign id to current transect
                    currTransect = dispData.loc[dispData["isTransect"] == True]
                    # remove newly assigned transect from display dataframe
                    dispData = dispData.loc[dispData["isTransect"] == False]
                    dispData = dispData.reset_index(drop=True)
                    meanY = np.mean(currTransect["yFRF"])
                    print("Close the window to continue.")
                    # plt.figure()
                    # plt.hist(currTransect["yFRF"])
                    # plt.title("FRFy coords of selected transect")
                    # plt.show()
                    print("Mean FRFy coord of selected transect: ", meanY)
                    transectIDstr = input(
                        "What profile number would you like to assign this transect? (float type, press ENTER for mean FRFy): "
                    )
                    transectID = meanY
                    if transectIDstr != "":  # if the response is not enter
                        transectID = float(transectIDstr)
                    currTransect["profileNumber"] = currTransect["profileNumber"].replace(
                        [float("nan")], transectID
                    )

                    print("Updating dataframe...")
                    # update primary dataframe
                    # search once to find first timestamp, iterate afterwards
                    startTime = currTransect["UNIX_timestamp"].iloc[0]
                    endTime = currTransect["UNIX_timestamp"].iloc[currTransect.shape[0] - 1]
                    firstI = 0
                    for y in range(data.shape[0]):
                        if data["UNIX_timestamp"].iloc[y] == startTime:
                            firstI = y
                            break

                    for x in range(currTransect.shape[0]):
                        data.loc[x + firstI, "profileNumber"] = transectID
                        data.loc[x + firstI, "isTransect"] = True
                else:
                    # ignore selected points if from different plots
                    print("Selected points from different plots. Discarding selected points.")
                    pointsValid = False
            else:
                # ignore selected points if more or less than 2 selected
                print("Selected more or less than 2 points. Discarding selected points.")
                pointsValid = False

            # display selected transects overlayed over all points, colored by profile number
            print("Displaying current progress. Close the window to continue.")
            transectsOnly = data.loc[data["isTransect"] == True]
            plt.figure()
            plt.scatter(data["xFRF"], data["yFRF"], c="black", s=1)
            a = plt.scatter(
                transectsOnly["xFRF"],
                transectsOnly["yFRF"],
                c=transectsOnly["profileNumber"].to_list(),
                cmap="hsv",
                s=1,
            )
            if pointsValid:
                plt.scatter(currTransect["xFRF"], currTransect["yFRF"], c="pink", marker="x")
            cbar = plt.colorbar(a)
            plt.xlabel("FRF Coordinate System X (m)")
            plt.ylabel("FRF Coordinate System Y (m)")
            cbar.set_label("Transect Number")
            plt.title("Current Progress")
            plt.show()
            transectIdentify = input(
                "Do you want to select another transect? yes-y, No-N, undo-u :"
            )

        elif transectIdentify.lower() == "u":
            # undo case
            dispData = prevDisp
            data = prevData
            print("Displaying current progress. Close the window to continue.")
            transectsOnly = data.loc[data["isTransect"] == True]
            plt.figure()
            plt.scatter(data["xFRF"], data["yFRF"], c="black", s=1)
            a = plt.scatter(
                transectsOnly["xFRF"],
                transectsOnly["yFRF"],
                c=transectsOnly["profileNumber"].to_list(),
                cmap="hsv",
                s=1,
            )
            cbar = plt.colorbar(a)
            plt.xlabel("FRF Coordinate System X (m)")
            plt.ylabel("FRF Coordinate System Y (m)")
            cbar.set_label("Transect Number")
            plt.title("Current Progress")
            plt.show()
            transectIdentify = input("Do you want to select another transect? (Y/N):")
        else:
            raise NotImplementedError("required inputs are (y)es/(n)o/(u)ndo")
    # prompts for saving charts, excel and pickle

    if plotting is True:
        transectsOnly = data.loc[data["isTransect"] == True]
        print("Close the window to continue.")
        plt.figure(figsize=(8, 16))
        a = plt.scatter(
            transectsOnly["xFRF"],
            transectsOnly["yFRF"],
            c=transectsOnly["profileNumber"].to_list(),
            cmap="hsv",
            s=1,
        )
        cbar = plt.colorbar(a)
        plt.xlabel("xFRF (m)")
        plt.ylabel("yFRF (m)")
        cbar.set_label("Transect Number")
        plt.title(f"Crawler Survey {ts_yyyymmdd}")
        plt.savefig(
            os.path.join(
                outputDir,
                f"Processed_linesWithNumbers_{ts_yyyymmdd}.png",
            )
        )
        plt.close()

        print("Close the window to continue.")
        plt.figure(figsize=(8, 16))
        plt.scatter(data["xFRF"], data["yFRF"], c="black", s=1)
        plt.scatter(
            transectsOnly["xFRF"],
            transectsOnly["yFRF"],
            c=transectsOnly["profileNumber"].to_list(),
            cmap="hsv",
            s=1,
        )
        cbar = plt.colorbar()
        plt.xlabel("xFRF (m)")
        plt.ylabel("yFRF (m)")
        cbar.set_label("Profile Number")
        plt.title(
            f"Identified Transects vs All Points\n{ts_yyyymmdd}"
        )
        plt.savefig(
            os.path.join(
                outputDir,
                f"Processed_linesWithAllData_{ts_yyyymmdd}.png",
            )
        )
        plt.close()

    return data


def plot_planview_FRF(ofname, coords, gnss_out, antenna_offset, elevation_out, sonar_instant_depth_out, sonar_smooth_depth_out, idxDataToSave):

        minloc = 800
        maxloc = 1000
        logic = (coords['yFRF'] > minloc) & (coords['yFRF'] < maxloc)


        plt.figure(figsize=(12, 8))
        plt.subplot(211)
        plt.title('plan view of survey')
        plt.scatter(coords['xFRF'], coords['yFRF'], c=elevation_out[idxDataToSave], vmax=-1)
        cbar = plt.colorbar()
        cbar.set_label('depth')
        plt.subplot(212)
        plt.title(f"profile at line y={np.median(coords['yFRF'][logic]).astype(int)}")
        plt.plot(coords['xFRF'][logic],
                 gnss_out[idxDataToSave][logic] - antenna_offset - sonar_instant_depth_out[idxDataToSave][logic], '.',
                 label='instant depths')
        plt.plot(coords['xFRF'][logic],
                 gnss_out[idxDataToSave][logic] - antenna_offset - sonar_smooth_depth_out[idxDataToSave][logic], '.',
                 label='smooth Depth')
        plt.plot(coords['xFRF'][logic], elevation_out[idxDataToSave][logic], '.', label='chosen depths')
        plt.legend()
        plt.xlabel('xFRF')
        plt.ylabel('elevation NAVD88[m]')
        plt.tight_layout()
        plt.savefig(ofname)

def plot_planview_lonlat(ofname, T_ppk, bad_lon_out, bad_lat_out, elevation_out, lat_out, lon_out, timeString, idxDataToSave, FRF, margin=0.1):
        fs = 16
        # make a final plot of all the processed data
        pierStart = geoprocess.FRFcoord(0, 515, coordType='FRF')
        pierEnd = geoprocess.FRFcoord(534, 515, coordType='FRF')

        plt.figure(figsize=(12, 8))
        min_elev = np.min(elevation_out[idxDataToSave])
        max_elev = np.max(elevation_out[idxDataToSave])
        diff_elev = max_elev - min_elev
        buffer = margin*diff_elev
        if abs(max_elev) < 1:
            # indicates a sea level survey, ceiling color map at 0.5m below 0m NAVD88 elevation
            # we assume max_elev will not be significantly greater than 1m elevation
            if max_elev > 0:
                vmax = max_elev*1.1
            else:
                vmax = 0
        else:
            # survey taken at an elevated location, add a margin of survey range (default 10%) to colorbar for readability
            vmax = max_elev+buffer
        vmin = min_elev-buffer
        plt.scatter(lon_out[idxDataToSave], lat_out[idxDataToSave], c=elevation_out[idxDataToSave], 
                    vmax=vmax, vmin=vmin, label='processed depths')
        cbar = plt.colorbar()
        cbar.set_label('NAVD88 Elevation [m]', fontsize=fs)
        plt.plot(T_ppk['lon'], T_ppk['lat'], 'k.', ms=0.25, label='vehicle trajectory')
        plt.plot(bad_lon_out, bad_lat_out, 'rx', ms=3, label='bad sonar data, good GPS')
        if FRF == True:
            plt.plot([pierStart['Lon'], pierEnd['Lon']], [pierStart['Lat'], pierEnd['Lat']], 'k-', lw=5, label='FRF pier')
        plt.ylabel('latitude', fontsize=fs)
        plt.xlabel('longitude', fontsize=fs)
        plt.title(f'final data with elevations {timeString}', fontsize=fs + 4)
        plt.tight_layout()
        plt.legend()
        plt.gca().ticklabel_format(useOffset=False, style='plain')
        plt.savefig(ofname)

def qaqc_post_sonar_time_shift(ofname, T_ppk, indsPPK, commonTime, ppkHeight_i, sonar_range_i, phaseLaginTime,
                               sonarData, sonarIndicies, sonar_range):
    # TODO pull this figure out to a function
    plt.figure(figsize=(16, 8))
    ax1 = plt.subplot(311)
    plt.plot(T_ppk['epochTime'][indsPPK], T_ppk['GNSS_elevation_NAVD88'][indsPPK], label='ppk elevation NAVD88 m')
    plt.plot(sonarData['time'][sonarIndicies], sonar_range[sonarIndicies], label='sonar_raw')
    plt.legend()

    plt.subplot(312, sharex=ax1)
    plt.title(f"sonar data needs to be adjusted by {phaseLaginTime} seconds")
    plt.plot(commonTime, signal.detrend(ppkHeight_i), label='ppk input')
    plt.plot(commonTime, signal.detrend(sonar_range_i), label='sonar input')
    plt.plot(commonTime + phaseLaginTime, signal.detrend(sonar_range_i), '.', label='interp _sonar shifted')
    plt.legend()

    plt.subplot(313, sharex=ax1)
    plt.title('shifted residual between sonar and GNSS (should be 0)')
    plt.plot(commonTime + phaseLaginTime, signal.detrend(sonar_range_i) - signal.detrend(ppkHeight_i), '.',
             label='residual')
    plt.ylim([-1, 1])
    plt.tight_layout()
    plt.show()
    plt.savefig(ofname)

def is_local_to_FRF(coords):
    return ((coords['yFRF'] < 2000) & (coords['yFRF'] > -20000)).all()

def qaqc_time_offset_determination(ofname, pc_time_off):
    # Compare GPS data to make sure timing is ok
    plt.figure()
    plt.suptitle('time offset between pc time and GPS time')
    ax1 = plt.subplot(121)
    ax1.plot(pc_time_off, '.')
    ax1.set_xlabel('PC time')
    ax1.set_ylabel('PC time - GGA string time (+leap seconds)')
    ax2 = plt.subplot(122)
    ax2.hist(pc_time_off, bins=50)
    ax2.set_xlabel('diff time')
    plt.tight_layout()
    plt.savefig(ofname)
    print(f'the PC time (sonar time stamp) is {np.median(pc_time_off):.2f} seconds behind the GNSS timestamp')
    plt.close()

def sonar_pick_cross_correlation_time(ofname, sonar_range):
    plt.figure(figsize=(10, 4))
    plt.subplot(211)
    plt.title('all data: select start/end point for measured depths to do time-syncing over ')
    plt.plot(sonar_range)
    plt.ylim([0, 10])
    d = plt.ginput(2, timeout=-999)
    plt.subplot(212)
    # Now pull corresponding indices for sonar data for same time
    assert len(d) == 2, "need 2 points from mouse clicks"
    sonarIndicies = np.arange(np.floor(d[0][0]).astype(int), np.ceil(d[1][0]).astype(int))
    plt.plot(sonar_range[sonarIndicies])
    plt.title('my selected data to proceed with cross-correlation/time syncing')
    plt.tight_layout()
    plt.savefig(ofname)
    return sonarIndicies

def qaqc_plot_all_data_in_time(ofname, sonarData, sonar_range, payloadGpsData, T_ppk):
    # 6.6 Now lets take a look at all of our data from the different sources
    plt.figure(figsize=(10, 4))
    plt.suptitle('all data sources elevation', fontsize=20)
    plt.title('These data need to overlap in time for processing to work')
    plt.plot([DT.datetime.fromtimestamp(float(i), tz=DT.timezone.utc) for i in sonarData['time']], sonar_range, 'b.', label='sonar depth')
    plt.plot([DT.datetime.fromtimestamp(float(i), tz=DT.timezone.utc) for i in payloadGpsData['gps_time']], payloadGpsData['altMSL'], '.g',
             label='L1 (only) GPS elev (MSL)')
    plt.plot([DT.datetime.fromtimestamp(float(i), tz=DT.timezone.utc) for i in T_ppk['epochTime']], T_ppk['GNSS_elevation_NAVD88'], '.r',
             label='ppk elevation [NAVD88 m]')
    plt.ylim([0, 10])
    plt.ylabel('elevation [m]')
    plt.xlabel('epoch time (s)')
    plt.legend()
    # plt.show()
    plt.savefig(ofname)

def qaqc_sonar_profiles(ofname, sonarData):

    plt.figure(figsize=(18, 6))
    cm = plt.pcolormesh([DT.datetime.fromtimestamp(float(i), tz=DT.timezone.utc) for i in sonarData['time']], sonarData['range_m'],
                        sonarData['profile_data'])
    cbar = plt.colorbar(cm)
    cbar.set_label('backscatter')
    plt.plot([DT.datetime.fromtimestamp(float(i), tz=DT.timezone.utc) for i in sonarData['time']], sonarData['this_ping_depth_m'], 'r-', lw=0.1,
             label='this ping Depth')
    plt.plot([DT.datetime.fromtimestamp(float(i), tz=DT.timezone.utc) for i in sonarData['time']], sonarData['smooth_depth_m'], 'k-', lw=0.5,
             label='smooth Depth')
    plt.ylim([10, 0])
    plt.legend(loc='lower left')
    # plt.gca().invert_yaxis()
    plt.tight_layout(rect=[0.05, 0.05, 0.99, 0.99], w_pad=0.01, h_pad=0.01)
    plt.savefig(ofname)
