# barebones workflow to fuse gps, imu, and sonar data into survey product
# assumes IMU csv from data-alignment/mavlogparse.py, sonar data h5 with QAQC'ed depths
import numpy as np
import pandas as pd
import utm
import h5py

from numpy.linalg import multi_dot
from yellowfinLib import *

eps = 1e-16

# helper function for rectify_sonar_position to produce DH parameter transformation matrix
# all displacements assumed to be in meters, all angles in radians
def DH_mtx_maker(a_i, alpha_i, d_i, theta_i):
    Hi = np.squeeze(np.asarray([[np.cos(theta_i), -np.sin(theta_i) * np.cos(alpha_i), np.sin(theta_i) * np.sin(alpha_i),
                                 a_i * np.cos(theta_i)],
                                [np.sin(theta_i), np.cos(theta_i) * np.cos(alpha_i), -np.cos(theta_i) * np.sin(alpha_i),
                                 a_i * np.sin(theta_i)],
                                [0.0, np.sin(alpha_i), np.cos(alpha_i), d_i],
                                [0.0, 0.0, 0.0, 1.0]]))
    Hi[np.abs(Hi) < eps] = 0.0
    return Hi

'''
Function to find the global position (lat, lon, depth) of sonar measurements.
- sonar_df: dataframe with sonar depth measurements and interpolated timesynced GPS and IMU data
Outputs:
- pandas dataframe containing global position rectified sonar measurements, rectified_df
'''
def rectify_sonar_position(merged_df):

    # blueboat/sounder geometry constants. For all, origin is at GPS antenna
    # TODO: measure location for ect-D032 and modify to toggle between sonar geometries
    gps_to_sonar_vert_offset_m = 0.73025 # vertical distance from gps antenna to sounder head, positive down. Measured as 28.75", does not include PSO
    # TODO: find PSO information for VSP6037L-MAR
    gps_to_sonar_len_offset_m = 0.32385 # alongbody distance from gps antenna to sounder head, positive from stern to bow. Measured as 12.75"
    gps_to_sonar_width_offset_m = 0.6858 # crossbody distance from gps antenna to sounder head, positive from port to starboard. Measured as 27"

    # constant transformation matrices (do not depend on vehicle attitude or sonar reading)
    # reference frame zero: x-north, y-west, z-up. Positive x is meters north, negative y is meters east
    # HO1: transformation from North-West aligned to vehicle aligned using yaw angle
    H12 = DH_mtx_maker(0, np.deg2rad(90), 0, 0)  # rotation to z-down
    # H23: rotation of x-z plane from parallel to North-East plane to vehicle attitude-aligned
    H34 = DH_mtx_maker(0, 0, gps_to_sonar_vert_offset_m, 0) # translation to elevation of sonar head
    H45 = DH_mtx_maker(gps_to_sonar_len_offset_m, 0, 0, 0) # translation to alongbody position of sonar head
    H56 = DH_mtx_maker(0, np.deg2rad(-90), 0, 0) # rotation to z-crossbody aligned
    H67 = DH_mtx_maker(0, 0, gps_to_sonar_width_offset_m, 0) # translation to crossbody position of sonar head
    H37 = multi_dot([H34, H45, H56, H67]) # transformation from gps antenna, z-up to sonar head, z-crossbody
    # H78: rotation to z aligned with sonar direction
    # H89: translation to bottom return location

    beam_width_deg = 5 # angle swept by s500 sonar, TODO: modify to toggle between s500 and ect-D032 beam widths
    beam_width_rad = np.deg2rad(beam_width_deg)

    # arrays to save new csv
    time = []
    depth = []
    yaw = []
    pitch = []
    roll = []
    rect_lat = []
    rect_lon = []
    rect_elev = []

    for index, row in merged_df.iterrows():
        time.append(row['timestamp'])
        sonar_dist = row['Depth QAQC (m)']
        depth.append(sonar_dist)
        # IMU angles in radians
        curr_yaw_rad = row['yaw']
        curr_pitch_rad = row['pitch']
        curr_roll_rad = row['roll']
        yaw.append(curr_yaw_rad)
        pitch.append(curr_pitch_rad)
        roll.append(curr_roll_rad)

        # constant transformation matrices (do not depend on vehicle attitude or sonar reading)
        # reference frame zero: x-north, y-west, z-up. Positive x is meters north, negative y is meters east
        # HO1: transformation from North-West aligned to vehicle aligned using yaw angle
        theta_0 = curr_yaw_rad
        H01 = DH_mtx_maker(a_i=0.0, alpha_i=0.0, d_i=0.0, theta_i=theta_0)

        # H23: rotation of x-z plane from parallel to North-East plane to vehicle attitude-aligned
        theta_2 = curr_pitch_rad
        alpha_2 = np.deg2rad(90) + curr_pitch_rad
        H23 = DH_mtx_maker(a_i=0.0, alpha_i=alpha_2, d_i=0.0, theta_i=theta_2)

        H03 = multi_dot([H01, H12, H23])

        H07 = H03.dot(H37)

        # H78: rotation to z aligned with sonar direction
        if abs(np.rad2deg(curr_roll_rad)) <= beam_width_deg:
            alpha_8 = np.deg2rad(90) - curr_roll_rad
        elif np.rad2deg(curr_roll_rad) > 0: # && |curr_roll_deg| > beam_width_deg
            alpha_8 = np.deg2rad(90) - beam_width_rad
        else: # curr_roll_deg < 0 && |curr_roll_deg| > beam_width_deg
            alpha_8 = np.deg2rad(90) + beam_width_rad

        if abs(np.rad2deg(curr_yaw_rad)) <= beam_width_deg:
            theta_8 = -curr_yaw_rad
        elif np.rad2deg(curr_yaw_rad) > 0: # && |curr_yaw_deg| > beam_width_deg
            theta_8 = -beam_width_rad
        else: # curr_yaw_deg < 0 && |curr_yaw_deg| > beam_width_deg
            theta_8 = beam_width_rad

        H78 = DH_mtx_maker(a_i=0.0, alpha_i=alpha_8, d_i=0.0, theta_i=theta_8)
        # H89: translation to bottom return location
        H89 = DH_mtx_maker(a_i=0.0, alpha_i=0.0, d_i=sonar_dist, theta_i=0.0)

        H09 = multi_dot([H07, H78, H89])

        # XYZ position of the sonar reading in the bot reference frame
        sonar_pos_0 = np.dot(H09, np.squeeze(np.asarray([[0.0], [0.0], [0.0], [1.0]])))

        north_pos_m = sonar_pos_0[0]
        west_pos_m = sonar_pos_0[1]
        vert_disp_m = sonar_pos_0[2]

        EASTING, NORTHING, ZONE_NUMBER, ZONE_LETTER = utm.from_latlon(row['lat'], row['lon'])

        EASTING -= west_pos_m # positive values of west_pos_m indicate sonar reading is to the west of the GPS location
        NORTHING += north_pos_m

        sonar_lat, sonar_lon = utm.to_latlon(EASTING, NORTHING, ZONE_NUMBER, ZONE_LETTER)

        rect_lat.append(sonar_lat)
        rect_lon.append(sonar_lon)
        if sonar_dist >= 0: # represents a valid depth measurement
            rect_elev.append(row['elev'] + vert_disp_m)
        else:
            rect_elev.append(np.nan)

    to_save = {"Timestamp": time,
               "GPS Lat": merged_df['lat'],
               "GPS Lon": merged_df['lon'],
               "GPS Height (m)": merged_df['elev'],
               "Sonar Depth (m)": depth,
               "Yaw": yaw,
               "Pitch": pitch,
               "Roll": roll,
               "Rectified Lat": rect_lat,
               "Rectified Lon": rect_lon,
               "Rectified Elevation (m)": rect_elev}
    rectified_df = pd.DataFrame.from_dict(to_save)
    rectified_df = rectified_df[abs(np.rad2deg(rectified_df["Roll"])) < 90]
    rectified_df = rectified_df[abs(np.rad2deg(rectified_df["Pitch"])) < 90]
    rectified_df = rectified_df.reset_index(drop=True)
    return rectified_df


# function to match GPS and IMU data in time to sonar data points, through linear interpolation
# based off implementation in BathybotDataProcessingLib
# currently data is cropped by the interpolation to the length of the sonar data file, meaning sections
# with valid GPS data before and after the span of the sonar file (ex, sonar stopped collecting)
# are not included in the final survey file
def merge_pos_sonar_imu_data(sonar_df, pos_df, imu_df):
    pos_df = pos_df.dropna()

    pos_df = pos_df.reset_index(drop=True)

    sonar_lat = np.interp(sonar_df["timestamp"], pos_df["timestamp"], pos_df["lat"], left=np.nan, right=np.nan)
    sonar_lon = np.interp(sonar_df["timestamp"], pos_df["timestamp"], pos_df["lon"], left=np.nan, right=np.nan)
    sonar_elev = np.interp(sonar_df["timestamp"], pos_df["timestamp"], pos_df["height"], left=np.nan, right=np.nan)
    sonar_yaw = np.interp(sonar_df["timestamp"], imu_df["timestamp"], imu_df["ATTITUDE.yaw"], left=np.nan, right=np.nan)
    sonar_pitch = np.interp(sonar_df["timestamp"], imu_df["timestamp"], imu_df["ATTITUDE.pitch"], left=np.nan, right=np.nan)
    sonar_roll = np.interp(sonar_df["timestamp"], imu_df["timestamp"], imu_df["ATTITUDE.roll"], left=np.nan, right=np.nan)

    sonar_df = sonar_df.dropna()
    sonar_df = sonar_df.reset_index(drop=True)

    sonar_df["lat"] = sonar_lat
    sonar_df["lon"] = sonar_lon
    sonar_df["elev"] = sonar_elev

    sonar_df["yaw"] = sonar_yaw
    sonar_df["pitch"] = sonar_pitch
    sonar_df["roll"] = sonar_roll

    return sonar_df


# function to parse sections of interest from blueboat tlog csv
# tlogs converted to csv by data-alignment/telemetry/mavlogparse.py
# https://github.com/ES-Alexander/data-alignment/blob/main/telemetry/mavlogparse.py
# functionality should be copied to yellowfinLib.py or added to asv_sbes_processing
# as a submodule
def parse_imu_csv_to_df(imu_csv_path):
    cols_of_interest = ["timestamp", "ATTITUDE.roll", "ATTITUDE.pitch", "ATTITUDE.yaw"]
    imu_df = pd.read_csv(imu_csv_path, usecols=cols_of_interest)
    imu_df = imu_df.dropna()
    imu_df = imu_df.reset_index(drop=True)
    return imu_df


def main():
    sonar_h5_path = "/data/blueboat/20251120/20251120_sonarRaw.h5"
    sonar_h5 = h5py.File(sonar_h5_path, 'r')
    sonar_dict = {"timestamp": sonar_h5["time"], "Depth QAQC (m)": sonar_h5["qaqc_depth_m"]}
    sonar_df = pd.DataFrame.from_dict(sonar_dict)

    imu_csv_path = "/data/blueboat/20251120/tlogs/00442-2025-11-20_14-43-34.csv"
    imu_df = parse_imu_csv_to_df(imu_csv_path)

    emlid_fldr_list = ["/data/blueboat/20251120/emlidRaw/20251120_RINEX"]
    pos_df = read_emlid_pos(emlid_fldr_list, saveFname="/data/blueboat/20251120/20251120_ppkRaw.h5")
    pos_df['timestamp'] = pos_df['datetime'].astype(int)
    # divide the resulting integer by the number of nanoseconds in a second
    pos_df['timestamp'] = pos_df['timestamp'].div(10 ** 9)

    merged_df = merge_pos_sonar_imu_data(sonar_df, pos_df, imu_df)
    rectified_df = rectify_sonar_position(merged_df)
    rectified_df.to_csv("/data/blueboat/20251120/20251120_blueboat_survey.csv")


if __name__ == '__main__':
    main()