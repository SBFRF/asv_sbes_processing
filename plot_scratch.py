# import scipy.interpolate
import tqdm
from matplotlib import pyplot as plt
import netCDF4 as nc
import rasterio
import numpy as np
import yellowfinLib
import sys

sys.path.append("/home/slug/repos/getdatatestbed/src")
from getdatatestbed import getDataFRF

# sys.path.append('/home/slug/repos/testbedutils/src')
from testbedutils import sblib
import datetime as DT
import os
from copy import deepcopy
import glob


def asv_sounder_animation_fig_setup(image_fname, d, interactive=False):
    """initial setup of the figure, axes defined and the argus image plotted"""
    plt.ion() if interactive else plt.ioff()

    timex = rasterio.open(image_fname)

    fig = plt.figure(figsize=(12, 8))
    ax1 = plt.subplot2grid((4, 4), (0, 0), colspan=2, rowspan=4)
    aa = rasterio.plot.show(timex, ax=ax1)

    ax2 = plt.subplot2grid((4, 4), (0, 2), colspan=2, rowspan=1)

    ax3 = plt.subplot2grid((4, 4), (1, 2), colspan=2, rowspan=3)

    fig.axes[0].set_xlim([d["easting"].min() - 10, d["easting"].max() + 10])
    fig.axes[0].set_ylim([d["northing"].min() - 10, d["northing"].max() + 10])

    return fig


def asv_sounder_animation_core(
    fig,
    sp_e,
    sp_n,
    elev,
    c_min,
    c_max,
    bad_lat,
    bad_lon,
    sonar_time,
    bs,
    inst_depth,
    smooth_depth,
    gnss_loc,
    fspec,
    fspec_bin,
):
    ## in function #############
    current_point_good = True if np.isnan(bad_lon) & np.isnan(bad_lat) else False

    d_range = -np.linspace(0, 20, bs.shape[1])
    ######### make plot
    ax1, ax2, ax3 = fig.get_axes()

    cmp = ax1.scatter(sp_e, sp_n, c=elev, vmin=c_min, vmax=c_max)
    if current_point_good is True:
        ax1.plot(sp_e[-1], sp_n[-1], marker="D", markersize=10, color="green")
    else:
        ax1.plot(sp_e[-1], sp_n[-1], marker="X", color="r")
    cbar = plt.colorbar(cmp, ax=ax1)
    cbar.set_label("depth [m]")

    # this one plots wave spectra
    ax2.plot(fspec_bin, fspec[-1], "-k", label="now", zorder=10)
    ax2.plot(fspec_bin, fspec[1], color="grey", zorder=2)
    ax2.plot(fspec_bin, fspec[0], color="lightgrey", zorder=1)
    ax2.legend()
    ax2.set_xlim([fspec_bin.min(), fspec_bin.max()])

    # this one plots sounder
    ax3.plot(sonar_time, -inst_depth, "-k", label="inst_depth")
    ax3.plot(sonar_time, -smooth_depth, label="smoothed depth")
    ax3.pcolormesh(sonar_time, d_range, bs.T, alpha=0.5)
    ax3.set_xlim([sonar_time[0], sonar_time[-1]])
    ax3.set_ylim([min(-inst_depth.max(), -smooth_depth.max()) * 1.5, 0])
    ax3.legend(loc="lower left")
    plt.subplots_adjust(left=0.1, right=0.99, top=0.965, bottom=0.11, wspace=0.25, hspace=0.3)
    return fig


def asv_sounder_animation(figure_out_base, d, wave, image_fname, plot_every_x_data_points=20):
    interactive = False  # plotting
    backupfig = asv_sounder_animation_fig_setup(image_fname, d, interactive)
    dont_plot_before_x_data_points = 300
    sounder_tail = 300  # approx 75 s of data (@4hz)
    assert sounder_tail <= dont_plot_before_x_data_points, "the tail needs to be shorter"
    for ttt in tqdm.tqdm(
        np.arange(dont_plot_before_x_data_points, len(d["time"])).astype(int)[::plot_every_x_data_points],
        desc="ASV animation",
    ):
        fig = deepcopy(backupfig)  # start with fresh copy of the initial figure passed back

        fname_out = f"{figure_out_base}_{ttt:0>6}.png"
        sp_e, sp_n = d["easting"][:ttt], d["northing"][:ttt]
        elev = d["elevation"][:ttt]
        c_min, c_max = d["elevation"].min(), d["elevation"].max()
        bad_lat, bad_lon = d["bad_lat"][ttt], d["bad_lon"][ttt]
        sonar_time = d["time"][ttt - sounder_tail : ttt]
        bs = d["sonar_backscatter_out"][ttt - sounder_tail : ttt]
        inst_depth, smooth_depth = (
            d["sonar_instant_depth"][ttt - sounder_tail : ttt],
            d["sonar_smooth_depth"][ttt - sounder_tail : ttt],
        )
        gnss_loc = d["gnss_elevation_navd_m"][:ttt]
        if np.size(sonar_time) < dont_plot_before_x_data_points:
            continue

        # now find the wave spec (find the difference that is positive and then subtract one)
        thresh = 450  # seconds between sonar time and wave time (450s is 1/4 the wave obs interval time)
        idx_wave = np.argwhere(wave["epochtime"] - sonar_time[-1] > -thresh).squeeze().min() - 1
        num_spec = 3  # how many spec's to plot
        fspec = wave["fspec"][idx_wave - num_spec + 1 : idx_wave + 1]
        fspec_bin = wave["wavefreqbin"]

        sonar_time = [DT.datetime.utcfromtimestamp(x) for x in sonar_time]  # convert to datetime object for plotting
        fig = asv_sounder_animation_core(
            fig,
            sp_e,
            sp_n,
            elev,
            c_min,
            c_max,
            bad_lat,
            bad_lon,
            sonar_time,
            bs,
            inst_depth,
            smooth_depth,
            gnss_loc,
            fspec,
            fspec_bin,
        )
        # fig.tight_layout()
        fig.savefig(fname_out)
        plt.close(fig)

    # now make animation
    flist = glob.glob("figures/*.png")
    sblib.makeMovie(flist, ofname="movie.mp4", fps=1)


fname_h5 = "/data/yellowfin/20240626/20240626_totalCombinedRawData.h5"
d = yellowfinLib.load_h5_to_dictionary(fname_h5)
coord = yellowfinLib.geoprocess.FRFcoord(d["longitude"], d["latitude"])
d["easting"], d["northing"] = coord["StateplaneE"], coord["StateplaneN"]
image_fname = "/data/yellowfin/20240626/figures/Argus_20240626.tif"

start = nc.num2date(
    d["time"][0], "seconds since 1970-01-01", only_use_python_datetimes=True, only_use_cftime_datetimes=False
)
start = start.replace(minute=0, second=0, hour=0, microsecond=0)
end = nc.num2date(
    d["time"][-1], "seconds since 1970-01-01", only_use_python_datetimes=True, only_use_cftime_datetimes=False
)
end = end.replace(hour=end.hour + 1, minute=0, second=0, microsecond=0)
go = getDataFRF.getObs(start, end)
wave = go.getWaveData("17m", spec=True)
# # interp to 2D surface
# f = interpolate.RegularGridInterpolator((d['easting'], d['northing']), d['elevation'])

# d[('Profile_number')]
#
# if profileNumber is True:
#     ax_loc

path_base = "figures"
figure_out_base = os.path.join(path_base, "asv_animation")
asv_sounder_animation(figure_out_base, d, wave, image_fname)

print('why do i start out with "good" data??? ')
