import matplotlib

matplotlib.use("QtAgg")
import datetime as DT
import os
from matplotlib import pyplot as plt
from testbedutils import geoprocess
import yellowfinLib
import numpy as np
from scipy import signal

###########
dateOfInterest = DT.datetime(2023, 11, 9, 13, 0, 0)  # "20231109T120000Z"
# start getting imagery early so we can do the rest of the flow in parallel
# argusName = yellowfinLib.threadGetArgusImagery(dateOfInterest)
# yellowfinLib.plotPlanViewOnArgus(data, argusName, ofName='')

yellowFinDatafname = "/data/yellowfin/20231109/20231109_totalCombinedRawData.h5"
data = yellowfinLib.unpackYellowfinCombinedRaw(yellowFinDatafname)
# convert to all coords
coords = geoprocess.FRFcoord(p1=data["longitude"], p2=data["latitude"], coordType="LL")
pierStart = geoprocess.FRFcoord(0, 515, coordType="FRF")
pierEnd = geoprocess.FRFcoord(534, 515, coordType="FRF")
lineNumbers = sorted(np.unique(data["Profile_number"]))[1:]
## isolate and focus on one FRF line profile
order = 2

cutoff = 1 / 10  # m
fig, axs = plt.subplots(ncols=1, nrows=len(lineNumbers), figsize=(15, 8))
for i, lineNumber in enumerate(lineNumbers):
    logic = data["Profile_number"] == lineNumbers[i]
    axs[i].plot(coords["xFRF"][logic], data["elevation"][logic], label="raw")
    axs[i].set_title(f"lineNumber {lineNumbers[i]:.1f}")
    axs[i].set_ylabel("elevation[m]")
    for cutoff in [1 / 20, 1 / 50]:
        for order in [2, 5, 10]:
            b, a = signal.butter(order, cutoff, "low", analog=False)
            output = signal.filtfilt(b, a, data["elevation"][logic])
            axs[i].plot(coords["xFRF"][logic], output, label=f"filtered c={cutoff} o={order}")

axs[i].legend()
plt.tight_layout()


# passedBathy = yellowfinLib.butter_lowpass_filter(data['elevation'], 5, fs=0.1, order=2)
#
# plt.figure(figsize=(12, 8));
# plt.subplot(211)
# plt.title('plan view of survey')
# plt.scatter(coords['xFRF'], coords['yFRF'], c=elevation_out[idxDataToSave],
#             vmax=-1)  # time_out[idxDataToSave])  #
# cbar = plt.colorbar()
# cbar.set_label('depth')
# plt.subplot(212)
# plt.title(f"profile at line y={np.median(coords['yFRF'][logic]).astype(int)}")
# plt.plot(coords['xFRF'][logic],
#          gnss_out[idxDataToSave][logic] - antenna_offset - sonar_instant_depth_out[idxDataToSave][logic],
#          label='instant depths')
# plt.plot(coords['xFRF'][logic],
#          gnss_out[idxDataToSave][logic] - antenna_offset - sonar_smooth_depth_out[idxDataToSave][logic],
#          label='smooth Depth')
# plt.plot(coords['xFRF'][logic], elevation_out[idxDataToSave][logic], label='chosen depths')
# plt.legend()
# plt.xlabel('xFRF')
# plt.ylabel('elevation NAVD88[m]')
# plt.tight_layout()
# plt.savefig(os.path.join(plotDir, 'singleProfile.png'))
