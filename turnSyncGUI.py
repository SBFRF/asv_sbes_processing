import matplotlib
try:
    matplotlib.use("TkAgg")  # Required for interactive plotting features
except (ImportError, ModuleNotFoundError):
    matplotlib.use("Agg")  # Fallback for headless CI/testing environments
import matplotlib.pyplot as plt

import numpy as np

class TurnSyncGUI:
    def __init__(self, T_ppk, sonarData):
        self.T_ppk = T_ppk
        self.sonarData = sonarData

        self.gps_turn_times = []
        self.sonar_turn_times = []

        self.fig, (self.ax_map, self.ax_sonar) = plt.subplots(2, 1, figsize=(10, 8))

        self._plot_data()

        self.cid_map = self.fig.canvas.mpl_connect("button_press_event", self.on_click)

        plt.title("Click GPS turns (map) or sonar events (bottom)")
        plt.show()

    def _plot_data(self):
        # --- GPS track ---
        self.ax_map.plot(self.T_ppk["lon"], self.T_ppk["lat"], "k-", linewidth=1)
        self.ax_map.set_title("GPS Track (Click turns here)")
        self.ax_map.set_xlabel("Lon")
        self.ax_map.set_ylabel("Lat")

        # --- Sonar ---
        self.ax_sonar.plot(self.sonarData["time"], self.sonarData["this_ping_depth_m"], "b-")
        self.ax_sonar.set_title("Sonar Depth (Optional turn alignment)")
        self.ax_sonar.set_xlabel("Time")
        self.ax_sonar.set_ylabel("Depth")

        self.fig.tight_layout()

    def on_click(self, event):
        if event.inaxes == self.ax_map:
            # GPS turn selection
            t = self._nearest_time_from_click(event.xdata, event.ydata)
            self.gps_turn_times.append(t)
            self.ax_map.plot(event.xdata, event.ydata, "ro")
            self.fig.canvas.draw()

            print(f"[GPS turn] {t}")

        elif event.inaxes == self.ax_sonar:
            # Sonar event selection
            self.sonar_turn_times.append(event.xdata)
            self.ax_sonar.axvline(event.xdata, color="r")
            self.fig.canvas.draw()

            print(f"[SONAR event] {event.xdata}")

    def _nearest_time_from_click(self, x, y):
        # find nearest GPS sample time
        d = (self.T_ppk["lon"] - x)**2 + (self.T_ppk["lat"] - y)**2
        idx = np.argmin(d)
        return self.T_ppk["epochTime"].iloc[idx]

    def compute_offset(self):
        gps = np.array(self.gps_turn_times)
        sonar = np.array(self.sonar_turn_times)

        n = min(len(gps), len(sonar))

        if n == 0:
            raise ValueError("No turn pairs selected")

        offsets = gps[:n] - sonar[:n]

        return {
            "mean_offset": float(np.mean(offsets)),
            "std_offset": float(np.std(offsets)),
            "offsets": offsets
        }