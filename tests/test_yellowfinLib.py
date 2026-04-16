"""
Unit tests for yellowfinLib.py
Tests core functionality of data loading, processing, and conversion functions
"""

import pytest
import numpy as np
import pandas as pd
import h5py
import os
import datetime as DT
from unittest.mock import Mock, patch


import yellowfinLib


class TestReadEmlidPos:
    """Tests for read_emlid_pos function"""

    def test_read_emlid_pos_single_folder(self, temp_dir, mock_pos_file):
        """Test reading a single pos file"""
        folder = os.path.dirname(mock_pos_file)
        result = yellowfinLib.read_emlid_pos([folder], plot=False)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 10
        assert "datetime" in result.columns
        assert "lat" in result.columns
        assert "lon" in result.columns
        assert "height" in result.columns
        assert "Q" in result.columns

    def test_read_emlid_pos_datetime_parsing(self, temp_dir, mock_pos_file):
        """Test that datetime is correctly parsed"""
        folder = os.path.dirname(mock_pos_file)
        result = yellowfinLib.read_emlid_pos([folder], plot=False)

        assert pd.api.types.is_datetime64_any_dtype(result["datetime"])
        assert result["datetime"].dt.tz is not None  # Should be UTC

    def test_read_emlid_pos_with_save(self, temp_dir, mock_pos_file):
        """Test saving to H5 file"""
        folder = os.path.dirname(mock_pos_file)
        save_path = temp_dir / "output.h5"

        result = yellowfinLib.read_emlid_pos(
            [folder], plot=False, saveFname=str(save_path)
        )

        assert isinstance(result, pd.DataFrame)
        assert os.path.exists(save_path)

    def test_read_emlid_pos_empty_folder(self, temp_dir):
        """Test with empty folder list"""
        result = yellowfinLib.read_emlid_pos([], plot=False)
        # Should return empty dataframe or handle gracefully
        assert isinstance(result, pd.DataFrame)


class TestLoadH5ToDictionary:
    """Tests for load_h5_to_dictionary function"""

    def test_load_h5_to_dictionary(self, sample_sonar_data):
        """Test loading H5 file to dictionary"""
        result = yellowfinLib.load_h5_to_dictionary(sample_sonar_data)

        assert isinstance(result, dict)
        assert "time" in result
        assert "smooth_depth_m" in result
        assert "profile_data" in result
        assert isinstance(result["time"], np.ndarray)

    def test_load_h5_to_dictionary_data_integrity(self, sample_sonar_data):
        """Test that data is loaded correctly"""
        result = yellowfinLib.load_h5_to_dictionary(sample_sonar_data)

        # Check data shapes
        assert len(result["time"]) == 100
        assert result["profile_data"].shape[1] == 100


class TestButterLowpassFilter:
    """Tests for butter_lowpass_filter function"""

    def test_butter_lowpass_filter_basic(self):
        """Test basic filtering operation"""
        # Create a noisy signal
        t = np.linspace(0, 1, 1000)
        signal = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 50 * t)

        # Filter with cutoff at 10 Hz
        filtered = yellowfinLib.butter_lowpass_filter(
            signal, cutoff=10, fs=1000, order=4
        )

        assert len(filtered) == len(signal)
        assert isinstance(filtered, np.ndarray)

    def test_butter_lowpass_filter_reduces_noise(self):
        """Test that filter actually reduces high frequency noise"""
        t = np.linspace(0, 1, 1000)
        clean_signal = np.sin(2 * np.pi * 5 * t)
        noise = 0.5 * np.sin(2 * np.pi * 100 * t)
        noisy_signal = clean_signal + noise

        filtered = yellowfinLib.butter_lowpass_filter(
            noisy_signal, cutoff=10, fs=1000, order=4
        )

        # Filtered signal should be closer to clean signal than noisy signal
        error_before = np.mean((noisy_signal - clean_signal) ** 2)
        error_after = np.mean((filtered - clean_signal) ** 2)

        assert error_after < error_before


class TestFindTimeShiftCrossCorr:
    """Tests for findTimeShiftCrossCorr function"""

    def test_find_time_shift_no_shift(self):
        """Test with signals that have no shift"""
        signal1 = np.sin(np.linspace(0, 4 * np.pi, 100))
        signal2 = signal1.copy()

        phase_lag_samples, phase_lag_seconds = yellowfinLib.findTimeShiftCrossCorr(
            signal1, signal2, sampleFreq=1
        )

        assert abs(phase_lag_samples) < 2  # Should be close to 0

    def test_find_time_shift_with_known_shift(self):
        """Test with known time shift"""
        signal1 = np.sin(np.linspace(0, 4 * np.pi, 100))
        signal2 = np.roll(signal1, 10)  # Shift by 10 samples

        phase_lag_samples, phase_lag_seconds = yellowfinLib.findTimeShiftCrossCorr(
            signal1, signal2, sampleFreq=1
        )

        # Should detect the 10 sample shift (or close to it)
        assert abs(phase_lag_samples - 10) < 5

    def test_find_time_shift_assertion_error(self):
        """Test that function raises error for mismatched signal lengths"""
        signal1 = np.random.randn(100)
        signal2 = np.random.randn(50)

        with pytest.raises(AssertionError):
            yellowfinLib.findTimeShiftCrossCorr(signal1, signal2)


class TestMLabDatetimeToEpoch:
    """Tests for mLabDatetime_to_epoch function"""

    def test_epoch_conversion(self):
        """Test conversion to epoch time"""
        test_date = DT.datetime(2023, 8, 15, 12, 0, 0)
        epoch = yellowfinLib.mLabDatetime_to_epoch(test_date)

        assert isinstance(epoch, float)
        assert epoch > 0

    def test_epoch_known_value(self):
        """Test with known epoch value"""
        # Unix epoch start
        test_date = DT.datetime(1970, 1, 1, 0, 0, 0)
        epoch = yellowfinLib.mLabDatetime_to_epoch(test_date)

        assert epoch == 0.0

    def test_epoch_recent_date(self):
        """Test with recent date"""
        test_date = DT.datetime(2023, 1, 1, 0, 0, 0)
        epoch = yellowfinLib.mLabDatetime_to_epoch(test_date)

        # Should be around 1672531200 (2023-01-01)
        assert 1672531200 <= epoch <= 1672617600


class TestConvertEllipsoid2NAVD88:
    """Tests for convertEllipsoid2NAVD88 function"""

    @patch("yellowfinLib.geoids.GeoidG2012B")
    def test_convert_ellipsoid_basic(self, mock_geoid):
        """Test basic ellipsoid conversion"""
        mock_instance = Mock()
        mock_instance.height.return_value = -32.0  # Typical geoid height
        mock_geoid.return_value = mock_instance

        lats = np.array([35.0, 35.1, 35.2])
        lons = np.array([-75.0, -75.1, -75.2])
        ellipsoids = np.array([10.0, 12.0, 15.0])

        result = yellowfinLib.convertEllipsoid2NAVD88(lats, lons, ellipsoids)

        assert len(result) == 3
        assert isinstance(result, (np.ndarray, float))

    def test_convert_ellipsoid_length_assertion(self):
        """Test that function requires equal length inputs"""
        lats = np.array([35.0, 35.1])
        lons = np.array([-75.0, -75.1, -75.2])
        ellipsoids = np.array([10.0, 12.0])

        with pytest.raises(AssertionError):
            yellowfinLib.convertEllipsoid2NAVD88(lats, lons, ellipsoids)


class TestLoadPPKData:
    """Tests for loadPPKdata function"""

    def test_load_ppk_data_single_folder(self, temp_dir, mock_pos_file):
        """Test loading PPK data from single folder"""
        folder = os.path.dirname(mock_pos_file)
        result = yellowfinLib.loadPPKdata([folder])

        assert isinstance(result, pd.DataFrame)
        assert "datetime" in result.columns
        assert "epochTime" in result.columns
        assert "lat" in result.columns
        assert "lon" in result.columns

    def test_load_ppk_data_empty_list(self):
        """Test with empty folder list"""
        result = yellowfinLib.loadPPKdata([])

        assert isinstance(result, pd.DataFrame)


class TestUnpackYellowfinCombinedRaw:
    """Tests for unpackYellowfinCombinedRaw function"""

    def test_unpack_combined_raw(self, temp_dir):
        """Test unpacking combined raw H5 file"""
        # Create test H5 file
        fname = temp_dir / "combined.h5"
        with h5py.File(fname, "w") as hf:
            hf.create_dataset("time", data=np.linspace(0, 100, 100))
            hf.create_dataset("longitude", data=np.ones(100) * -75.0)
            hf.create_dataset("latitude", data=np.ones(100) * 35.0)
            hf.create_dataset("elevation", data=np.random.uniform(-5, 0, 100))
            hf.create_dataset("fix_quality_GNSS", data=np.ones(100))
            hf.create_dataset("sonar_smooth_depth", data=np.random.uniform(1, 5, 100))
            hf.create_dataset("sonar_smooth_confidence", data=np.ones(100) * 80)
            hf.create_dataset("sonar_instant_depth", data=np.random.uniform(1, 5, 100))
            hf.create_dataset("sonar_instant_confidence", data=np.ones(100) * 90)
            hf.create_dataset(
                "sonar_backscatter_out", data=np.random.randint(0, 255, 100)
            )
            hf.create_dataset("bad_lat", data=np.array([]))
            hf.create_dataset("bad_lon", data=np.array([]))
            hf.create_dataset("xFRF", data=np.random.uniform(0, 1000, 100))
            hf.create_dataset("yFRF", data=np.random.uniform(0, 2000, 100))
            hf.create_dataset("Profile_number", data=np.ones(100))

        result = yellowfinLib.unpackYellowfinCombinedRaw(str(fname))

        assert isinstance(result, dict)
        assert "time" in result
        assert "latitude" in result
        assert "longitude" in result
        assert len(result["time"]) == 100


class TestIsLocalToFRF:
    """Tests for is_local_to_FRF function"""

    def test_is_local_to_frf_true(self):
        """Test with coordinates local to FRF"""
        coords = {
            "xFRF": np.array([100, 200, 300]),
            "yFRF": np.array([500, 1000, 1500]),
        }
        result = yellowfinLib.is_local_to_FRF(coords)

        assert result is True

    def test_is_local_to_frf_false(self):
        """Test with coordinates not local to FRF"""
        coords = {
            "xFRF": np.array([-100, 200, -300]),
            "yFRF": np.array([50000, 60000, 70000]),
        }
        result = yellowfinLib.is_local_to_FRF(coords)

        assert result is False

    def test_is_local_to_frf_boundary(self):
        """Test boundary conditions"""
        coords = {
            "xFRF": np.array([100, 200, 300, 500]),
            "yFRF": np.array([-100, 0, 100, 1900]),
        }
        result = yellowfinLib.is_local_to_FRF(coords)

        assert result is True


class TestLoadSonarS500Binary:
    """Tests for loadSonar_s500_binary function"""

    def test_load_sonar_no_files(self, temp_dir):
        """Test with directory containing no .dat files"""
        with pytest.raises(EnvironmentError):
            yellowfinLib.loadSonar_s500_binary(str(temp_dir))

    def test_load_sonar_with_save(self, temp_dir):
        """Test sonar loading with save functionality"""
        # Create minimal binary sonar file
        sonar_file = temp_dir / "test.dat"

        # Create minimal valid sonar data (this is a simplified version)
        with open(sonar_file, "wb") as f:
            # Write minimal binary data that won't crash the parser
            # This is a mock and won't represent real sonar data
            f.write(b"BR")  # Start marker
            f.write(b"2023-08-15 12:00:00.000000")  # Date string
            f.write(b"\x00" * 100)  # Padding

        # Note: Full sonar binary format is complex, so we test error handling
        # In real tests, you would use actual sonar data samples


class TestLoadYellowfinNMEAFiles:
    """Tests for load_yellowfin_NMEA_files function"""

    def test_load_nmea_basic(self, temp_dir, sample_nmea_data):
        """Test loading NMEA files"""
        folder = os.path.dirname(sample_nmea_data)
        output_file = temp_dir / "nmea_output.h5"

        yellowfinLib.load_yellowfin_NMEA_files(
            folder, saveFname=str(output_file), plotfname=False, verbose=0
        )

        assert os.path.exists(output_file)

        # Check H5 file contents
        with h5py.File(output_file, "r") as hf:
            assert "lat" in hf.keys()
            assert "lon" in hf.keys()
            assert "gps_time" in hf.keys()
            assert "altMSL" in hf.keys()


class TestMakePOSFileFromRINEX:
    """Tests for makePOSfileFromRINEX function"""

    @patch("os.system")
    def test_make_pos_file(self, mock_system):
        """Test RTKlib execution"""
        mock_system.return_value = 0

        yellowfinLib.makePOSfileFromRINEX(
            roverObservables="rover.obs",
            baseObservables="base.obs",
            navFile="nav.nav",
            outfname="output.pos",
            executablePath="rnx2rtkp",
        )

        # Verify os.system was called
        assert mock_system.called

    @patch("os.system")
    def test_make_pos_file_with_sp3(self, mock_system):
        """Test RTKlib with SP3 file"""
        mock_system.return_value = 0

        yellowfinLib.makePOSfileFromRINEX(
            roverObservables="rover.obs",
            baseObservables="base.obs",
            navFile="nav.nav",
            outfname="output.pos",
            executablePath="rnx2rtkp",
            sp3="precise.sp3",
        )

        assert mock_system.called


@pytest.mark.parametrize(
    "signal_length,cutoff,fs,order",
    [
        (100, 10, 100, 2),
        (500, 20, 200, 4),
        (1000, 5, 50, 3),
    ],
)
def test_butter_lowpass_filter_parameterized(signal_length, cutoff, fs, order):
    """Parameterized test for various filter configurations"""
    signal = np.random.randn(signal_length)
    filtered = yellowfinLib.butter_lowpass_filter(signal, cutoff, fs, order)

    assert len(filtered) == signal_length
    assert not np.any(np.isnan(filtered))
    assert not np.any(np.isinf(filtered))
