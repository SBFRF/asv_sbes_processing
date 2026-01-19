"""
Shared test fixtures and configuration for pytest
"""
import pytest
import numpy as np
import pandas as pd
import h5py
import tempfile
import os
from datetime import datetime, timezone
import yaml
import sys
from unittest.mock import MagicMock

# Mock unavailable external dependencies
sys.modules['testbedutils'] = MagicMock()
sys.modules['testbedutils.geoprocess'] = MagicMock()
sys.modules['pygeodesy'] = MagicMock()
sys.modules['pygeodesy.geoids'] = MagicMock()
sys.modules['wget'] = MagicMock()


@pytest.fixture
def temp_dir(tmp_path):
    """Create a temporary directory for test files"""
    return tmp_path


@pytest.fixture
def sample_pos_data():
    """Sample PPK position data"""
    return pd.DataFrame({
        'date': ['2023/08/15'] * 10,
        'time': ['12:00:00.000', '12:00:01.000', '12:00:02.000', '12:00:03.000', '12:00:04.000',
                 '12:00:05.000', '12:00:06.000', '12:00:07.000', '12:00:08.000', '12:00:09.000'],
        'lat': [35.0 + i*0.0001 for i in range(10)],
        'lon': [-75.0 + i*0.0001 for i in range(10)],
        'height': [10.0 + i*0.1 for i in range(10)],
        'Q': [1] * 10,  # Fixed quality
        'ns': [15] * 10,
        'sdn(m)': [0.01] * 10,
        'sde(m)': [0.01] * 10,
        'sdu(m)': [0.02] * 10,
        'sdne(m)': [0.005] * 10,
        'sdeu(m)': [0.005] * 10,
        'sdun(m)': [0.005] * 10,
        'age(s)': [0.5] * 10,
        'ratio': [5.0] * 10
    })


@pytest.fixture
def sample_sonar_data(temp_dir):
    """Create sample sonar H5 file"""
    fname = temp_dir / "test_sonar.h5"
    n_pings = 100
    n_bins = 50

    with h5py.File(fname, 'w') as hf:
        hf.create_dataset('time', data=np.linspace(1692100000, 1692100100, n_pings))
        hf.create_dataset('smooth_depth_m', data=np.random.uniform(1, 5, n_pings))
        hf.create_dataset('this_ping_depth_m', data=np.random.uniform(1, 5, n_pings))
        hf.create_dataset('profile_data', data=np.random.randint(0, 255, (n_bins, n_pings)))
        hf.create_dataset('range_m', data=np.linspace(0, 10, n_bins))
        hf.create_dataset('min_pwr', data=np.random.uniform(0, 100, n_pings))
        hf.create_dataset('max_pwr', data=np.random.uniform(100, 200, n_pings))
        hf.create_dataset('analog_gain', data=np.ones(n_pings) * 1.5)
        hf.create_dataset('num_results', data=n_bins)
        hf.create_dataset('start_mm', data=np.zeros(n_pings))
        hf.create_dataset('length_mm', data=np.ones(n_pings) * 10000)
        hf.create_dataset('ping_duration', data=np.ones(n_pings) * 0.01)
        hf.create_dataset('timestamp_msec', data=np.linspace(0, 100000, n_pings))

    return str(fname)


@pytest.fixture
def sample_nmea_data(temp_dir):
    """Create sample NMEA data file"""
    fname = temp_dir / "test_nmea.dat"

    with open(fname, 'w') as f:
        for i in range(10):
            timestamp = f"2023-08-15 12:00:{i:02d}.000"
            lat_deg = 35
            lat_min = i * 0.01
            lon_deg = -75
            lon_min = i * 0.01

            nmea_string = (f"#{timestamp}$GNGGA,120000.00,{lat_deg}{lat_min:06.3f},N,"
                          f"{abs(lon_deg)}{lon_min:06.3f},W,1,12,0.8,5.0,M,-32.0,M,1.2,0000*4E\n")
            f.write(nmea_string)

    return str(fname)


@pytest.fixture
def sample_h5_dict():
    """Sample H5 data as dictionary"""
    return {
        'time': np.linspace(1692100000, 1692100100, 100),
        'latitude': np.ones(100) * 35.0,
        'longitude': np.ones(100) * -75.0,
        'elevation': np.random.uniform(-5, 0, 100),
        'fix_quality_GNSS': np.ones(100) * 1,
        'sonar_smooth_depth': np.random.uniform(1, 5, 100),
    }


@pytest.fixture
def sample_yaml_global(temp_dir):
    """Create sample global YAML for netCDF"""
    fname = temp_dir / "global.yml"
    data = {
        'title': 'Test Survey',
        'institution': 'USACE',
        'source': 'ASV Yellowfin',
        'conventions': 'CF-1.8',
        'summary': 'Test bathymetric survey'
    }
    with open(fname, 'w') as f:
        yaml.dump(data, f)
    return str(fname)


@pytest.fixture
def sample_yaml_variables(temp_dir):
    """Create sample variables YAML for netCDF"""
    fname = temp_dir / "variables.yml"
    data = {
        '_variables': ['time', 'latitude', 'longitude', 'elevation'],
        '_dimensions': ['time'],
        'time': {
            'name': 'time',
            'data_type': 'f8',
            'dim': ['time'],
            'units': 'seconds since 1970-01-01 00:00:00',
            'standard_name': 'time',
            'long_name': 'time'
        },
        'latitude': {
            'name': 'latitude',
            'data_type': 'f8',
            'dim': ['time'],
            'units': 'degrees_north',
            'standard_name': 'latitude',
            'long_name': 'latitude'
        },
        'longitude': {
            'name': 'longitude',
            'data_type': 'f8',
            'dim': ['time'],
            'units': 'degrees_east',
            'standard_name': 'longitude',
            'long_name': 'longitude'
        },
        'elevation': {
            'name': 'elevation',
            'data_type': 'f8',
            'dim': ['time'],
            'units': 'm',
            'standard_name': 'height_above_reference_ellipsoid',
            'long_name': 'elevation NAVD88'
        }
    }
    with open(fname, 'w') as f:
        yaml.dump(data, f)
    return str(fname)


@pytest.fixture
def mock_pos_file(temp_dir):
    """Create a mock .pos file for testing"""
    fname = temp_dir / "test.pos"

    # Create header (12 lines)
    header_lines = [
        "% program   : RTKLIB ver.demo5 b34e\n",
        "% inp file  : test.obs\n",
        "% obs start : 2023/08/15 12:00:00.0 GPST\n",
        "% obs end   : 2023/08/15 13:00:00.0 GPST\n",
        "% # of epochs: 3600\n",
        "% # of sat   : 15\n",
        "% processing: post\n",
        "% baseline  : 1234.567 m\n",
        "% parameters:\n",
        "% pos mode  : kinematic\n",
        "% frequencies: L1+L2\n",
        "% # of epochs: 10\n"
    ]

    with open(fname, 'w') as f:
        f.writelines(header_lines)

        # Add data rows with fixed width format
        for i in range(10):
            date_str = "2023/08/15"
            time_str = f"12:00:{i:02d}.000"
            lat = 35.0 + i * 0.0001
            lon = -75.0 + i * 0.0001
            height = 10.0 + i * 0.1

            # Fixed width format matching RTKlib output
            line = f"{date_str} {time_str}   {lat:13.9f}  {lon:14.9f}  {height:9.4f}  1  15   0.0100   0.0100   0.0200   0.0050   0.0050   0.0050    0.5  5.00\n"
            f.write(line)

    return str(fname)


@pytest.fixture
def mock_netcdf_data():
    """Sample data dictionary for netCDF creation"""
    n_points = 100
    return {
        'time': np.linspace(1692100000, 1692110000, n_points),
        'latitude': np.random.uniform(35.0, 35.1, n_points),
        'longitude': np.random.uniform(-75.1, -75.0, n_points),
        'elevation': np.random.uniform(-5, 0, n_points),
    }
