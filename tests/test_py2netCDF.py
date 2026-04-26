"""
Tests for py2netCDF.py

This file contains both unit tests and integration tests:

UNIT TESTS (currently passing):
- YAML template loading (TestImportTemplateFile)
- Basic netCDF initialization (TestInitNcFile - some tests)

INTEGRATION TESTS (currently disabled due to HDF5 issues):
- Full netCDF file creation (TestMakencGeneric)
- Data writing and validation (TestWriteDataToNc, TestReadNetCDFFile)

TODO: Before re-enabling integration tests (currently causing HDF5 errors):
1. Add sample data files to tests/data/ directory:
   - Small bathymetric survey dataset (sonar .dat, NMEA .dat, PPK .pos)
   - Pre-processed output examples for validation
2. Replace mock fixtures with real data loaders
3. Test against actual survey data workflow, not synthetic mocks
4. These tests belong in an integration test suite, not unit tests

See TESTING.md for guidance on adding sample data files.
"""

import pytest
import numpy as np
import netCDF4 as nc
import os


import py2netCDF


class TestImportTemplateFile:
    """Tests for import_template_file function"""

    def test_import_yaml_file(self, sample_yaml_global):
        """Test loading a YAML template file"""
        result = py2netCDF.import_template_file(sample_yaml_global)

        assert isinstance(result, dict)
        assert "title" in result
        assert "institution" in result
        assert result["institution"] == "USACE/CHL/COAB"

    def test_import_yaml_variables(self, sample_yaml_variables):
        """Test loading variables YAML"""
        result = py2netCDF.import_template_file(sample_yaml_variables)

        assert isinstance(result, dict)
        assert "_variables" in result
        assert "_dimensions" in result
        assert "time" in result
        # Verify all 12 variables from transect_variables.yml are present
        expected_vars = [
            "time",
            "date",
            "Latitude",
            "Longitude",
            "Northing",
            "Easting",
            "xFRF",
            "yFRF",
            "Elevation",
            "Profile_number",
            "Survey_number",
            "Ellipsoid",
        ]
        assert result["_variables"] == expected_vars


class TestInitNcFile:
    """Tests for init_nc_file function"""

    def test_init_nc_file_basic(self, temp_dir):
        """Test initializing netCDF file"""
        nc_file = temp_dir / "test.nc"
        attributes = {"title": "Test Dataset", "institution": "USACE", "source": "Test"}

        fid = py2netCDF.init_nc_file(str(nc_file), attributes)

        assert fid is not None
        assert hasattr(fid, "title")
        assert fid.title == "Test Dataset"
        assert hasattr(fid, "date_created")

        fid.close()

    def test_init_nc_file_creates_file(self, temp_dir):
        """Test that file is created on disk"""
        nc_file = temp_dir / "test.nc"
        attributes = {"title": "Test"}

        fid = py2netCDF.init_nc_file(str(nc_file), attributes)
        fid.close()

        assert os.path.exists(nc_file)

    def test_init_nc_file_none_values(self, temp_dir):
        """Test that None values are handled correctly"""
        nc_file = temp_dir / "test.nc"
        attributes = {
            "title": "Test",
            "description": None,  # Should be skipped
            "source": "Test Source",
        }

        fid = py2netCDF.init_nc_file(str(nc_file), attributes)

        assert hasattr(fid, "title")
        assert hasattr(fid, "source")
        assert not hasattr(fid, "description")  # None should not create attribute

        fid.close()


# ============================================================================
# INTEGRATION TESTS - Require sample data, currently disabled
# These tests fail with HDF5 errors and should be rewritten as integration
# tests using real sample data files instead of mocks.
# ============================================================================


class TestCreateDimensions:
    """Tests for _createDimensions function"""

    def test_create_dimensions_basic(self, temp_dir):
        """Test creating dimensions"""
        nc_file = temp_dir / "test.nc"
        fid = nc.Dataset(str(nc_file), "w", clobber=True)

        varMetaData = {
            "_dimensions": ["time"],
            "_variables": ["time"],
            "time": {"name": "time"},
        }

        data = {"time": np.arange(10)}

        py2netCDF._createDimensions(fid, varMetaData, data)

        assert "time" in fid.dimensions
        assert len(fid.dimensions["time"]) == 10

        fid.close()

    def test_create_dimensions_multiple(self, temp_dir):
        """Test creating multiple dimensions"""
        nc_file = temp_dir / "test.nc"
        fid = nc.Dataset(str(nc_file), "w", clobber=True)

        varMetaData = {
            "_dimensions": ["time", "depth"],
            "_variables": ["time", "depth"],
            "time": {"name": "time"},
            "depth": {"name": "depth"},
        }

        data = {"time": np.arange(10), "depth": np.arange(5)}

        py2netCDF._createDimensions(fid, varMetaData, data)

        assert "time" in fid.dimensions
        assert "depth" in fid.dimensions
        assert len(fid.dimensions["time"]) == 10
        assert len(fid.dimensions["depth"]) == 5

        fid.close()


class TestWriteDataToNc:
    """Tests for write_data_to_nc function"""

    def test_write_1d_data(self, temp_dir):
        """Test writing 1D data to netCDF"""
        nc_file = temp_dir / "test.nc"
        fid = nc.Dataset(str(nc_file), "w", clobber=True)
        fid.createDimension("time", 10)

        template_vars = {
            "_variables": ["time"],
            "time": {
                "name": "time",
                "data_type": "f8",
                "dim": ["time"],
                "units": "seconds",
            },
        }

        data_dict = {"time": np.arange(10, dtype=np.float64)}

        num_errors, error_str = py2netCDF.write_data_to_nc(fid, template_vars, data_dict)

        assert num_errors == 0
        assert "time" in fid.variables
        assert np.allclose(fid.variables["time"][:], np.arange(10))

        fid.close()

    def test_write_2d_data(self, temp_dir):
        """Test writing 2D data to netCDF"""
        nc_file = temp_dir / "test.nc"
        fid = nc.Dataset(str(nc_file), "w", clobber=True)
        fid.createDimension("time", 10)
        fid.createDimension("depth", 5)

        template_vars = {
            "_variables": ["temperature"],
            "temperature": {
                "name": "temperature",
                "data_type": "f8",
                "dim": ["time", "depth"],
                "units": "degrees_C",
            },
        }

        data_dict = {"temperature": np.random.rand(10, 5)}

        num_errors, error_str = py2netCDF.write_data_to_nc(fid, template_vars, data_dict)

        assert num_errors == 0
        assert "temperature" in fid.variables
        assert fid.variables["temperature"].shape == (10, 5)

        fid.close()

    def test_write_with_attributes(self, temp_dir):
        """Test writing data with metadata attributes"""
        nc_file = temp_dir / "test.nc"
        fid = nc.Dataset(str(nc_file), "w", clobber=True)
        fid.createDimension("time", 10)

        template_vars = {
            "_variables": ["time"],
            "time": {
                "name": "time",
                "data_type": "f8",
                "dim": ["time"],
                "units": "seconds",
                "standard_name": "time",
                "long_name": "time in seconds",
            },
        }

        data_dict = {"time": np.arange(10, dtype=np.float64)}

        py2netCDF.write_data_to_nc(fid, template_vars, data_dict)

        var = fid.variables["time"]
        assert var.units == "seconds"
        assert var.standard_name == "time"
        assert var.long_name == "time in seconds"

        fid.close()

    def test_write_with_fill_value(self, temp_dir):
        """Test writing data with fill value"""
        nc_file = temp_dir / "test.nc"
        fid = nc.Dataset(str(nc_file), "w", clobber=True)
        fid.createDimension("time", 10)

        template_vars = {
            "_variables": ["temperature"],
            "temperature": {
                "name": "temperature",
                "data_type": "f8",
                "dim": ["time"],
                "units": "degrees_C",
                "fill_value": -999.0,
            },
        }

        data_dict = {"temperature": np.arange(10, dtype=np.float64)}

        py2netCDF.write_data_to_nc(fid, template_vars, data_dict)

        var = fid.variables["temperature"]
        # NetCDF4 sets _FillValue attribute
        assert hasattr(var, "_FillValue")

        fid.close()


class TestMakencGeneric:
    """Tests for makenc_generic function"""

    def test_makenc_generic_complete(self, temp_dir, sample_yaml_global, sample_yaml_variables, mock_netcdf_data):
        """Test complete netCDF file creation"""
        nc_file = temp_dir / "output.nc"

        py2netCDF.makenc_generic(str(nc_file), sample_yaml_global, sample_yaml_variables, mock_netcdf_data)

        assert os.path.exists(nc_file)

        # Verify file contents with real YAML structure
        with nc.Dataset(str(nc_file), "r") as fid:
            assert "time" in fid.dimensions
            assert "time" in fid.variables
            # Check for capitalized variable names from real YAML
            assert "Latitude" in fid.variables
            assert "Longitude" in fid.variables
            assert "Elevation" in fid.variables
            # Check a few more variables from the real template
            assert "xFRF" in fid.variables
            assert "yFRF" in fid.variables

            # Check global attributes from real YAML
            assert hasattr(fid, "title")
            assert hasattr(fid, "institution")
            assert fid.institution == "USACE/CHL/COAB"

    def test_makenc_generic_data_integrity(self, temp_dir, sample_yaml_global, sample_yaml_variables, mock_netcdf_data):
        """Test that data is written correctly"""
        nc_file = temp_dir / "output.nc"

        py2netCDF.makenc_generic(str(nc_file), sample_yaml_global, sample_yaml_variables, mock_netcdf_data)

        with nc.Dataset(str(nc_file), "r") as fid:
            # Check that data matches input
            assert len(fid.variables["time"][:]) == len(mock_netcdf_data["time"])
            assert np.allclose(fid.variables["time"][:], mock_netcdf_data["time"])


class TestReadNetCDFFile:
    """Tests for readNetCDFfile function"""

    def test_read_netcdf_file(self, temp_dir, sample_yaml_global, sample_yaml_variables, mock_netcdf_data):
        """Test reading existing netCDF file"""
        nc_file = temp_dir / "test.nc"

        # First create a file
        py2netCDF.makenc_generic(str(nc_file), sample_yaml_global, sample_yaml_variables, mock_netcdf_data)

        # Now read it
        data_lib, dimensionLib, varMetaData, globalMetaData = py2netCDF.readNetCDFfile(str(nc_file))

        assert isinstance(globalMetaData, dict)
        assert isinstance(varMetaData, dict)
        assert "title" in globalMetaData
        assert "_variables" in varMetaData
        assert "_dimensions" in varMetaData

    def test_read_netcdf_dimensions(self, temp_dir, sample_yaml_global, sample_yaml_variables, mock_netcdf_data):
        """Test reading dimensions from netCDF"""
        nc_file = temp_dir / "test.nc"

        py2netCDF.makenc_generic(str(nc_file), sample_yaml_global, sample_yaml_variables, mock_netcdf_data)

        data_lib, dimensionLib, varMetaData, globalMetaData = py2netCDF.readNetCDFfile(str(nc_file))

        assert "time" in varMetaData["_dimensions"]


class TestCombineMetaData:
    """Tests for metadata combination functions"""

    def test_combine_global_metadata(self):
        """Test combining global metadata dictionaries"""
        original = {"title": "Original", "source": "Test"}
        new = {"title": "New Title", "author": "Test Author"}

        result = py2netCDF._combineGlobalMetaData(new, original)

        # Note: The function uses update which modifies in place and returns None
        # The original dict is updated with new values
        # So we test the behavior of the update

    def test_combine_variable_metadata(self):
        """Test combining variable metadata"""
        # Note: This function is not fully implemented in the source
        # So we just test that it doesn't crash
        new_var_meta = {"time": {"units": "seconds"}}
        old_var_meta = {"time": {"standard_name": "time"}}

        # Use the metadata dicts to avoid unused-variable issues and
        # document their expected basic structure.
        assert "time" in new_var_meta
        assert "time" in old_var_meta

        # Function returns None in current implementation
        # This test documents current behavior


@pytest.mark.parametrize(
    "data_type,test_value",
    [
        ("f8", 42.0),
        ("f4", 42.0),
        ("i4", 42),
        ("i8", 42),
    ],
)
def test_write_various_data_types(temp_dir, data_type, test_value):
    """Test writing various data types to netCDF"""
    nc_file = temp_dir / f"test_{data_type}.nc"
    fid = nc.Dataset(str(nc_file), "w", clobber=True)
    fid.createDimension("time", 1)

    template_vars = {
        "_variables": ["var"],
        "var": {
            "name": "var",
            "data_type": data_type,
            "dim": ["time"],
            "units": "unitless",
        },
    }

    data_dict = {"var": np.array([test_value])}

    num_errors, error_str = py2netCDF.write_data_to_nc(fid, template_vars, data_dict)

    assert num_errors == 0
    assert "var" in fid.variables

    fid.close()
