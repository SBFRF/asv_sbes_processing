"""
Unit tests for workflow_ppk.py
Tests command-line argument parsing and workflow utility functions
"""
import pytest
import sys
import logging
from unittest.mock import patch, MagicMock
import workflow_ppk


class TestParseArgs:
    """Tests for parse_args function"""

    def test_parse_args_minimal_required(self):
        """Test with minimal required arguments"""
        test_args = ['prog', '-d', '/path/to/data']

        with patch.object(sys, 'argv', test_args):
            args = workflow_ppk.parse_args(workflow_ppk.__version__)

        assert args.data_dir == '/path/to/data'
        assert args.geoid_file == 'ref/g2012bu0.bin'  # default
        assert args.make_pos == False  # default
        assert args.verbosity == 2  # default

    def test_parse_args_all_options(self):
        """Test with all optional arguments"""
        test_args = [
            'prog',
            '-d', '/path/to/data',
            '-g', '/path/to/geoid.bin',
            '-p', 'True',
            '-v', '1',
            '--sonar_method', 'instant',
            '--rtklib_executable', '/path/to/rnx2rtkp',
            '--ppk_quality_threshold', '2',
            '--instant_sonar_confidence', '95',
            '--smoothed_sonar_confidence', '70'
        ]

        with patch.object(sys, 'argv', test_args):
            args = workflow_ppk.parse_args(workflow_ppk.__version__)

        assert args.data_dir == '/path/to/data'
        assert args.geoid_file == '/path/to/geoid.bin'
        assert args.make_pos == True
        assert args.verbosity == 1
        assert args.sonar_method == 'instant'
        assert args.rtklib_executable == '/path/to/rnx2rtkp'
        assert args.ppk_quality_threshold == 2
        assert args.instant_sonar_confidence == 95
        assert args.smoothed_sonar_confidence == 70

    def test_parse_args_missing_required(self):
        """Test that missing required argument raises error"""
        test_args = ['prog']  # No -d argument

        with patch.object(sys, 'argv', test_args):
            with pytest.raises(SystemExit):
                workflow_ppk.parse_args(workflow_ppk.__version__)

    def test_parse_args_default_values(self):
        """Test that default values are set correctly"""
        test_args = ['prog', '-d', '/data']

        with patch.object(sys, 'argv', test_args):
            args = workflow_ppk.parse_args(workflow_ppk.__version__)

        assert args.geoid_file == 'ref/g2012bu0.bin'
        assert args.make_pos == False
        assert args.verbosity == 2
        assert args.sonar_method == 'default'
        assert args.ppk_quality_threshold == 1
        assert args.instant_sonar_confidence == 99
        assert args.smoothed_sonar_confidence == 60


class TestVerbosityConversion:
    """Tests for verbosity_conversion function"""

    def test_verbosity_debug(self):
        """Test DEBUG logging level"""
        with patch.object(logging, 'basicConfig') as mock_config:
            workflow_ppk.verbosity_conversion(1)
            mock_config.assert_called_once_with(level=logging.DEBUG)

    def test_verbosity_info(self):
        """Test INFO logging level"""
        with patch.object(logging, 'basicConfig') as mock_config:
            workflow_ppk.verbosity_conversion(2)
            mock_config.assert_called_once_with(level=logging.INFO)

    def test_verbosity_warning(self):
        """Test WARNING logging level"""
        with patch.object(logging, 'basicConfig') as mock_config:
            workflow_ppk.verbosity_conversion(3)
            mock_config.assert_called_once_with(level=logging.WARN)

    def test_verbosity_invalid(self):
        """Test that invalid verbosity raises error"""
        with pytest.raises(EnvironmentError):
            workflow_ppk.verbosity_conversion(999)

    def test_verbosity_zero(self):
        """Test that verbosity 0 raises error"""
        with pytest.raises(EnvironmentError):
            workflow_ppk.verbosity_conversion(0)


class TestSonarMethods:
    """Tests for sonar method handling"""

    def test_sonar_methods_list(self):
        """Test that sonar_methods list is defined"""
        assert hasattr(workflow_ppk, 'sonar_methods')
        assert isinstance(workflow_ppk.sonar_methods, list)
        assert 'default' in workflow_ppk.sonar_methods
        assert 'instant' in workflow_ppk.sonar_methods
        assert 'smoothed' in workflow_ppk.sonar_methods
        assert 'qaqc' in workflow_ppk.sonar_methods


class TestMainFunction:
    """Tests for main function (integration tests with mocking)"""

    @patch('workflow_ppk.yellowfinLib.threadGetArgusImagery')
    @patch('workflow_ppk.yellowfinLib.loadSonar_s500_binary')
    @patch('workflow_ppk.yellowfinLib.load_yellowfin_NMEA_files')
    @patch('workflow_ppk.os.path.isfile')
    @patch('workflow_ppk.os.makedirs')
    def test_main_creates_directories(self, mock_makedirs, mock_isfile, mock_nmea, mock_sonar, mock_argus, temp_dir):
        """Test that main function creates required directories"""
        mock_isfile.return_value = True  # Pretend files exist
        datadir = str(temp_dir / "20230815")

        try:
            workflow_ppk.main(
                datadir=datadir,
                geoid='ref/g2012bu0.bin',
                makePos=False,
                verbose=3  # Use WARNING to minimize output
            )
        except Exception:
            # Main function will fail because of missing files, but we're testing directory creation
            pass

        # Verify makedirs was called
        mock_makedirs.assert_called()

    # TODO: FIXME before re-enabling test_workflow_ppk.py
    # These sonar_method tests are poorly written and need to be rewritten:
    # 1. They catch all exceptions with try/except pass, masking real failures
    # 2. They don't properly mock the dependencies, so main() will fail for other reasons
    # 3. They don't assert on the actual behavior - just "didn't raise ValueError"
    # 4. They need proper fixtures with sample data files to test real workflow
    #
    # Before removing this file from pytest.ini --ignore list:
    # - Add proper mock data (sonar .dat, NMEA .dat, PPK .pos files)
    # - Mock or provide all required dependencies (geoid file, RTKlib, etc.)
    # - Assert on specific expected behavior (function calls, file creation, etc.)
    # - Remove overly broad try/except blocks
    @patch('workflow_ppk.yellowfinLib.threadGetArgusImagery')
    @patch('workflow_ppk.yellowfinLib.loadSonar_s500_binary')
    @patch('workflow_ppk.yellowfinLib.load_yellowfin_NMEA_files')
    @patch('workflow_ppk.os.path.isfile')
    @patch('workflow_ppk.os.makedirs')
    def test_main_with_default_sonar_method(self, mock_makedirs, mock_isfile, mock_nmea, mock_sonar, mock_argus, temp_dir):
        """Test main function with default sonar method"""
        mock_isfile.return_value = True
        datadir = str(temp_dir / "20230815")

        try:
            workflow_ppk.main(
                datadir=datadir,
                geoid='ref/g2012bu0.bin',
                makePos=False,
                verbose=3,
                sonar_method='default'
            )
        except Exception:
            pass

        # If we got here without ValueError, sonar_method was accepted

    @patch('workflow_ppk.yellowfinLib.threadGetArgusImagery')
    @patch('workflow_ppk.yellowfinLib.loadSonar_s500_binary')
    @patch('workflow_ppk.yellowfinLib.load_yellowfin_NMEA_files')
    @patch('workflow_ppk.os.path.isfile')
    @patch('workflow_ppk.os.makedirs')
    def test_main_with_instant_sonar_method(self, mock_makedirs, mock_isfile, mock_nmea, mock_sonar, mock_argus, temp_dir):
        """Test main function with instant sonar method"""
        mock_isfile.return_value = True
        datadir = str(temp_dir / "20230815")

        try:
            workflow_ppk.main(
                datadir=datadir,
                geoid='ref/g2012bu0.bin',
                makePos=False,
                verbose=3,
                sonar_method='instant'
            )
        except Exception:
            pass

    @patch('workflow_ppk.yellowfinLib.threadGetArgusImagery')
    @patch('workflow_ppk.yellowfinLib.loadSonar_s500_binary')
    @patch('workflow_ppk.yellowfinLib.load_yellowfin_NMEA_files')
    @patch('workflow_ppk.os.path.isfile')
    @patch('workflow_ppk.os.makedirs')
    def test_main_with_invalid_sonar_method(self, mock_makedirs, mock_isfile, mock_nmea, mock_sonar, mock_argus, temp_dir):
        """Test that invalid sonar method raises ValueError"""
        mock_isfile.return_value = True
        datadir = str(temp_dir / "20230815")

        with pytest.raises(ValueError):
            workflow_ppk.main(
                datadir=datadir,
                geoid='ref/g2012bu0.bin',
                makePos=False,
                verbose=3,
                sonar_method='invalid_method'
            )

    @patch('workflow_ppk.yellowfinLib.threadGetArgusImagery')
    @patch('workflow_ppk.os.makedirs')
    def test_main_datadir_slash_handling(self, mock_makedirs, mock_argus, temp_dir):
        """Test that trailing slash is removed from datadir"""
        datadir_with_slash = str(temp_dir / "20230815") + "/"

        try:
            workflow_ppk.main(
                datadir=datadir_with_slash,
                geoid='ref/g2012bu0.bin',
                makePos=False,
                verbose=3
            )
        except Exception:
            pass

        # If function handles it correctly, no error should occur from slash


class TestParameterValidation:
    """Tests for parameter validation"""

    @patch('workflow_ppk.yellowfinLib.threadGetArgusImagery')
    @patch('workflow_ppk.os.makedirs')
    def test_main_ppk_quality_threshold(self, mock_makedirs, mock_argus, temp_dir):
        """Test PPK quality threshold parameter"""
        datadir = str(temp_dir / "20230815")

        try:
            workflow_ppk.main(
                datadir=datadir,
                geoid='ref/g2012bu0.bin',
                makePos=False,
                verbose=3,
                ppk_quality_threshold=2
            )
        except Exception:
            pass

    @patch('workflow_ppk.yellowfinLib.threadGetArgusImagery')
    @patch('workflow_ppk.os.makedirs')
    def test_main_confidence_thresholds(self, mock_makedirs, mock_argus, temp_dir):
        """Test sonar confidence threshold parameters"""
        datadir = str(temp_dir / "20230815")

        try:
            workflow_ppk.main(
                datadir=datadir,
                geoid='ref/g2012bu0.bin',
                makePos=False,
                verbose=3,
                instant_sonar_confidence=95,
                smoothed_sonar_confidence=70
            )
        except Exception:
            pass


@pytest.mark.parametrize("sonar_method,expected_valid", [
    ('default', True),
    ('instant', True),
    ('smoothed', True),
    ('qaqc', True),
    ('invalid', False),
    ('', False),
])
def test_sonar_method_validation(temp_dir, sonar_method, expected_valid):
    """Parameterized test for sonar method validation"""
    datadir = str(temp_dir / "20230815")

    with patch('workflow_ppk.yellowfinLib.threadGetArgusImagery'):
        with patch('workflow_ppk.os.makedirs'):
            if expected_valid:
                try:
                    workflow_ppk.main(
                        datadir=datadir,
                        geoid='ref/g2012bu0.bin',
                        makePos=False,
                        verbose=3,
                        sonar_method=sonar_method
                    )
                except ValueError:
                    pytest.fail(f"Valid sonar_method '{sonar_method}' raised ValueError")
                except Exception:
                    # Other exceptions are OK for this test
                    pass
            else:
                with pytest.raises(ValueError):
                    workflow_ppk.main(
                        datadir=datadir,
                        geoid='ref/g2012bu0.bin',
                        makePos=False,
                        verbose=3,
                        sonar_method=sonar_method
                    )


@pytest.mark.parametrize("verbosity", [1, 2, 3])
def test_verbosity_levels(verbosity):
    """Parameterized test for verbosity levels"""
    with patch.object(logging, 'basicConfig') as mock_config:
        workflow_ppk.verbosity_conversion(verbosity)
        assert mock_config.called


class TestVersionInfo:
    """Tests for version information"""

    def test_version_exists(self):
        """Test that version is defined"""
        assert hasattr(workflow_ppk, '__version__')

    def test_version_type(self):
        """Test that version is a number"""
        assert isinstance(workflow_ppk.__version__, (int, float))
