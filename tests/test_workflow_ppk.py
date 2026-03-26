"""
Unit tests for workflow_ppk.py
Tests command-line argument parsing and workflow utility functions
"""
import pytest
import sys
import logging
import os
from unittest.mock import patch
import workflow_ppk


class _SentinelException(Exception):
    """Raised by mocks to halt main() execution at a known point after
    the code under test has already run (e.g. after sonar_method validation).
    This avoids broad try/except blocks and lets tests assert precisely."""


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
            '-p',
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
    """Tests for main function (integration tests with mocking)

    Uses a sentinel exception pattern: mock a dependency that executes
    *after* the code under test to raise _SentinelException, halting
    main() at a known point.  If main() raises _SentinelException
    (not ValueError or another error), the tested code path succeeded.
    """

    @patch('workflow_ppk.yellowfinLib.threadGetArgusImagery', side_effect=_SentinelException)
    @patch('workflow_ppk.os.makedirs')
    def test_main_creates_directories(self, mock_makedirs, mock_argus, temp_dir):
        """Test that main function creates required directories"""
        datadir = str(temp_dir / "20230815")

        with pytest.raises(_SentinelException):
            workflow_ppk.main(
                datadir=datadir,
                geoid='ref/g2012bu0.bin',
                makePos=False,
                verbose=3
            )

        # Verify makedirs was called with the figures directory
        expected_plot_dir = os.path.join(datadir, 'figures')
        mock_makedirs.assert_called_once_with(expected_plot_dir, exist_ok=True)

    @patch('workflow_ppk.os.makedirs', side_effect=_SentinelException)
    def test_main_with_default_sonar_method(self, mock_makedirs, temp_dir):
        """Test main function accepts 'default' sonar method without ValueError"""
        datadir = str(temp_dir / "20230815")

        # _SentinelException proves we got past sonar_method validation (no ValueError)
        with pytest.raises(_SentinelException):
            workflow_ppk.main(
                datadir=datadir,
                geoid='ref/g2012bu0.bin',
                makePos=False,
                verbose=3,
                sonar_method='default'
            )

        mock_makedirs.assert_called_once()

    @patch('workflow_ppk.os.makedirs', side_effect=_SentinelException)
    def test_main_with_instant_sonar_method(self, mock_makedirs, temp_dir):
        """Test main function accepts 'instant' sonar method without ValueError"""
        datadir = str(temp_dir / "20230815")

        with pytest.raises(_SentinelException):
            workflow_ppk.main(
                datadir=datadir,
                geoid='ref/g2012bu0.bin',
                makePos=False,
                verbose=3,
                sonar_method='instant'
            )

        mock_makedirs.assert_called_once()

    def test_main_with_invalid_sonar_method(self, temp_dir):
        """Test that invalid sonar method raises ValueError"""
        datadir = str(temp_dir / "20230815")

        with pytest.raises(ValueError, match='acceptable sonar methods'):
            workflow_ppk.main(
                datadir=datadir,
                geoid='ref/g2012bu0.bin',
                makePos=False,
                verbose=3,
                sonar_method='invalid_method'
            )

    @patch('workflow_ppk.yellowfinLib.threadGetArgusImagery', side_effect=_SentinelException)
    @patch('workflow_ppk.os.makedirs')
    def test_main_datadir_slash_handling(self, mock_makedirs, mock_argus, temp_dir):
        """Test that trailing slash is removed from datadir"""
        datadir_with_slash = str(temp_dir / "20230815") + "/"

        with pytest.raises(_SentinelException):
            workflow_ppk.main(
                datadir=datadir_with_slash,
                geoid='ref/g2012bu0.bin',
                makePos=False,
                verbose=3
            )

        # Verify makedirs was called with a path under the slash-stripped datadir
        expected_plot_dir = os.path.join(str(temp_dir / "20230815"), 'figures')
        mock_makedirs.assert_called_once_with(expected_plot_dir, exist_ok=True)


class TestParameterValidation:
    """Tests for parameter validation"""

    @patch('workflow_ppk.os.makedirs', side_effect=_SentinelException)
    def test_main_ppk_quality_threshold(self, mock_makedirs, temp_dir):
        """Test PPK quality threshold parameter is accepted"""
        datadir = str(temp_dir / "20230815")

        with pytest.raises(_SentinelException):
            workflow_ppk.main(
                datadir=datadir,
                geoid='ref/g2012bu0.bin',
                makePos=False,
                verbose=3,
                ppk_quality_threshold=2
            )

        mock_makedirs.assert_called_once()

    @patch('workflow_ppk.os.makedirs', side_effect=_SentinelException)
    def test_main_confidence_thresholds(self, mock_makedirs, temp_dir):
        """Test sonar confidence threshold parameters are accepted"""
        datadir = str(temp_dir / "20230815")

        with pytest.raises(_SentinelException):
            workflow_ppk.main(
                datadir=datadir,
                geoid='ref/g2012bu0.bin',
                makePos=False,
                verbose=3,
                instant_sonar_confidence=95,
                smoothed_sonar_confidence=70
            )

        mock_makedirs.assert_called_once()


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

    if expected_valid:
        # _SentinelException on os.makedirs halts main() right after validation;
        # if we see it instead of ValueError, the sonar_method was accepted.
        with patch('workflow_ppk.os.makedirs', side_effect=_SentinelException):
            with pytest.raises(_SentinelException):
                workflow_ppk.main(
                    datadir=datadir,
                    geoid='ref/g2012bu0.bin',
                    makePos=False,
                    verbose=3,
                    sonar_method=sonar_method
                )
    else:
        with pytest.raises(ValueError, match='acceptable sonar methods'):
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
