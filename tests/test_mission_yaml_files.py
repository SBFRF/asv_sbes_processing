"""
Unit tests for mission_yaml_files.py
Tests YAML metadata file generation functions
"""
import pytest
import os
import yaml
from unittest.mock import patch, MagicMock
import mission_yaml_files


class TestMakeSummaryYaml:
    """Tests for make_summary_yaml function"""

    @patch('mission_yaml_files._is_non_interactive', return_value=False)
    @patch('builtins.input', side_effect=['y', 'n', 'n', ''])
    def test_make_summary_yaml_basic(self, mock_input, mock_non_interactive, temp_dir):
        """Test creating summary YAML with basic inputs"""
        mission_yaml_files.make_summary_yaml(str(temp_dir))

        # Check file was created
        yaml_file = temp_dir / "mission_summary_metadata.yaml"
        assert os.path.exists(yaml_file)

        # Read and verify contents
        with open(yaml_file, 'r') as f:
            content = f.read()
            assert 'frf: y' in content
            assert 'overlap: n' in content
            assert 'repeat: n' in content

    @patch('mission_yaml_files._is_non_interactive', return_value=False)
    @patch('builtins.input', side_effect=['y', 'y', 'y', '3', 'Test notes'])
    def test_make_summary_yaml_with_repeats(self, mock_input, mock_non_interactive, temp_dir):
        """Test creating summary YAML with repeat lines"""
        mission_yaml_files.make_summary_yaml(str(temp_dir))

        yaml_file = temp_dir / "mission_summary_metadata.yaml"
        assert os.path.exists(yaml_file)

        with open(yaml_file, 'r') as f:
            content = f.read()
            assert 'frf: y' in content
            assert 'overlap: y' in content
            assert 'repeat: y' in content
            assert 'repeat_count:' in content
            assert 'Test notes' in content

    @patch('mission_yaml_files._is_non_interactive', return_value=False)
    @patch('builtins.input', side_effect=['n'])
    def test_make_summary_yaml_no_overwrite(self, mock_input, mock_non_interactive, temp_dir):
        """Test not overwriting existing file"""
        yaml_file = temp_dir / "mission_summary_metadata.yaml"

        # Create existing file
        with open(yaml_file, 'w') as f:
            f.write("existing content")

        mission_yaml_files.make_summary_yaml(str(temp_dir))

        # File should still have original content
        with open(yaml_file, 'r') as f:
            content = f.read()
            assert content == "existing content"

    @patch('mission_yaml_files._is_non_interactive', return_value=False)
    @patch('builtins.input', side_effect=['y', 'y', 'n', 'n', ''])
    def test_make_summary_yaml_overwrite(self, mock_input, mock_non_interactive, temp_dir):
        """Test overwriting existing file"""
        yaml_file = temp_dir / "mission_summary_metadata.yaml"

        # Create existing file
        with open(yaml_file, 'w') as f:
            f.write("old content")

        mission_yaml_files.make_summary_yaml(str(temp_dir))

        # File should have new content
        with open(yaml_file, 'r') as f:
            content = f.read()
            assert "old content" not in content
            assert "frf: y" in content

    @patch('mission_yaml_files._is_non_interactive', return_value=False)
    @patch('builtins.input', side_effect=['invalid', 'y', 'n', 'n', ''])
    def test_make_summary_yaml_invalid_input(self, mock_input, mock_non_interactive, temp_dir):
        """Test handling of invalid input"""
        # Should handle invalid input and retry
        mission_yaml_files.make_summary_yaml(str(temp_dir))

        yaml_file = temp_dir / "mission_summary_metadata.yaml"
        assert os.path.exists(yaml_file)

    @patch('mission_yaml_files._is_non_interactive', return_value=False)
    @patch('builtins.input', side_effect=['y', 'y', 'y', 'invalid', '2', ''])
    def test_make_summary_yaml_invalid_number(self, mock_input, mock_non_interactive, temp_dir):
        """Test handling of invalid repeat count"""
        mission_yaml_files.make_summary_yaml(str(temp_dir))

        yaml_file = temp_dir / "mission_summary_metadata.yaml"
        assert os.path.exists(yaml_file)

        with open(yaml_file, 'r') as f:
            content = f.read()
            assert 'repeat_count:' in content


class TestMakeFailureYaml:
    """Tests for make_failure_yaml function"""

    @patch('mission_yaml_files._is_non_interactive', return_value=False)
    @patch('builtins.input', side_effect=['n', ''])
    def test_make_failure_yaml_no_failure(self, mock_input, mock_non_interactive, temp_dir):
        """Test creating failure YAML with no failures"""
        mission_yaml_files.make_failure_yaml(str(temp_dir))

        yaml_file = temp_dir / "mission_failure_metadata.yaml"
        assert os.path.exists(yaml_file)

        with open(yaml_file, 'r') as f:
            content = f.read()
            assert 'mission_failure: n' in content

    @patch('mission_yaml_files._is_non_interactive', return_value=False)
    @patch('builtins.input', side_effect=['y', '1', 'Mechanical issue', '0', '0', 'General notes'])
    def test_make_failure_yaml_with_mechanical(self, mock_input, mock_non_interactive, temp_dir):
        """Test creating failure YAML with mechanical failure"""
        mission_yaml_files.make_failure_yaml(str(temp_dir))

        yaml_file = temp_dir / "mission_failure_metadata.yaml"
        assert os.path.exists(yaml_file)

        with open(yaml_file, 'r') as f:
            content = f.read()
            assert 'mission_failure: y' in content
            assert 'mechanical_failure:' in content
            assert 'Mechanical issue' in content

    @patch('mission_yaml_files._is_non_interactive', return_value=False)
    @patch('builtins.input', side_effect=['y', '0', '1', 'Data quality issue', '2', 'Hydrodynamic problem', 'Notes'])
    def test_make_failure_yaml_multiple_failures(self, mock_input, mock_non_interactive, temp_dir):
        """Test creating failure YAML with multiple failure types"""
        mission_yaml_files.make_failure_yaml(str(temp_dir))

        yaml_file = temp_dir / "mission_failure_metadata.yaml"
        assert os.path.exists(yaml_file)

        with open(yaml_file, 'r') as f:
            content = f.read()
            assert 'mission_failure: y' in content
            assert 'Data quality issue' in content
            assert 'Hydrodynamic problem' in content

    @patch('mission_yaml_files._is_non_interactive', return_value=False)
    @patch('builtins.input', side_effect=['n'])
    def test_make_failure_yaml_no_overwrite(self, mock_input, mock_non_interactive, temp_dir):
        """Test not overwriting existing failure file"""
        yaml_file = temp_dir / "mission_failure_metadata.yaml"

        # Create existing file
        with open(yaml_file, 'w') as f:
            f.write("existing failure data")

        mission_yaml_files.make_failure_yaml(str(temp_dir))

        with open(yaml_file, 'r') as f:
            content = f.read()
            assert content == "existing failure data"

    @patch('mission_yaml_files._is_non_interactive', return_value=False)
    @patch('builtins.input', side_effect=['y', 'y', '2', 'Rescue required', '0', '0', ''])
    def test_make_failure_yaml_overwrite(self, mock_input, mock_non_interactive, temp_dir):
        """Test overwriting existing failure file"""
        yaml_file = temp_dir / "mission_failure_metadata.yaml"

        # Create existing file
        with open(yaml_file, 'w') as f:
            f.write("old data")

        mission_yaml_files.make_failure_yaml(str(temp_dir))

        with open(yaml_file, 'r') as f:
            content = f.read()
            assert "old data" not in content
            assert "mission_failure: y" in content

    @patch('mission_yaml_files._is_non_interactive', return_value=False)
    @patch('builtins.input', side_effect=['invalid', 'y', '0', '0', '0', ''])
    def test_make_failure_yaml_invalid_yn_input(self, mock_input, mock_non_interactive, temp_dir):
        """Test handling of invalid y/n input"""
        mission_yaml_files.make_failure_yaml(str(temp_dir))

        yaml_file = temp_dir / "mission_failure_metadata.yaml"
        assert os.path.exists(yaml_file)

    @patch('mission_yaml_files._is_non_interactive', return_value=False)
    @patch('builtins.input', side_effect=['y', 'invalid', '1', 'Comment', '0', '0', ''])
    def test_make_failure_yaml_invalid_category_input(self, mock_input, mock_non_interactive, temp_dir):
        """Test handling of invalid category input"""
        mission_yaml_files.make_failure_yaml(str(temp_dir))

        yaml_file = temp_dir / "mission_failure_metadata.yaml"
        assert os.path.exists(yaml_file)


class TestYamlFileStructure:
    """Tests for YAML file structure and content"""

    @patch('mission_yaml_files._is_non_interactive', return_value=False)
    @patch('builtins.input', side_effect=['y', 'n', 'n', ''])
    def test_summary_yaml_is_valid_yaml(self, mock_input, mock_non_interactive, temp_dir):
        """Test that generated summary YAML is valid"""
        mission_yaml_files.make_summary_yaml(str(temp_dir))

        yaml_file = temp_dir / "mission_summary_metadata.yaml"

        # Should be loadable as YAML
        with open(yaml_file, 'r') as f:
            # Skip comment lines
            lines = [line for line in f if not line.strip().startswith('#')]
            yaml_content = ''.join(lines)
            data = yaml.safe_load(yaml_content)

        assert isinstance(data, dict)
        assert 'frf' in data
        assert 'overlap' in data
        assert 'repeat' in data

    @patch('mission_yaml_files._is_non_interactive', return_value=False)
    @patch('builtins.input', side_effect=['y', '0', '0', '0', ''])
    def test_failure_yaml_is_valid_yaml(self, mock_input, mock_non_interactive, temp_dir):
        """Test that generated failure YAML is valid"""
        mission_yaml_files.make_failure_yaml(str(temp_dir))

        yaml_file = temp_dir / "mission_failure_metadata.yaml"

        with open(yaml_file, 'r') as f:
            lines = [line for line in f if not line.strip().startswith('#')]
            yaml_content = ''.join(lines)
            data = yaml.safe_load(yaml_content)

        assert isinstance(data, dict)
        assert 'mission_failure' in data

    @patch('mission_yaml_files._is_non_interactive', return_value=False)
    @patch('builtins.input', side_effect=['y', 'n', 'y', '5', 'Custom notes here'])
    def test_summary_yaml_preserves_user_notes(self, mock_input, mock_non_interactive, temp_dir):
        """Test that user notes are preserved in summary YAML"""
        mission_yaml_files.make_summary_yaml(str(temp_dir))

        yaml_file = temp_dir / "mission_summary_metadata.yaml"

        with open(yaml_file, 'r') as f:
            content = f.read()
            assert 'Custom notes here' in content
            assert '# User Notes:' in content

    @patch('mission_yaml_files._is_non_interactive', return_value=False)
    @patch('builtins.input', side_effect=['y', '1', 'Issue description', '0', '0', 'Additional notes'])
    def test_failure_yaml_preserves_comments(self, mock_input, mock_non_interactive, temp_dir):
        """Test that failure comments are preserved"""
        mission_yaml_files.make_failure_yaml(str(temp_dir))

        yaml_file = temp_dir / "mission_failure_metadata.yaml"

        with open(yaml_file, 'r') as f:
            content = f.read()
            assert 'Issue description' in content
            assert 'Additional notes' in content


@pytest.mark.parametrize("frf,overlap,repeat,count", [
    ('y', 'y', 'y', '3'),
    ('n', 'n', 'n', ''),
    ('y', 'n', 'y', '1'),
    ('n', 'y', 'n', ''),
])
@patch('mission_yaml_files._is_non_interactive', return_value=False)
def test_summary_yaml_various_inputs(mock_non_interactive, temp_dir, frf, overlap, repeat, count):
    """Parameterized test for various input combinations"""
    inputs = [frf, overlap, repeat]
    if repeat == 'y':
        inputs.append(count)
    inputs.append('')  # Notes

    with patch('builtins.input', side_effect=inputs):
        mission_yaml_files.make_summary_yaml(str(temp_dir))

        yaml_file = temp_dir / "mission_summary_metadata.yaml"
        assert os.path.exists(yaml_file)

        with open(yaml_file, 'r') as f:
            content = f.read()
            assert f'frf: {frf}' in content
            assert f'overlap: {overlap}' in content
            assert f'repeat: {repeat}' in content
