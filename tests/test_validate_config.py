import pytest
from sbi_smfs.utils.config_utils import validate_config


def test_validate_config():
    assert validate_config("tests/config_files/test_2.config")
    assert validate_config("tests/config_files/test.config")


def test_validate_corrupted_config():
    with pytest.raises(KeyError):
        validate_config("tests/config_files/fail.config")
