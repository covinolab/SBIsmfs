import pytest
from sbi_smfs.utils.config_utils import validate_config


def test_validate_config():
    assert validate_config("tests/test_2.config")
    assert validate_config("tests/test.config")


def test_validate_corrupted_config():
    with pytest.raises(KeyError):
        validate_config("tests/fail.config")
