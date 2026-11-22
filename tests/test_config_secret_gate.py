"""Regression tests for the SECRET_KEY strength gate (P0-4)."""
from api.core.config import Settings


def test_default_secret_key_is_weak():
    assert Settings(_env_file=None).secret_key_is_weak is True


def test_placeholder_env_example_value_is_weak():
    config = Settings(_env_file=None, SECRET_KEY="change-me-in-production-min-32-chars")
    assert config.secret_key_is_weak is True


def test_short_secret_key_is_weak():
    config = Settings(_env_file=None, SECRET_KEY="short")
    assert config.secret_key_is_weak is True


def test_strong_secret_key_is_not_weak():
    config = Settings(_env_file=None, SECRET_KEY="wst_" + "x" * 48)
    assert config.secret_key_is_weak is False
