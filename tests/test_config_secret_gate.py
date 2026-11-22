"""Regression tests for the SECRET_KEY strength gate (S2 / P0-4).

Covers the startup gate for every environment/key combination:
production missing/empty/placeholder/valid key, and development bypass.
No real secret values are used or printed.
"""
import pytest

from api.core.config import Settings


def _strong_key() -> str:
    return "wst_" + "x" * 48


# --- secret_key_is_weak property -------------------------------------------


def test_default_secret_key_is_weak():
    assert Settings(_env_file=None).secret_key_is_weak is True


def test_placeholder_env_example_value_is_weak():
    config = Settings(_env_file=None, SECRET_KEY="change-me-in-production-min-32-chars")
    assert config.secret_key_is_weak is True


def test_known_placeholder_value_is_weak():
    config = Settings(_env_file=None, SECRET_KEY="change-me-in-production")
    assert config.secret_key_is_weak is True


def test_short_secret_key_is_weak():
    config = Settings(_env_file=None, SECRET_KEY="short")
    assert config.secret_key_is_weak is True


def test_strong_secret_key_is_not_weak():
    config = Settings(_env_file=None, SECRET_KEY=_strong_key())
    assert config.secret_key_is_weak is False


# --- production gate (DEBUG=false): startup must be refused ----------------


def test_production_missing_secret_key_refused():
    # No SECRET_KEY provided at all -> default empty -> weak.
    config = Settings(_env_file=None, DEBUG=False)
    with pytest.raises(RuntimeError, match="SECRET_KEY"):
        config.enforce_secret_key_for_production()


def test_production_empty_secret_key_refused():
    config = Settings(_env_file=None, DEBUG=False, SECRET_KEY="")
    with pytest.raises(RuntimeError, match="SECRET_KEY"):
        config.enforce_secret_key_for_production()


def test_production_placeholder_secret_key_refused():
    config = Settings(_env_file=None, DEBUG=False, SECRET_KEY="change-me-in-production")
    with pytest.raises(RuntimeError, match="SECRET_KEY"):
        config.enforce_secret_key_for_production()


def test_production_short_secret_key_refused():
    config = Settings(_env_file=None, DEBUG=False, SECRET_KEY="abc123")
    with pytest.raises(RuntimeError, match="SECRET_KEY"):
        config.enforce_secret_key_for_production()


def test_production_valid_secret_key_accepted():
    config = Settings(_env_file=None, DEBUG=False, SECRET_KEY=_strong_key())
    config.enforce_secret_key_for_production()  # must not raise


# --- development gate (DEBUG=true): weak key allowed -----------------------


def test_development_weak_secret_key_allowed():
    config = Settings(_env_file=None, DEBUG=True, SECRET_KEY="")
    config.enforce_secret_key_for_production()  # must not raise


def test_development_placeholder_secret_key_allowed():
    config = Settings(_env_file=None, DEBUG=True, SECRET_KEY="change-me-in-production")
    config.enforce_secret_key_for_production()  # must not raise


# --- error message must not leak the secret value ---------------------------


def test_gate_error_message_does_not_leak_secret_value():
    # Short distinctive value: weak because <32 chars, so the gate must raise.
    secret_value = "LEAKCHECK-distinctive-value"
    config = Settings(_env_file=None, DEBUG=False, SECRET_KEY=secret_value)
    with pytest.raises(RuntimeError) as exc_info:
        config.enforce_secret_key_for_production()
    assert secret_value not in str(exc_info.value)
    assert "SECRET_KEY" in str(exc_info.value)
