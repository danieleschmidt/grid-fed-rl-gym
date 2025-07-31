"""Test command-line interface."""

import pytest
from grid_fed_rl.cli import main


def test_cli_help():
    """Test CLI help output."""
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])
    assert exc_info.value.code == 0


def test_cli_version():
    """Test CLI version output.""" 
    with pytest.raises(SystemExit) as exc_info:
        main(["--version"])
    assert exc_info.value.code == 0


def test_cli_no_command():
    """Test CLI with no command shows help."""
    result = main([])
    assert result == 1


def test_cli_train_command():
    """Test train command with required arguments."""
    result = main(["train", "--config", "test_config.yaml"])
    assert result == 0


def test_cli_evaluate_command():
    """Test evaluate command with required arguments."""
    result = main(["evaluate", "--model", "test_model.pt", "--env", "test_env.yaml"])
    assert result == 0


def test_cli_simulate_command():
    """Test simulate command with required arguments."""
    result = main(["simulate", "--feeder", "IEEE13Bus", "--duration", "3600"])
    assert result == 0