"""Baseline tests: the package imports and the CLI is wired up."""

import subprocess
import sys

import alpinemet
from alpinemet.cli import build_parser, main


def test_version_is_exposed():
    assert isinstance(alpinemet.__version__, str)
    assert alpinemet.__version__


def test_parser_builds():
    parser = build_parser()
    assert parser.prog == "alpinemet"


def test_main_without_command_returns_nonzero(capsys):
    assert main([]) == 1
    assert "usage" in capsys.readouterr().out


def test_console_script_runs():
    result = subprocess.run(
        [sys.executable, "-m", "alpinemet.cli", "--version"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "alpinemet" in result.stdout
