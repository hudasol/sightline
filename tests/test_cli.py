"""Command line contract: exit codes, version and the parser itself."""
import tomllib
from pathlib import Path

import pytest

import sightline
from sightline.cli import build_parser, main
from sightline.plots import make_all

ROOT = Path(__file__).resolve().parent.parent


def test_usage_error_exits_with_1_not_2(capsys):
    # Exit code 2 means UNAVAILABLE, so a mistyped command must not look like it.
    with pytest.raises(SystemExit) as e:
        main(["no-such-command"])
    assert e.value.code == 1
    assert "invalid choice" in capsys.readouterr().err


def test_missing_command_and_missing_argument_exit_with_1():
    for argv in ([], ["replay", "x.mp4"]):
        with pytest.raises(SystemExit) as e:
            main(argv)
        assert e.value.code == 1


def test_version_exits_0_and_matches_pyproject(capsys):
    with pytest.raises(SystemExit) as e:
        main(["--version"])
    assert e.value.code == 0
    declared = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
    assert sightline.__version__ == declared
    assert declared in capsys.readouterr().out


def test_every_subcommand_has_working_help(capsys):
    parser = build_parser()
    names = [a for a in parser._subparsers._group_actions[0].choices]
    assert len(names) >= 25
    for name in names:
        with pytest.raises(SystemExit) as e:
            main([name, "--help"])
        assert e.value.code == 0, name
    assert "usage:" in capsys.readouterr().out


def test_plots_on_an_empty_results_folder_makes_nothing(tmp_path):
    assert make_all(tmp_path) == []
