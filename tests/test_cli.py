"""The CLI admits only explicit fixed-H GEV fitting and offline guidance."""

import pytest

from ffa_methods import __version__
from ffa_methods.cli import main


def test_help_is_successful(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--help"])
    assert exit_info.value.code == 0
    output = capsys.readouterr().out
    assert "FFA Toolkit: independent flood frequency analysis" in output
    assert "complete-positive-AMS fixed-H GEV fitting is" in output


def test_version_distinguishes_package_and_vendor(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])
    assert exit_info.value.code == 0
    output = capsys.readouterr().out
    assert f"ffa {__version__}" in output
    assert "5.0.306.0" in output
    assert "file 3.10" in output


@pytest.mark.parametrize("args", [[], ["fit"], ["--unexpected"], ["--ver"]])
def test_unimplemented_operation_is_failure(args: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(args)
    assert exit_info.value.code == 2
    assert "error:" in capsys.readouterr().err
