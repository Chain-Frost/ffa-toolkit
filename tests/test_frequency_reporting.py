"""Verify the opt-in presentation against independent plotting coordinates and report values."""

import csv
import json
from dataclasses import replace
from math import ceil, isclose, log10
from pathlib import Path
from typing import cast

import pytest

from ffa_methods.cli import main
from ffa_methods.contracts import ResultMeaning
from ffa_methods.deterministic_workflow import calculate_from_bytes
from ffa_methods.frequency_reporting import render_frequency_files

EXAMPLES = Path(__file__).resolve().parents[1] / "docs" / "examples"


def _run(ams: Path, config: Path, destination: Path, *, plot: bool = True) -> int:
    args = ["fit", "--ams", str(ams), "--settings", str(config), "--output", str(destination)]
    return main([*args, "--frequency-report"] if plot else args)


def _rows(directory: Path) -> list[dict[str, str]]:
    with (directory / "frequency-coordinates.csv").open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


@pytest.mark.parametrize(
    ("estimator", "extra"),
    [
        ("gumbel_lmoments", {}),
        ("lognormal_lmoments", {"reference_discharge": 2}),
        ("lp3_lmoments", {"reference_discharge": 2}),
        ("gp_lmoments", {"gp_location_mode": "fixed", "gp_location": 0.25}),
        ("gp_lmoments", {"gp_location_mode": "free"}),
    ],
)
def test_ordinary_frequency_plot_coordinates(tmp_path: Path, estimator: str, extra: dict[str, object]) -> None:
    ams = EXAMPLES / "ordinary-ams.csv"
    config = tmp_path / "settings.json"
    settings = cast(dict[str, object], json.loads((EXAMPLES / "ordinary-settings.json").read_text(encoding="utf-8")))
    config.write_text(json.dumps({**settings, "estimator": estimator, **extra}), encoding="utf-8")
    plotted, plain = tmp_path / "plotted", tmp_path / "plain"
    assert _run(ams, config, plotted) == 0
    assert _run(ams, config, plain, plot=False) == 0
    for name in ("results.json", "results.csv", "report.txt", "source-ams.csv", "source-settings.json"):
        assert (plotted / name).read_bytes() == (plain / name).read_bytes()
    rows = _rows(plotted)
    observed, fitted = rows[:3], rows[3:]
    assert len(rows) == 5
    assert [int(row["year"]) for row in observed] == [2002, 2001, 2000]
    assert [int(row["source_index"]) for row in observed] == [3, 2, 1]
    assert [int(row["descending_rank"]) for row in observed] == [1, 2, 3]
    assert [float(row["discharge"]) for row in observed] == [4.0, 2.0, 1.0]
    for row, expected in zip(observed, (3 / 16, 1 / 2, 13 / 16), strict=True):
        assert isclose(float(row["AEP"]), expected, rel_tol=1e-14)
        assert isclose(float(row["F"]), 1 - expected, rel_tol=1e-14)
    with (plotted / "results.csv").open(encoding="utf-8", newline="") as stream:
        results = list(csv.DictReader(stream))
    for row, result in zip(fitted, results, strict=True):
        assert float(row["discharge"]) == float(result["value"])
        assert float(row["F"]) == float(result["input_non_exceedance_probability"])
        assert float(row["AEP"]) == float(result["input_annual_exceedance_probability"])
        assert row["unit"] == result["unit"]
    svg = (plotted / "frequency.svg").read_text(encoding="utf-8")
    html = (plotted / "report.html").read_text(encoding="utf-8")
    low = min(0.0, *(float(row["discharge"]) for row in rows))
    high = max(float(row["discharge"]) for row in rows)
    decades = max(1, ceil(-log10(min(float(row["AEP"]) for row in rows))))
    for row in rows:
        aep, q = float(row["AEP"]), float(row["discharge"])
        x, y = float(row["svg_x"]), float(row["svg_y"])
        assert isclose(x, 110 - 850 * log10(aep) / decades, abs_tol=1e-10)
        assert isclose(y, 445 - 390 * (q - low) / (high - low), abs_tol=1e-10)
        assert f'cx="{row["svg_x"]}" cy="{row["svg_y"]}"' in svg
    assert svg.strip() in html
    assert "No uncertainty intervals are calculated" in html


@pytest.mark.parametrize("shift", range(5))
def test_fixed_gev_frequency_report_custody(tmp_path: Path, shift: int) -> None:
    ams = EXAMPLES / "fixed-gev-ams.csv"
    config = tmp_path / "settings.json"
    settings = cast(dict[str, object], json.loads((EXAMPLES / "fixed-gev-settings.json").read_text(encoding="utf-8")))
    config.write_text(json.dumps({**settings, "lh_shift": shift}), encoding="utf-8")
    plotted, plain = tmp_path / "plot", tmp_path / "plain"
    assert _run(ams, config, plotted) == 0
    assert _run(ams, config, plain, plot=False) == 0
    assert (plotted / "source-ams.csv").read_bytes() == ams.read_bytes()
    assert (plotted / "source-settings.json").read_bytes() == config.read_bytes()
    for name in ("results.json", "results.csv", "report.txt"):
        assert (plotted / name).read_bytes() == (plain / name).read_bytes()
    assert len([row for row in _rows(plotted) if row["series"] == "observed"]) == 10
    assert "GEV support:" in (plotted / "frequency.svg").read_text(encoding="utf-8")


def test_ties_and_unordered_source_keep_every_row(tmp_path: Path) -> None:
    ams = tmp_path / "ams.csv"
    ams.write_bytes(b"Year,AMS\n2002,4\n2000,4\n2001,1\n")
    destination = tmp_path / "result"
    assert _run(ams, EXAMPLES / "ordinary-settings.json", destination) == 0
    observed = _rows(destination)[:3]
    assert [int(row["year"]) for row in observed] == [2000, 2002, 2001]
    assert [int(row["source_index"]) for row in observed] == [2, 1, 3]


@pytest.mark.parametrize(
    "data",
    [
        b"Year,AMS\n2000,1\n2001,2\n2003,4\n",
        b"Year,AMS\n2000,1\n2001,\n2002,4\n",
    b"Year,AMS\n2000,1\n2001,0\n2002,4\n",
    b"Year,AMS,Flag\n2000,1,0\n2001,2,1\n2002,4,0\n",
    ],
)
def test_unadmitted_frequency_data_never_publish(tmp_path: Path, data: bytes) -> None:
    ams = tmp_path / "ams.csv"
    ams.write_bytes(data)
    destination = tmp_path / "result"
    with pytest.raises(SystemExit) as exc:
        _run(ams, EXAMPLES / "ordinary-settings.json", destination)
    assert exc.value.code == 2
    assert not destination.exists()
    assert not list(tmp_path.glob(".result-*"))


def test_single_signed_quantile_and_escaped_unit(tmp_path: Path) -> None:
    config = tmp_path / "settings.json"
    settings = cast(dict[str, object], json.loads((EXAMPLES / "ordinary-settings.json").read_text(encoding="utf-8")))
    config.write_text(
        json.dumps({**settings, "non_exceedance_probabilities": [0.01], "discharge_unit": "m3/s <&>"}),
        encoding="utf-8",
    )
    destination = tmp_path / "report"
    assert _run(EXAMPLES / "ordinary-ams.csv", config, destination) == 0
    row = _rows(destination)[-1]
    assert float(row["discharge"]) < 0
    assert row["unit"] == "m3/s <&>"
    svg = (destination / "frequency.svg").read_text(encoding="utf-8")
    assert "<polyline" not in svg
    assert "m3/s &lt;&amp;&gt;" in svg
    assert "Negative mathematical quantiles" in svg


def test_gev_support_violations_remain_visible(tmp_path: Path) -> None:
    ams = tmp_path / "ams.csv"
    ams.write_text(
        "Year,AMS\n" + "".join(f"{2000 + i},{q}\n" for i, q in enumerate((1, *range(100, 109)))),
        encoding="utf-8",
    )
    destination = tmp_path / "report"
    assert _run(ams, EXAMPLES / "fixed-gev-settings.json", destination) == 0
    rows = _rows(destination)
    assert [int(row["year"]) for row in rows[:2]] == [2009, 2008]
    assert "GEV support: 2 outside; 0 at endpoint" in (destination / "frequency.svg").read_text(encoding="utf-8")
    assert "outside: year=2009" in (destination / "report.html").read_text(encoding="utf-8")


def test_non_deterministic_estimand_rejected() -> None:
    raw_ams = (EXAMPLES / "ordinary-ams.csv").read_bytes()
    result, settings = calculate_from_bytes(raw_ams, (EXAMPLES / "ordinary-settings.json").read_bytes())
    posterior = replace(
        result, values=(replace(result.values[0], meaning=ResultMeaning.POSTERIOR_PREDICTIVE_QUANTILE),)
    )
    with pytest.raises(ValueError, match="deterministic fitted quantiles"):
        render_frequency_files(posterior, settings=settings, raw_ams=raw_ams, report_text="")
