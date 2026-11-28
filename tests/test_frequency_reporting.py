"""Verify the opt-in presentation against independent plotting coordinates and report values."""

import csv
import io
import json
from dataclasses import replace
from math import ceil, isclose, log10, nextafter
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


def test_frequency_report_refuses_a_refused_bootstrap(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ams = EXAMPLES / "ordinary-ams.csv"
    config = tmp_path / "settings.json"
    settings = cast(dict[str, object], json.loads((EXAMPLES / "ordinary-settings.json").read_text(encoding="utf-8")))
    settings["bootstrap"] = {
        "replications": 4,
        "seed": 42,
        "confidence": 0.8,
        "maximum_failure_fraction": 0.0,
        "endpoint_tolerance": 1.0,
    }
    config.write_text(json.dumps(settings), encoding="utf-8")
    plain, plotted = tmp_path / "plain", tmp_path / "plotted"
    assert _run(ams, config, plain, plot=False) == 2
    result = json.loads((plain / "results.json").read_text(encoding="utf-8"))
    assert result["settings"]["bootstrap_status"] == "refused"
    assert result["settings"]["bootstrap_attempted"] == 4
    with pytest.raises(SystemExit) as exc:
        _run(ams, config, plotted)
    assert exc.value.code == 2
    assert "without a bootstrap request" in capsys.readouterr().err
    assert not plotted.exists()


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


def test_intermediate_aep_ticks_and_coincident_marker_visibility() -> None:
    raw_ams = (EXAMPLES / "ordinary-ams.csv").read_bytes()
    result, settings = calculate_from_bytes(raw_ams, (EXAMPLES / "ordinary-settings.json").read_bytes())
    # Synthetic coincidence checks the renderer, not the calculated model.
    coincident = replace(result, values=(replace(result.values[0], value=2.0), *result.values[1:]))
    files = render_frequency_files(coincident, settings=settings, raw_ams=raw_ams, report_text="")
    rows = list(csv.DictReader(io.StringIO(files["frequency-coordinates.csv"])))
    observed, fitted = rows[1], rows[3]
    assert float(observed["AEP"]) == float(fitted["AEP"]) == 0.5
    assert observed["svg_x"] == fitted["svg_x"]
    assert observed["svg_y"] == fitted["svg_y"]
    coords = f'cx="{observed["svg_x"]}" cy="{observed["svg_y"]}"'
    svg = files["frequency.svg"]
    assert ">50%</text>" in svg
    assert ">20%</text>" in svg
    assert f'<circle data-series="observed" {coords} r="4" fill="#202c39">' in svg
    assert f'<circle data-series="fitted" {coords} r="5.5" fill="none" stroke="#1767a0"' in svg


def test_ranks_use_source_precision() -> None:
    raw_ams = b"Year,AMS\n2000,1\n2001,1.00000000000000000000000000001\n2002,4\n"
    result, settings = calculate_from_bytes(raw_ams, (EXAMPLES / "ordinary-settings.json").read_bytes())
    files = render_frequency_files(result, settings=settings, raw_ams=raw_ams, report_text="")
    observed = list(csv.DictReader(io.StringIO(files["frequency-coordinates.csv"])))[:3]
    assert [row["year"] for row in observed] == ["2002", "2001", "2000"]
    for row, expected in zip(observed, (3 / 16, 1 / 2, 13 / 16), strict=True):
        assert isclose(float(row["AEP"]), expected, rel_tol=1e-15)


def test_frequency_output_collision_preserves_existing_directory(tmp_path: Path) -> None:
    destination = tmp_path / "occupied"
    destination.mkdir()
    with pytest.raises(SystemExit) as exc:
        _run(EXAMPLES / "ordinary-ams.csv", EXAMPLES / "ordinary-settings.json", destination)
    assert exc.value.code == 2
    assert list(tmp_path.iterdir()) == [destination]
    assert not list(destination.iterdir())


@pytest.mark.parametrize("value", [5e-324, 1.7976931348623157e308])
def test_extreme_discharge_and_rare_aep(value: float) -> None:
    source = (EXAMPLES / "ordinary-ams.csv").read_bytes()
    result, settings = calculate_from_bytes(source, (EXAMPLES / "ordinary-settings.json").read_bytes())
    altered = replace(
        result,
        values=(replace(result.values[0], value=-value, input_probability=nextafter(1.0, 0.0)),),
    )
    raw_ams = f"Year,AMS\n2000,{value}\n2001,{value}\n2002,{value}\n".encode()
    files = render_frequency_files(altered, settings=settings, raw_ams=raw_ams, report_text="")
    fitted = list(csv.DictReader(io.StringIO(files["frequency-coordinates.csv"])))[-1]
    assert float(fitted["AEP"]) == 2**-53
    assert float(fitted["discharge"]) == -value
    assert float(fitted["svg_y"]) == 445
    svg = files["frequency.svg"]
    assert ">1e-14%</text>" in svg
    y_ticks = [line.split(">")[1].split("<")[0] for line in svg.splitlines() if 'x="100"' in line]
    assert len(set(y_ticks)) == 6
    assert "inf</text>" not in svg
