"""WP08: independent complete-AMS fixed-H GEV file -> estimator -> report tests.

The example is synthetic mathematical data, not native Flike regression.
"""

import csv
import io
import json
from dataclasses import replace
from hashlib import sha256
from math import exp, isclose, log
from pathlib import Path
from typing import cast

import pytest

from ffa_methods.cli import main
from ffa_methods.contracts import ParameterSummary, RunDiagnostic
from ffa_methods.deterministic_workflow import calculate_fixed_gev_from_files, load_fixed_gev_settings
from ffa_methods.reporting import report_document


def _example(name: str) -> bytes:
    return (Path(__file__).resolve().parents[1] / "docs/examples" / name).read_bytes()


def _settings() -> dict[str, object]:
    return cast(dict[str, object], json.loads(_example("fixed-gev-settings.json")))


def _fixture_files(
    tmp_path: Path, *, settings: dict[str, object] | None = None, csv_bytes: bytes | None = None
) -> tuple[Path, Path]:
    ams = tmp_path / "ams.csv"
    config = tmp_path / "settings.json"
    ams.write_bytes(_example("fixed-gev-ams.csv") if csv_bytes is None else csv_bytes)
    config.write_text(json.dumps(_settings() if settings is None else settings), encoding="utf-8")
    return ams, config


def _fit_args(ams: Path, config: Path, output: Path) -> list[str]:
    return ["fit", "--ams", str(ams), "--settings", str(config), "--output", str(output)]


def _number(value: object) -> float:
    if not isinstance(value, int | float):
        msg = "expected JSON numeric value"
        raise TypeError(msg)
    return float(value)


def test_fixed_h_file_to_fit_report_has_independent_quantile_and_source_custody(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ams, settings = _fixture_files(tmp_path)
    report = tmp_path / "result"
    assert main(_fit_args(ams, settings, report)) == 0
    assert "not native Flike-qualified" in capsys.readouterr().out
    files = {path.name: path for path in report.iterdir()}
    assert set(files) == {"results.json", "results.csv", "report.txt"}
    doc = cast(dict[str, object], json.loads(files["results.json"].read_text(encoding="utf-8")))
    provenance = cast(dict[str, object], doc["provenance"])
    assert provenance["method_id"] == "lh-moments-gev"
    assert provenance["method_version"] == "arr-wang-gev-fixed-h/1"
    assert provenance["input_sha256"] == sha256(ams.read_bytes()).hexdigest()
    assert provenance["settings_sha256"] == sha256(settings.read_bytes()).hexdigest()
    assert provenance["reference_evidence_status"] == "observed_artifact"
    assert doc["method_qualification"] == "not_established_by_formatter"
    assert doc["legacy_comparison_status"] == "not_established_by_formatter"
    config = cast(dict[str, object], doc["settings"])
    assert config["lh_shift"] == 0
    assert config["year_definition"] == "synthetic_calendar_year"
    assert config["non_exceedance_probabilities"] == "0.5,0.8,0.9,0.99"

    parameters = cast(list[dict[str, object]], doc["parameters"])
    assert [item["summary_kind"] for item in parameters] == ["point_estimate"] * 3
    assert all(item["mean"] is None and item["standard_deviation"] is None for item in parameters)
    assert [item["coordinate"] for item in parameters] == [
        "arr.gev.location",
        "arr.gev.scale",
        "arr.gev.shape_k",
    ]
    u, scale, shape = (_number(item["point_estimate"]) for item in parameters)
    assert scale > 0
    assert shape > -1
    rows = cast(list[dict[str, object]], doc["results"])
    assert [item["meaning"] for item in rows] == ["deterministic_fitted_quantile"] * 4
    assert [item["input_probability_kind"] for item in rows] == ["non_exceedance"] * 4
    for point, expected_f in zip(rows, (0.5, 0.8, 0.9, 0.99), strict=True):
        f = _number(point["input_non_exceedance_probability"])
        aep = _number(point["input_annual_exceedance_probability"])
        assert f == pytest.approx(expected_f)  # pyright: ignore[reportUnknownMemberType]
        assert isclose(f + aep, 1.0, abs_tol=1e-15)
        assert point["unit"] == "m3/s"
        # Independent formula evaluated from the *reported* point estimates.
        reduced = -log(f)
        expected_q = (
            u - scale * log(reduced) if abs(shape) < 1e-12 else u + scale * (1 - exp(shape * log(reduced))) / shape
        )
        assert isclose(_number(point["value"]), expected_q, rel_tol=5e-12, abs_tol=1e-12)

    diagnostics = cast(list[dict[str, object]], doc["diagnostics"])
    assert diagnostics[0]["name"] == "exact_annual_maxima_count"
    assert diagnostics[0]["value"] == 10.0
    assert any(item["name"] == "raw_lh_fourth_to_second_ratio_residual_unstandardized" for item in diagnostics)
    csv_rows = list(csv.DictReader(io.StringIO(files["results.csv"].read_text(encoding="utf-8"))))
    assert len(csv_rows) == 4
    assert csv_rows[0]["meaning"] == "deterministic_fitted_quantile"
    assert "point_estimate=" in files["report.txt"].read_text(encoding="utf-8")
    assert "COMPUTED DIAGNOSTICS" in files["report.txt"].read_text(encoding="utf-8")
    assert "not a verified legacy Flike comparison" in files["report.txt"].read_text(encoding="utf-8")

    before = {name: path.read_bytes() for name, path in files.items()}
    with pytest.raises(SystemExit) as exc:
        main(_fit_args(ams, settings, report))
    assert exc.value.code == 2
    assert {name: path.read_bytes() for name, path in files.items()} == before
    assert not any(p.name.startswith(".result-") for p in tmp_path.iterdir())


@pytest.mark.parametrize("shift", [0, 1, 2, 3, 4])
def test_explicit_each_fixed_shift_is_mathematically_executable(tmp_path: Path, shift: int) -> None:
    settings = _settings()
    settings["lh_shift"] = shift
    ams, config = _fixture_files(tmp_path, settings=settings)
    result, details = calculate_fixed_gev_from_files(ams, config)
    assert len(result.values) == 4
    assert result.provenance.method_id == "lh-moments-gev"
    assert details["lh_shift"] == shift
    assert all(item.summary_kind == "point_estimate" for item in result.parameters)
    assert any(item.name == "sample_lh_moment_1" for item in result.diagnostics)
    assert all(point.meaning.value == "deterministic_fitted_quantile" for point in result.values)


@pytest.mark.parametrize(
    ("csv_data", "reason"),
    [
        (b"Year,AMS\n2000,5\n2001,\n2002,7\n2003,8\n", "only complete exact"),
        (b"Year,AMS\n2000,5\n2001,0\n2002,7\n2003,8\n", "only complete exact"),
        (b"Year,AMS\n2000,5\n2001,-1\n2002,7\n2003,8\n", "Negative AMS"),
        (b"Year,AMS\n2000,5\n2000,6\n2001,7\n2002,8\n", "Duplicate"),
        (b"Year,AMS\n2000,5\n2001,6\n", "at least"),
        (b"Year,AMS\n2000,5\n2001,5\n2002,5\n2003,5\n", "nonconstant"),
        (b"Year,Q\n2000,5\n2001,6\n2002,7\n2003,8\n", "Year,AMS"),
        (b"FLIKE data file 310\n", "Year,AMS"),
        (b"Year,AMS\n2000,5\n2001,6\n2002,1e9999\n2003,8\n", "nonfinite/underflowed"),
    ],
)
def test_unadmitted_ams_fails_without_output(
    tmp_path: Path, csv_data: bytes, reason: str, capsys: pytest.CaptureFixture[str]
) -> None:
    ams, settings = _fixture_files(tmp_path, csv_bytes=csv_data)
    output = tmp_path / "report"
    with pytest.raises(SystemExit) as exc:
        main(_fit_args(ams, settings, output))
    assert exc.value.code == 2
    assert reason in capsys.readouterr().err
    assert not output.exists()


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("lh_shift", 5, "lh_shift"),
        ("lh_shift", True, "lh_shift"),
        ("estimator", "lp3", "schema_version"),
        ("schema_version", "not-that-schema", "schema_version"),
        ("source_classification", "unspecified", "caller must explicitly assert"),
        ("non_exceedance_probabilities", [0.2, 0.2], "strictly increasing"),
        ("non_exceedance_probabilities", [0.9, 0.8], "strictly increasing"),
        ("non_exceedance_probabilities", [0.0], "strictly between"),
        ("non_exceedance_probabilities", [1.0], "strictly between"),
        ("non_exceedance_probabilities", [True], "not boolean"),
        ("non_exceedance_probabilities", [], "1 to 200"),
        ("discharge_unit", "probability", "physical flow unit"),
        ("year_definition", "", "nonempty string"),
    ],
)
def test_invalid_explicit_settings_fail_without_output(
    tmp_path: Path, field: str, value: object, reason: str, capsys: pytest.CaptureFixture[str]
) -> None:
    settings = _settings()
    settings[field] = value
    ams, config = _fixture_files(tmp_path, settings=settings)
    destination = tmp_path / "report"
    with pytest.raises(SystemExit) as exc:
        main(_fit_args(ams, config, destination))
    assert exc.value.code == 2
    assert reason in capsys.readouterr().err
    assert not destination.exists()


def test_json_duplicate_unknown_and_nonstandard_constants_are_rejected() -> None:
    valid = _example("fixed-gev-settings.json")
    with pytest.raises(ValueError, match="duplicate"):
        load_fixed_gev_settings(valid.replace(b'"lh_shift": 0', b'"lh_shift": 1, "lh_shift": 0'))
    with pytest.raises(ValueError, match="unknown"):
        load_fixed_gev_settings(valid.replace(b'"lh_shift": 0', b'"unknown_option": 1, "lh_shift": 0'))
    with pytest.raises(ValueError, match="nonstandard"):
        load_fixed_gev_settings(valid.replace(b'"lh_shift": 0', b'"lh_shift": NaN'))


def test_invalid_io_and_missing_paths_have_stable_error_code(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ams, config = _fixture_files(tmp_path)
    absent = tmp_path / "not-existing.csv"
    assert main(_fit_args(absent, config, tmp_path / "result")) == 1
    assert "I/O failure" in capsys.readouterr().err
    assert main(_fit_args(ams, config, tmp_path / "missing-parent" / "report")) == 1
    assert "I/O failure" in capsys.readouterr().err


def test_diagnostic_finiteness_and_point_summary_kind_are_validated(tmp_path: Path) -> None:
    ams, config = _fixture_files(tmp_path)
    result, settings = calculate_fixed_gev_from_files(ams, config)
    invalid_diagnostic = replace(result, diagnostics=(RunDiagnostic(name="bad", value=float("nan"), unit="count"),))
    with pytest.raises(ValueError, match="finite"):
        report_document(invalid_diagnostic, settings=settings)
    original = result.parameters[0]
    bad_parameter = replace(
        result,
        parameters=(
            ParameterSummary(
                name=original.name,
                coordinate=original.coordinate,
                mean=original.mean,
                standard_deviation=0.1,
                summary_kind="point_estimate",
            ),
        ),
    )
    with pytest.raises(ValueError, match="posterior standard deviation"):
        report_document(bad_parameter, settings=settings)
