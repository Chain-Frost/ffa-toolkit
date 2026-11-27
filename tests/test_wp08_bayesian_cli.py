"""Independent Bayesian CLI orchestration, provenance and failure checks."""

import json
from hashlib import sha256
from math import exp, isclose, sqrt
from pathlib import Path
from statistics import NormalDist
from typing import cast

import pytest

from ffa_methods.cli import main
from ffa_methods.contracts import ResultMeaning
from ffa_methods.deterministic_workflow import calculate_from_bytes
from ffa_methods.importance_precision import PrecisionState
from ffa_methods.reporting import report_document

EXAMPLES = Path(__file__).resolve().parents[1] / "docs/examples"


def _example(kind: str) -> tuple[bytes, bytes]:
    return (
        (EXAMPLES / f"bayesian-{kind}-ams.csv").read_bytes(),
        (EXAMPLES / f"bayesian-{kind}-settings.json").read_bytes(),
    )


def _settings(kind: str) -> dict[str, object]:
    return cast(dict[str, object], json.loads(_example(kind)[1]))


def _encode(settings: dict[str, object]) -> bytes:
    return json.dumps(settings, allow_nan=False).encode()


def _fit_args(ams: Path, settings: Path, output: Path) -> list[str]:
    return ["fit", "--ams", str(ams), "--settings", str(settings), "--output", str(output)]


def test_known_scale_bayesian_file_to_reports_and_precision_replay(tmp_path: Path) -> None:
    raw, settings = _example("fixed")
    # Preserve unusual but valid original bytes, including CRLF and BOM.
    raw = b"\xef\xbb\xbf" + raw.replace(b"\n", b"\r\n")
    ams, config = tmp_path / "ams.csv", tmp_path / "settings.json"
    ams.write_bytes(raw)
    config.write_bytes(settings)
    first, second = tmp_path / "first", tmp_path / "second"
    assert main(_fit_args(ams, config, first)) == 0
    assert main(_fit_args(ams, config, second)) == 0
    assert {p.name for p in first.iterdir()} == {
        "results.json", "results.csv", "report.txt", "source-ams.csv", "source-settings.json",
    }
    for p in first.iterdir():
        assert p.read_bytes() == (second / p.name).read_bytes()
    assert (first / "source-ams.csv").read_bytes() == raw
    assert (first / "source-settings.json").read_bytes() == settings
    result, report_settings = calculate_from_bytes(raw, settings)
    assert result.provenance.input_sha256 == sha256(raw).hexdigest()
    assert result.provenance.settings_sha256 == sha256(settings).hexdigest()
    # Completing the square: log flows 1,2,3, known SD=1 and N(0,1)
    # prior give N(1.5,1/4); predictive log flow has variance 5/4.
    z = NormalDist().inv_cdf(0.9)
    reference_aep = 1 - NormalDist(1.5, sqrt(1.25)).cdf(2.0)
    means = {point.meaning: point.value for point in result.values}
    assert isclose(result.parameters[0].mean, 1.5, abs_tol=0.16)
    assert isclose(means[ResultMeaning.QUANTILE_AT_EXPECTED_PARAMETERS], exp(1.5 + z), abs_tol=5)
    assert isclose(means[ResultMeaning.POSTERIOR_MEAN_QUANTILE], exp(1.5 + z + 0.125), abs_tol=6)
    assert isclose(means[ResultMeaning.POSTERIOR_PREDICTIVE_QUANTILE], exp(1.5 + sqrt(1.25) * z), abs_tol=9)
    assert means[ResultMeaning.POSTERIOR_MEAN_QUANTILE] != means[ResultMeaning.QUANTILE_AT_EXPECTED_PARAMETERS]
    assert isclose(means[ResultMeaning.EXPECTED_ANNUAL_EXCEEDANCE_PROBABILITY], reference_aep, abs_tol=0.09)
    assert isclose(means[ResultMeaning.RETURN_PERIOD_FROM_EXPECTED_AEP], 1 / means[ResultMeaning.EXPECTED_ANNUAL_EXCEEDANCE_PROBABILITY])
    audit = result.precision_audit
    assert audit is not None
    assert audit.admission is not None
    assert audit.state == PrecisionState.CONDITIONAL_PASS
    assert audit.completed_count == 3
    assert tuple(a.seed for a in audit.attempts) == (51, 52, 53)
    document = report_document(result, settings=report_settings)
    assert document["method_qualification"] == "not_established_by_formatter"
    rows = cast(list[dict[str, object]], document["results"])
    aep = next(row for row in rows if row["meaning"] == ResultMeaning.EXPECTED_ANNUAL_EXCEEDANCE_PROBABILITY)
    assert aep["precision_state"] == "conditional_pass"
    assert all(row["precision_state"] == "not_assessed" for row in rows if row is not aep)
    assert "precision_audit" in document
    assert "precision_state" in (first / "results.csv").read_text(encoding="utf-8").splitlines()[0]


def test_estimated_scale_censored_reference_and_undefined_mean() -> None:
    raw, settings = _example("censored")
    result, report_settings = calculate_from_bytes(raw, settings)
    # Independent midpoint quadrature, 81x81 and 161x161 grids over nine
    # prior SDs in each coordinate; references agree to better than 1e-12.
    # Six reference asymptotic standard errors, rounded outward; the
    # numerical test tolerance is not a general output precision admission.
    assert isclose(result.parameters[0].mean, 0.003598304886871672, abs_tol=0.032)
    assert isclose(result.parameters[1].mean, -0.01338564667751898, abs_tol=0.012)
    aep = next(point for point in result.values if point.meaning == ResultMeaning.EXPECTED_ANNUAL_EXCEEDANCE_PROBABILITY)
    assert isclose(aep.value, 0.3686203803796388, abs_tol=0.012)
    assert not any(point.meaning == ResultMeaning.POSTERIOR_MEAN_QUANTILE for point in result.values)
    assert any(point.meaning == ResultMeaning.POSTERIOR_PREDICTIVE_QUANTILE for point in result.values)
    assert any("UNDEFINED" in warning and "diverges" in warning for warning in result.warnings)
    assert result.precision_audit is None
    assert report_settings["precision_status"] == "not_assessed"
    assert result == calculate_from_bytes(raw, settings)[0]


@pytest.mark.parametrize(
    "override",
    [
        {"source_classification": "native"},
        {"fixed_log_sd": True},
        {"reference_discharge": 0},
        {"source_format": "legacy_fld"},
        {"unknown": True},
        {"precision": {"mode": "pass"}},
        {"outputs": {"non_exceedance_probabilities": [0.9, 0.9], "discharges": [], "interval_mass": 0.9}},
    ],
)
def test_reject_unsupported_bayesian_settings(override: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        calculate_from_bytes(_example("fixed")[0], _encode({**_settings("fixed"), **override}))


@pytest.mark.parametrize("data", [b"Year,AMS\n2000,0\n", b"Year,AMS\n2000,\n", b"Year,AMS\n2000,1e-500\n"])
def test_exact_csv_never_coerces_missing_zero_or_underflow(data: bytes) -> None:
    with pytest.raises(ValueError):
        calculate_from_bytes(data, _example("fixed")[1])


@pytest.mark.parametrize(
    "row",
    [
        b"2000,left_censored,,,1\n",
        b"2004,interval_censored,,3,2\n",
        b"2004,at_or_below_zero_threshold,0,,\n",
        b"2004,historical,3,,\n",
        b"2004,missing,3,,\n",
        b"2004,exact,,,\n",
    ],
)
def test_classified_csv_rejects_overlap_and_unadmitted_records(row: bytes) -> None:
    raw, settings = _example("censored")
    with pytest.raises(ValueError):
        calculate_from_bytes(raw + row, settings)


def test_explicit_audit_budget_duplicate_keys_and_output_collision(tmp_path: Path) -> None:
    raw, settings = _example("fixed")
    config = _settings("fixed")
    precision = cast(dict[str, object], config["precision"])
    precision["maximum_total_draws"] = 7199
    with pytest.raises(ValueError, match="budget"):
        calculate_from_bytes(raw, _encode(config))
    duplicate = settings.replace(b'"gamma": 2.05,', b'"gamma": 2.05, "gamma": 3.0,')
    assert duplicate != settings
    with pytest.raises(ValueError, match="duplicate settings JSON key"):
        calculate_from_bytes(raw, duplicate)
    ams, options, output = tmp_path / "ams.csv", tmp_path / "settings.json", tmp_path / "existing"
    ams.write_bytes(raw)
    options.write_bytes(settings)
    output.mkdir()
    sentinel = output / "sentinel.txt"
    sentinel.write_text("preserve", encoding="utf-8")
    with pytest.raises(SystemExit) as error:
        main(_fit_args(ams, options, output))
    assert error.value.code == 2
    assert list(output.iterdir()) == [sentinel]
    assert sentinel.read_text(encoding="utf-8") == "preserve"
