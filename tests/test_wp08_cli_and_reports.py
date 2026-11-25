"""WP08 guidance/report contract checks; numerical values are synthetic, not Flike goldens."""

import csv
import io
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

from ffa_methods.cli import main
from ffa_methods.compatibility import LEGACY_FLIKE_PROFILE
from ffa_methods.contracts import (
    AnalysisResult,
    EvidenceStatus,
    ParameterSummary,
    ProbabilityKind,
    ResultMeaning,
    ResultPoint,
    RunProvenance,
)
from ffa_methods.reporting import render_report_files, report_document, write_report_bundle


def sample_result() -> AnalysisResult:
    """A deliberately synthetic result for testing labels and serialization only."""
    return AnalysisResult(
        provenance=RunProvenance(
            compatibility_profile=LEGACY_FLIKE_PROFILE.identifier,
            method_id="quantile-estimands",
            method_version="synthetic-test/1",
            package_version="0.0.0.dev0",
            evidence_status=EvidenceStatus.CANDIDATE_PAIR,
            input_sha256="a" * 64,
            settings_sha256="b" * 64,
        ),
        parameters=(
            ParameterSummary(
                name="mean natural-log flow", coordinate="lp3.mean_loge_flow", mean=2.5, standard_deviation=0.2
            ),
        ),
        values=(
            ResultPoint(
                meaning=ResultMeaning.QUANTILE_AT_EXPECTED_PARAMETERS,
                value=100.0,
                unit="m3/s",
                input_probability=0.8,
                input_probability_kind=ProbabilityKind.NON_EXCEEDANCE,
            ),
            ResultPoint(
                meaning=ResultMeaning.QUANTILE_MONTE_CARLO_PROBABILITY_LIMIT,
                value=120.0,
                unit="m3/s",
                input_probability=0.2,
                input_probability_kind=ProbabilityKind.ANNUAL_EXCEEDANCE,
                confidence_level=0.9,
                probability_limit_side="upper",
                probability_limit_construction="synthetic_pointwise_posterior_quantile",
            ),
            ResultPoint(
                meaning=ResultMeaning.EXPECTED_NON_EXCEEDANCE_PROBABILITY,
                value=0.9,
                unit="probability",
                input_discharge=115.0,
                input_discharge_unit="m3/s",
            ),
            ResultPoint(
                meaning=ResultMeaning.LOG10_QUANTILE_POSTERIOR_MOMENT,
                value=2.2,
                unit="log10(m3/s)",
                log10_moment_kind="mean",
                input_probability=0.8,
                input_probability_kind=ProbabilityKind.NON_EXCEEDANCE,
            ),
        ),
        warnings=("Synthetic input: do not use for engineering design.",),
    )


def test_explain_lists_registered_methods(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["explain"]) == 0
    output = capsys.readouterr().out
    assert "lp3" in output
    assert "GUIDANCE ONLY" in output
    assert "legacy-qualified" in output


def test_explain_method_and_machine_readable_guidance(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["explain", "lp3"]) == 0
    assert "ARR3" in capsys.readouterr().out
    assert main(["explain", "lp3", "--format", "json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["schema_version"] == "ffa-method-guidance/1"
    assert [method["method_id"] for method in data["methods"]] == ["lp3"]
    assert data["methods"][0]["qualification_status"] == "not_started"


def test_explain_bad_method_is_usage_error(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as error:
        main(["explain", "missing-method"])
    assert error.value.code == 2
    assert "unknown method" in capsys.readouterr().err


def test_bundled_markdown_pages_and_missing_page(capsys: pytest.CaptureFixture[str]) -> None:
    for method_id in (
        "mgb-low-flow-screen",
        "legacy-zero-threshold",
        "censored-historical-likelihood",
        "quantile-estimands",
    ):
        assert main(["explain", method_id, "--format", "markdown"]) == 0
        page = capsys.readouterr().out
        assert page.startswith("# ")
        assert f"Method ID:** `{method_id}`" in page
        assert "not" in page.lower()
    with pytest.raises(SystemExit) as missing:
        main(["explain", "lp3", "--format", "markdown"])
    assert missing.value.code == 2
    assert "no bundled technical page" in capsys.readouterr().err


def test_cli_still_rejects_fit(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as error:
        main(["fit", "--distribution", "lp3"])
    assert error.value.code == 2
    assert "error:" in capsys.readouterr().err


def test_result_outputs_agree_without_changing_estimands() -> None:
    files = render_report_files(sample_result(), settings={"distribution": "lp3", "seed": 42})
    assert set(files) == {"results.json", "results.csv", "report.txt"}
    data = json.loads(files["results.json"])
    assert data["schema_version"] == "ffa-analysis-report/1"
    assert "### ARR3" in data["method_guidance"]["reference_entries"]["ARR3"]
    assert data["method_qualification"] == "not_established_by_formatter"
    assert data["legacy_comparison_status"] == "not_established_by_formatter"
    assert data["provenance"]["reference_evidence_status"] == "candidate_pair"
    assert data["provenance"]["settings_sha256"] == "b" * 64
    assert data["parameters"][0]["coordinate"] == "lp3.mean_loge_flow"
    assert data["results"][0]["value"] == 100
    assert data["results"][0]["input_non_exceedance_probability"] == 0.8
    assert math.isclose(float(data["results"][0]["input_annual_exceedance_probability"]), 0.2, rel_tol=1e-12)
    assert data["results"][1]["input_annual_exceedance_probability"] == 0.2
    assert data["results"][1]["input_non_exceedance_probability"] == 0.8
    assert data["results"][1]["reported_probability_level"] == 0.9
    assert data["results"][1]["probability_limit_side"] == "upper"
    assert data["results"][1]["probability_limit_construction"] == "synthetic_pointwise_posterior_quantile"
    assert data["results"][2]["input_discharge_unit"] == "m3/s"
    assert data["results"][3]["log10_moment_kind"] == "mean"
    assert data["results"][2]["meaning"] == "expected_non_exceedance_probability"
    assert "not the expected quantile" in data["results"][0]["interpretation"]
    assert "log10" in data["results"][3]["interpretation"]
    csv_rows = list(csv.DictReader(io.StringIO(files["results.csv"])))
    assert len(csv_rows) == 4
    assert csv_rows[1]["input_annual_exceedance_probability"] == "0.2"
    assert csv_rows[1]["probability_limit_side"] == "upper"
    assert csv_rows[2]["input_discharge_unit"] == "m3/s"
    assert csv_rows[3]["log10_moment_kind"] == "mean"
    assert csv_rows[2]["value"] == "0.9"
    assert csv_rows[0]["method_qualification"] == "not_established_by_formatter"
    assert csv_rows[0]["legacy_comparison_status"] == "not_established_by_formatter"
    assert "AEP=0.2" in files["report.txt"]
    assert "input discharge=115.0 m3/s" in files["report.txt"]
    assert "probability limit: upper" in files["report.txt"]
    assert "log10 moment=mean" in files["report.txt"]
    assert "UNQUALIFIED" in files["report.txt"]
    assert "PARAMETERS (reported source coordinates)" in files["report.txt"]
    assert "mean natural-log flow" in files["report.txt"]
    assert "Synthetic input" in files["report.txt"]
    assert "SOURCE REFERENCES (bundled source-register entries)" in files["report.txt"]


def test_bundle_is_staged_and_existing_directory_never_overwritten(tmp_path: Path) -> None:
    output = tmp_path / "report"
    names = write_report_bundle(sample_result(), output, settings={"distribution": "lp3"})
    assert [p.name for p in names] == ["results.json", "results.csv", "report.txt"]
    for path in names:
        assert path.is_file()
    before = {p.name: p.read_bytes() for p in names}
    with pytest.raises(FileExistsError):
        write_report_bundle(sample_result(), output, settings={"distribution": "lp3"})
    assert {p.name: p.read_bytes() for p in names} == before
    assert not any(p.name.startswith(".report-") for p in tmp_path.iterdir())


def test_unqualified_warning_when_replay_details_are_absent() -> None:
    base = sample_result()
    provenance = replace(base.provenance, input_sha256=None, settings_sha256=None)
    doc = report_document(replace(base, provenance=provenance), settings={})
    warnings = cast(list[str], doc["warnings"])
    assert len(warnings) >= 4  # Original warning plus missing information notices
    assert doc["method_qualification"] == "not_established_by_formatter"


@pytest.mark.parametrize(
    ("point", "message"),
    [
        (ResultPoint(meaning=ResultMeaning.QUANTILE_AT_EXPECTED_PARAMETERS, value=float("inf"), unit="m3/s"), "finite"),
        (
            ResultPoint(
                meaning=ResultMeaning.QUANTILE_AT_EXPECTED_PARAMETERS, value=1, unit="m3/s", input_probability=0.9
            ),
            "together",
        ),
        (
            ResultPoint(meaning=ResultMeaning.EXPECTED_NON_EXCEEDANCE_PROBABILITY, value=1.1, unit="probability"),
            "within",
        ),
        (
            ResultPoint(
                meaning=ResultMeaning.QUANTILE_AT_EXPECTED_PARAMETERS,
                value=1,
                unit="m3/s",
                input_probability=2,
                input_probability_kind=ProbabilityKind.ANNUAL_EXCEEDANCE,
            ),
            "within",
        ),
    ],
)
def test_invalid_numerical_records_are_rejected(point: ResultPoint, message: str) -> None:
    result = replace(sample_result(), values=(point,))
    with pytest.raises(ValueError, match=message):
        report_document(result, settings={"distribution": "lp3"})


def test_missing_or_misapplied_report_semantics_fail_closed() -> None:
    result = sample_result()
    quantile, limit, probability, moment = result.values
    cases: tuple[tuple[ResultPoint, str], ...] = (
        (replace(quantile, input_probability=None, input_probability_kind=None), "requires an input probability"),
        (replace(quantile, confidence_level=0.9), "probability limit metadata"),
        (replace(quantile, input_discharge=3.0, input_discharge_unit="m3/s"), "cannot also declare"),
        (replace(probability, input_discharge_unit=None), "requires input discharge"),
        (replace(probability, input_discharge=-1.0), "nonnegative"),
        (replace(probability, unit="m3/s"), "unit 'probability'"),
        (
            replace(probability, input_probability=0.1, input_probability_kind=ProbabilityKind.ANNUAL_EXCEEDANCE),
            "must not also",
        ),
        (replace(limit, confidence_level=None), "requires level"),
        (replace(limit, probability_limit_side=None), "requires level"),
        (replace(limit, probability_limit_construction=None), "requires level"),
        (replace(moment, log10_moment_kind=None), "requires its"),
        (replace(moment, log10_moment_kind="variance", value=-1.0), "nonnegative"),
        (replace(moment, unit="m3/s"), "log10 discharge unit"),
        (replace(moment, probability_limit_side="upper"), "invalid for a non-limit"),
    )
    for point, message in cases:
        with pytest.raises(ValueError, match=message):
            report_document(replace(result, values=(point,)), settings={})


def test_negative_modelled_quantile_warns_without_clipping() -> None:
    result = sample_result()
    point = replace(result.values[0], value=-1.0)
    doc = report_document(replace(result, values=(point,)), settings={"synthetic_test": True})
    records = cast(list[dict[str, object]], doc["results"])
    warnings = cast(list[str], doc["warnings"])
    assert records[0]["value"] == -1.0
    assert any("not a physically possible discharge" in warning for warning in warnings)
    assert doc["method_qualification"] == "not_established_by_formatter"


def test_unknown_methods_and_unrecognised_settings_fail_closed() -> None:
    result = sample_result()
    with pytest.raises(LookupError, match="unknown method"):
        report_document(
            replace(result, provenance=replace(result.provenance, method_id="not-in-register")), settings={}
        )
    with pytest.raises(ValueError, match="finite"):
        report_document(result, settings={"bad": float("nan")})
    with pytest.raises(ValueError, match="control characters"):
        report_document(result, settings={"line\nbreak": "invalid"})
    with pytest.raises(ValueError, match="SHA-256"):
        report_document(replace(result, provenance=replace(result.provenance, input_sha256="bad")), settings={})
    with pytest.raises(ValueError, match="no numerical result"):
        report_document(replace(result, values=()), settings={})


def test_cross_profile_cannot_be_misreported_as_legacy() -> None:
    result = sample_result()
    result = replace(result, provenance=replace(result.provenance, compatibility_profile="wrong-build"))
    with pytest.raises(ValueError, match="compatibility profile"):
        report_document(result, settings={})


def test_reference_citations_available_offline(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["references", "ARR3"]) == 0
    output = capsys.readouterr().out
    assert "### ARR3" in output
    assert "ARR1" not in output
    assert main(["references"]) == 0
    assert "### KUCZERA1999" in capsys.readouterr().out
    with pytest.raises(SystemExit) as error:
        main(["references", "DOES_NOT_EXIST"])
    assert error.value.code == 2
    assert "reference key" in capsys.readouterr().err


def test_mismatched_installed_package_version_fails_closed() -> None:
    result = sample_result()
    with pytest.raises(ValueError, match="installed package"):
        report_document(replace(result, provenance=replace(result.provenance, package_version="2.0")), settings={})
