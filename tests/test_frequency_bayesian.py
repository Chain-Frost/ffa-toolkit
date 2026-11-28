"""Exact-record Bayesian static presentation: no new inference or precision claim."""

import csv
import json
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

from ffa_methods.cli import main
from ffa_methods.contracts import ResultMeaning
from ffa_methods.deterministic_workflow import calculate_from_bytes
from ffa_methods.frequency_quantiles import BAYESIAN_SERIES, POSTERIOR_LIMIT, bayesian_quantile_rows
from ffa_methods.reporting import report_document

EXAMPLES = Path(__file__).resolve().parents[1] / "docs" / "examples"


def _settings() -> bytes:
    return (EXAMPLES / "bayesian-fixed-settings.json").read_bytes()


def _ams() -> bytes:
    return (EXAMPLES / "bayesian-fixed-ams.csv").read_bytes()


def _result_rows(document: dict[str, object]) -> list[dict[str, object]]:
    return cast(list[dict[str, object]], document["results"])


def test_bayesian_quantile_plot_keeps_report_identity(tmp_path: Path) -> None:
    ams = EXAMPLES / "bayesian-fixed-ams.csv"
    settings = EXAMPLES / "bayesian-fixed-settings.json"
    plain, plotted = tmp_path / "plain", tmp_path / "plotted"
    args = ["fit", "--ams", str(ams), "--settings", str(settings), "--output"]
    assert main([*args, str(plain)]) == 0
    assert main([*args, str(plotted), "--frequency-report"]) == 0
    for path in plain.iterdir():
        assert path.read_bytes() == (plotted / path.name).read_bytes()
    document = cast(dict[str, object], json.loads((plotted / "results.json").read_text(encoding="utf-8")))
    canonical = _result_rows(document)
    with (plotted / "frequency-coordinates.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 9
    assert [float(row["AEP"]) for row in rows[:3]] == pytest.approx([3 / 16, 1 / 2, 13 / 16])
    assert [int(row["year"]) for row in rows[:3]] == [2003, 2002, 2001]
    svg = (plotted / "frequency.svg").read_text(encoding="utf-8")
    html = (plotted / "report.html").read_text(encoding="utf-8")
    model = rows[3:]
    assert {row["meaning"] for row in model} == {meaning.value for meaning in (*BAYESIAN_SERIES, POSTERIOR_LIMIT)}
    for row in model:
        source = canonical[int(row["result_row"]) - 1]
        assert row["meaning"] == source["meaning"]
        assert float(row["discharge"]) == source["value"]
        assert float(row["F"]) == source["input_non_exceedance_probability"]
        assert float(row["AEP"]) == source["input_annual_exceedance_probability"]
        assert row["unit"] == source["unit"]
        assert row["precision_state"] == source["precision_state"] == "not_assessed"
        assert f'data-result-row="{row["result_row"]}"' in svg
        assert f'cx="{row["svg_x"]}" cy="{row["svg_y"]}"' in svg
        if row["meaning"] == POSTERIOR_LIMIT.value:
            assert float(row["reported_probability_level"]) == source["reported_probability_level"] == 0.8
            assert row["probability_limit_side"] == source["probability_limit_side"]
            assert row["probability_limit_construction"] == source["probability_limit_construction"]
    assert svg.strip() in html
    assert 'data-interval="posterior_quantile"' in svg
    assert "not predictive intervals" in html


def test_bayesian_series_contract_rejects_bad_rows() -> None:
    result, settings = calculate_from_bytes(_ams(), _settings())
    canonical = _result_rows(report_document(result, settings=settings))
    rows = bayesian_quantile_rows(result, settings=settings, unit="m3/s", numerical_rows=canonical)
    assert len(rows) == 6
    altered = [dict(row) for row in canonical]
    altered[rows[0].index - 1]["value"] = -100.0
    with pytest.raises(ValueError, match="canonical numerical report row"):
        bayesian_quantile_rows(result, settings=settings, unit="m3/s", numerical_rows=altered)
    with pytest.raises(ValueError, match="exact systematic"):
        bayesian_quantile_rows(
            result,
            settings={**settings, "source_format": "classified_systematic_csv"},
            unit="m3/s",
            numerical_rows=canonical,
        )
    incomplete = replace(
        result, values=tuple(p for p in result.values if p.probability_limit_side != "upper")
    )
    with pytest.raises(ValueError, match="incomplete or duplicate"):
        bayesian_quantile_rows(
            incomplete,
            settings=settings,
            unit="m3/s",
            numerical_rows=_result_rows(report_document(incomplete, settings=settings)),
        )


def test_undefined_posterior_mean_is_not_reconstructed() -> None:
    # Presentation-only synthetic case: use the admitted upper-tail omission rule.
    result, settings = calculate_from_bytes(_ams(), _settings())
    reduced = replace(
        result,
        values=tuple(p for p in result.values if p.meaning != ResultMeaning.POSTERIOR_MEAN_QUANTILE),
        warnings=(*result.warnings, "UNDEFINED posterior mean quantile; upper-tail mean diverges."),
    )
    estimated = {**settings, "fixed_log_sd": None}
    rows = bayesian_quantile_rows(
        reduced,
        settings=estimated,
        unit="m3/s",
        numerical_rows=_result_rows(report_document(reduced, settings=estimated)),
    )
    assert len(rows) == 5
    assert not any(row.point.meaning == ResultMeaning.POSTERIOR_MEAN_QUANTILE for row in rows)
