"""Independent analytic and typed interval fixtures for the reporting layer."""

import csv
import io
from dataclasses import replace
from pathlib import Path
from typing import Literal, cast

import pytest

from ffa_methods.contracts import ProbabilityKind, ResultPoint
from ffa_methods.deterministic_workflow import calculate_from_bytes
from ffa_methods.frequency_quantiles import BOOTSTRAP_CONSTRUCTION, BOOTSTRAP_LIMIT, deterministic_quantile_rows
from ffa_methods.frequency_reporting import render_frequency_files
from ffa_methods.reporting import report_document

EXAMPLES = Path(__file__).resolve().parents[1] / "docs" / "examples"


def test_completed_interval_rows_are_not_mc_error() -> None:
    # Exercise actual result contracts and formatter with explicit synthetic endpoints,
    # separate from the end-to-end bootstrap reference tests.
    ams = (EXAMPLES / "ordinary-ams.csv").read_bytes()
    result, settings = calculate_from_bytes(ams, (EXAMPLES / "ordinary-settings.json").read_bytes())
    original = result.values
    endpoints: list[ResultPoint] = []
    for point in original:
        sides: tuple[tuple[Literal["lower", "upper"], float], ...] = (("lower", 0.8), ("upper", 1.3))
        for side, multiplier in sides:
            endpoints.append(
                ResultPoint(
                    meaning=BOOTSTRAP_LIMIT,
                    value=point.value * multiplier,
                    unit=point.unit,
                    input_probability=point.input_probability,
                    input_probability_kind=ProbabilityKind.NON_EXCEEDANCE,
                    confidence_level=0.8,
                    probability_limit_side=side,
                    probability_limit_construction=BOOTSTRAP_CONSTRUCTION,
                )
            )
    result = replace(result, values=(*original, *endpoints))
    settings.update(
        bootstrap_status="completed",
        bootstrap_confidence=0.8,
        bootstrap_interval_target="conditional_on_successful_refits",
    )
    rows = cast(list[dict[str, object]], report_document(result, settings=settings)["results"])
    files = render_frequency_files(result, settings=settings, raw_ams=ams, report_text="", numerical_rows=rows)
    svg = files["frequency.svg"]
    assert svg.count('data-interval="parametric_bootstrap"') == len(original)
    assert "Conditional on successful refits" in svg
    assert "not posterior or predictive intervals, simultaneous bands or endpoint Monte Carlo brackets" in files["report.html"]
    plotted = [
        row
        for row in csv.DictReader(io.StringIO(files["frequency-coordinates.csv"]))
        if row["series"] != "observed"
    ]
    assert len(plotted) == len(result.values)
    for row in plotted:
        point = result.values[int(row["result_row"]) - 1]
        assert float(row["discharge"]) == point.value
        assert row["meaning"] == point.meaning.value
        assert row["precision_state"] == ""
        assert row["probability_limit_side"] == (point.probability_limit_side or "")

    altered = [dict(row) for row in rows]
    altered[0]["precision_state"] = "conditional_pass"
    with pytest.raises(ValueError, match="cannot claim a Bayesian"):
        deterministic_quantile_rows(result, settings=settings, unit="m3/s", numerical_rows=altered)
    with pytest.raises(ValueError, match="completed bootstrap"):
        deterministic_quantile_rows(
            result,
            settings={**settings, "bootstrap_status": "refused"},
            unit="m3/s",
            numerical_rows=rows,
        )
