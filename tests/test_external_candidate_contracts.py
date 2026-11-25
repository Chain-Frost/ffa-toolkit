"""Synthetic exposure and custody checks; no third-party gauge data."""

import json
from dataclasses import replace
from decimal import Decimal
from hashlib import sha256
from pathlib import Path

import pytest

from ffa_methods.rmc_candidates import (
    HistoricalWindow,
    PublicCandidateError,
    PublicCase,
    PublicDataset,
    PublicMagnitude,
    calendar_exposure_years,
    candidate_annual_series,
    read_rmc_candidates,
)


def _dataset() -> PublicDataset:
    return PublicDataset(
        "synthetic",
        "Synthetic",
        "m^3/s",
        "calendar_year_labels_as_published",
        "generated",
        tuple(PublicMagnitude(i, 2000 + i, 2000 + i, Decimal(i + 1), str(i + 1)) for i in range(3)),
    )


def test_synthetic_mapping_preserves_values_without_native_admission() -> None:
    mapped = candidate_annual_series(_dataset())
    assert [row.year for row in mapped.series.annual] == [2000, 2001, 2002]
    assert [row.value for row in mapped.series.annual] == [1.0, 2.0, 3.0]
    assert not mapped.ready_for_native_likelihood
    assert mapped.series.groups == ()
    assert mapped.series.zero_threshold is None


def test_history_counts_exposure_without_fabricating_observations() -> None:
    history = HistoricalWindow(1990, 1999, Decimal(20), 9, 1)
    case = PublicCase("synthetic", "synthetic", (history,))
    assert calendar_exposure_years(case, _dataset()) == 13
    assert len(_dataset().observations) == 3
    with pytest.raises(PublicCandidateError, match="overlaps"):
        calendar_exposure_years(replace(case, historical_windows=(history, history)), _dataset())
    with pytest.raises(PublicCandidateError, match="partition"):
        HistoricalWindow(1990, 1999, Decimal(20), 8, 1)


def test_artificial_years_and_missing_calendar_evidence_are_rejected() -> None:
    dataset = _dataset()
    with pytest.raises(PublicCandidateError, match="Artificial"):
        candidate_annual_series(replace(dataset, chronology_status="artificial"))
    unknown = replace(dataset, observations=(replace(dataset.observations[0], observed_year=None),))
    with pytest.raises(PublicCandidateError, match="Missing or conflicting"):
        candidate_annual_series(unknown)
    with pytest.raises(PublicCandidateError, match="Undated"):
        calendar_exposure_years(PublicCase("synthetic", "synthetic", ()), unknown)


def test_unpinned_external_corpus_fails_closed(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_text('{"schema_version":1}', encoding="utf-8")
    with pytest.raises(PublicCandidateError, match="Unsupported"):
        read_rmc_candidates(tmp_path)


_STATUS: dict[str, object] = {
    "status": "candidate_external_comparison",
    "native_qualification_eligible": False,
}


def _synthetic_corpus(root: Path) -> None:
    """Exercise the strict reader schema with generated rows, never source gauge values."""
    datasets: list[dict[str, object]] = []
    for name, count, start, calendar in (
        ("hunter-singleton-31", 31, 1938, True),
        ("wimmera-glynwylin-56-magnitudes", 56, 1960, False),
    ):
        datasets.append(
            {
                **_STATUS,
                "dataset_id": name,
                "name": "Generated fixture",
                "source_symbol": "synthetic",
                "discharge_unit": "m^3/s",
                "observation_count": count,
                "chronology_status": (
                    "calendar_year_labels_as_published"
                    if calendar
                    else "artificial_source_labels_observation_years_unknown"
                ),
                "observations": [
                    {
                        "source_row": i + 1,
                        "source_year_label": start + i,
                        "observed_year": start + i if calendar else None,
                        "source_observation_kind": "exact",
                        "discharge": (i + 1) ** 2,
                        "source_discharge_literal": str((i + 1) ** 2),
                    }
                    for i in range(count)
                ],
            }
        )
    cases: list[dict[str, object]] = []
    for case_id in ("rmc-arr-ex3", "rmc-arr-ex4", "rmc-arr-ex5", "rmc-arr-ex6a", "rmc-arr-ex6b"):
        history: list[dict[str, object]] = []
        if case_id == "rmc-arr-ex4":
            history = [
                {
                    "start_year": 1930,
                    "end_year": 1937,
                    "threshold_discharge": 50,
                    "count_above": 1,
                    "count_not_above": 7,
                    "inclusive_exposure_years": 8,
                }
            ]
        cases.append(
            {
                **_STATUS,
                "case_id": case_id,
                "dataset_id": datasets[0]["dataset_id"],
                "native_rerun_performed": False,
                "native_settings_sha256": None,
                "ffa_acceptance": {"status": "not_admitted", "adopts_source_test_tolerance": False},
                "historical_groups": history,
            }
        )
    common: dict[str, object] = {**_STATUS, "schema_version": 1, "corpus_id": "rmc-arr-flike"}
    for name, key, records in (("datasets.json", "datasets", datasets), ("cases.json", "cases", cases)):
        (root / name).write_text(json.dumps({**common, key: records}), encoding="utf-8")
    (root / "README.md").write_text("Generated test corpus, not published observations.\n", encoding="utf-8")
    files: list[dict[str, object]] = []
    for name in ("README.md", "datasets.json", "cases.json"):
        data = (root / name).read_bytes()
        files.append({"path": name, "size_bytes": len(data), "sha256": sha256(data).hexdigest()})
    manifest = {**common, "upstream_commit": "bfce2ddbebb8d75e6cf6e21c4fef8ecfb5058269", "files": files}
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_generated_full_corpus_reader_preserves_literal_custody_and_status(tmp_path: Path) -> None:
    _synthetic_corpus(tmp_path)
    corpus = read_rmc_candidates(tmp_path)
    assert not corpus.native_qualification_eligible
    assert len(corpus.datasets) == 2
    assert len(corpus.cases) == 5
    assert corpus.datasets[0].observations[2].discharge == Decimal(9)
    assert corpus.datasets[0].observations[2].source_discharge_literal == "9"
    assert calendar_exposure_years(corpus.case("rmc-arr-ex4"), corpus.datasets[0]) == 39
    assert all(row.observed_year is None for row in corpus.datasets[1].observations)
    with pytest.raises(PublicCandidateError, match="Artificial"):
        candidate_annual_series(corpus.datasets[1])


@pytest.mark.parametrize("name", ["datasets.json", "cases.json", "README.md"])
def test_generated_corpus_rejects_hash_or_size_tampering(tmp_path: Path, name: str) -> None:
    _synthetic_corpus(tmp_path)
    path = tmp_path / name
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(PublicCandidateError, match="hash/size mismatch"):
        read_rmc_candidates(tmp_path)
