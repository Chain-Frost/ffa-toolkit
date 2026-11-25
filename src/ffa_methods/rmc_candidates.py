"""WP02 read-only RMC/ARR external candidate datasets, not native Flike inputs.

The source labels for Wimmera's 56 magnitude-only rows are artificial.
There is no inference admission or retroactive conversion into event years.
Raw literal values and candidate provenance remain available to the caller.
"""

import json
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from hashlib import sha256
from math import isfinite
from pathlib import Path
from typing import cast

from .contracts import AnnualObservation, ObservationSeries
from .contracts import ExposureKind as ContractExposureKind
from .contracts import ObservationKind as ContractObservationKind

_STATUS = "candidate_external_comparison"
_CORPUS_ID = "rmc-arr-flike"
_UPSTREAM_COMMIT = "bfce2ddbebb8d75e6cf6e21c4fef8ecfb5058269"
_ALLOWED_FILES = frozenset({"README.md", "datasets.json", "cases.json"})
_EXPECTED_DATASETS = {
    "hunter-singleton-31": (31, 1938, 1968, "calendar_year_labels_as_published"),
    "wimmera-glynwylin-56-magnitudes": (56, 1960, 2015, "artificial_source_labels_observation_years_unknown"),
}
_EXPECTED_CASES = frozenset({"rmc-arr-ex3", "rmc-arr-ex4", "rmc-arr-ex5", "rmc-arr-ex6a", "rmc-arr-ex6b"})


class PublicCandidateError(ValueError):
    """Corrupt, unsupported or unqualified source-candidate evidence."""


@dataclass(frozen=True, slots=True)
class PublicMagnitude:
    source_row: int
    source_year_label: int
    observed_year: int | None
    discharge: Decimal
    source_discharge_literal: str


@dataclass(frozen=True, slots=True)
class PublicDataset:
    dataset_id: str
    name: str
    discharge_unit: str
    chronology_status: str
    source_symbol: str
    observations: tuple[PublicMagnitude, ...]

    @property
    def candidate_status(self) -> str:
        return _STATUS


@dataclass(frozen=True, slots=True)
class HistoricalWindow:
    """Count evidence, not 118 separately observed discharges."""

    start_year: int
    end_year: int
    threshold: Decimal
    count_not_above: int
    count_above: int

    @property
    def inclusive_exposure_years(self) -> int:
        return self.end_year - self.start_year + 1

    def __post_init__(self) -> None:
        if (
            self.start_year < 1
            or self.end_year < self.start_year
            or self.count_not_above < 0
            or self.count_above < 0
            or self.count_not_above + self.count_above != self.inclusive_exposure_years
            or not self.threshold.is_finite()
            or self.threshold < 0
        ):
            msg = "Historical counts must partition the explicit inclusive calendar window"
            raise PublicCandidateError(msg)


@dataclass(frozen=True, slots=True)
class PublicCase:
    case_id: str
    dataset_id: str
    historical_windows: tuple[HistoricalWindow, ...]
    candidate_status: str = _STATUS


@dataclass(frozen=True, slots=True)
class PublicCandidateCorpus:
    source_commit: str
    manifest_sha256: str
    datasets_sha256: str
    cases_sha256: str
    datasets: tuple[PublicDataset, ...]
    cases: tuple[PublicCase, ...]
    candidate_status: str = _STATUS
    native_qualification_eligible: bool = False

    def dataset(self, dataset_id: str) -> PublicDataset:
        for item in self.datasets:
            if item.dataset_id == dataset_id:
                return item
        msg = f"Unknown public candidate dataset: {dataset_id}"
        raise PublicCandidateError(msg)

    def case(self, case_id: str) -> PublicCase:
        for item in self.cases:
            if item.case_id == case_id:
                return item
        msg = f"Unknown public candidate case: {case_id}"
        raise PublicCandidateError(msg)


@dataclass(frozen=True, slots=True)
class CandidateAnnualMapping:
    """Only a published calendar-labelled source series; native admission remains false."""

    series: ObservationSeries
    source_dataset: PublicDataset
    candidate_status: str = _STATUS
    ready_for_native_likelihood: bool = False


def _record(value: object, context: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in cast(dict[object, object], value)):
        msg = f"Expected an object with string keys for {context}"
        raise PublicCandidateError(msg)
    return cast(dict[str, object], value)


def _rows(value: object, context: str) -> list[dict[str, object]]:
    if not isinstance(value, list):
        msg = f"Expected a list of records for {context}"
        raise PublicCandidateError(msg)
    return [_record(item, context) for item in cast(list[object], value)]


def _text(value: object, context: str) -> str:
    if not isinstance(value, str) or not value:
        msg = f"Expected nonempty text for {context}"
        raise PublicCandidateError(msg)
    return value


def _integer(value: object, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        msg = f"Expected integer for {context}"
        raise PublicCandidateError(msg)
    return value


def _decimal(value: object, literal: object, context: str) -> Decimal:
    token = _text(literal, context)
    if isinstance(value, bool) or not isinstance(value, (int, Decimal)):
        msg = f"Expected JSON number for {context}"
        raise PublicCandidateError(msg)
    try:
        original = Decimal(token)
        numerical = Decimal(str(value))
    except InvalidOperation as exc:
        msg = f"Invalid numeric source literal in {context}"
        raise PublicCandidateError(msg) from exc
    if not original.is_finite() or original < 0 or original != numerical:
        msg = f"Source literal and numeric field conflict in {context}"
        raise PublicCandidateError(msg)
    return original


def _reject_constant(token: str) -> object:
    msg = f"Unsupported nonfinite JSON constant {token}"
    raise PublicCandidateError(msg)


def _load_json(path: Path) -> tuple[dict[str, object], bytes]:
    try:
        data = path.read_bytes()
        decoded: object = json.loads(data.decode("utf-8"), parse_float=Decimal, parse_constant=_reject_constant)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        msg = f"Unreadable public candidate document: {path.name}"
        raise PublicCandidateError(msg) from exc
    return _record(decoded, path.name), data


def _candidate(record: dict[str, object], context: str) -> None:
    if record.get("status") != _STATUS or record.get("native_qualification_eligible") is not False:
        msg = f"{context} is not an unqualified external candidate"
        raise PublicCandidateError(msg)


def _verify_staged_files(root: Path, manifest: dict[str, object]) -> None:
    declared: set[str] = set()
    for entry in _rows(manifest.get("files"), "manifest files"):
        name = _text(entry.get("path"), "staged path")
        if name not in _ALLOWED_FILES or name in declared:
            msg = f"Unexpected or repeated public candidate path: {name}"
            raise PublicCandidateError(msg)
        declared.add(name)
        try:
            data = (root / name).read_bytes()
        except OSError as exc:
            msg = f"Missing public candidate artifact: {name}"
            raise PublicCandidateError(msg) from exc
        size = _integer(entry.get("size_bytes"), "artifact size")
        expected_hash = _text(entry.get("sha256"), "artifact hash")
        if size != len(data) or sha256(data).hexdigest() != expected_hash:
            msg = f"Public candidate artifact hash/size mismatch: {name}"
            raise PublicCandidateError(msg)
    if declared != set(_ALLOWED_FILES):
        msg = "Public candidate manifest omits a required artifact"
        raise PublicCandidateError(msg)


def _parse_dataset(record: dict[str, object]) -> PublicDataset:
    _candidate(record, "dataset")
    dataset_id = _text(record.get("dataset_id"), "dataset_id")
    if dataset_id not in _EXPECTED_DATASETS:
        msg = f"Unsupported public candidate dataset: {dataset_id}"
        raise PublicCandidateError(msg)
    count, first_year, last_year, chronology = _EXPECTED_DATASETS[dataset_id]
    if record.get("chronology_status") != chronology or record.get("discharge_unit") != "m^3/s":
        msg = f"Unexpected chronology or unit for {dataset_id}"
        raise PublicCandidateError(msg)
    if _integer(record.get("observation_count"), "observation_count") != count:
        msg = f"Incorrect declared count for {dataset_id}"
        raise PublicCandidateError(msg)
    observations = _parse_magnitudes(
        _rows(record.get("observations"), "observations"), dataset_id, first_year, chronology
    )
    if len(observations) != count or observations[-1].source_year_label != last_year:
        msg = f"Incomplete numeric source records for {dataset_id}"
        raise PublicCandidateError(msg)
    return PublicDataset(
        dataset_id=dataset_id,
        name=_text(record.get("name"), "dataset name"),
        discharge_unit="m^3/s",
        chronology_status=chronology,
        source_symbol=_text(record.get("source_symbol"), "source_symbol"),
        observations=observations,
    )


def _parse_magnitudes(
    records: list[dict[str, object]], dataset_id: str, first_year: int, chronology: str
) -> tuple[PublicMagnitude, ...]:
    parsed: list[PublicMagnitude] = []
    seen_years: set[int] = set()
    for source_row, record in enumerate(records, start=1):
        index = _integer(record.get("source_row"), "source_row")
        label = _integer(record.get("source_year_label"), "source_year_label")
        recorded = record.get("observed_year")
        if index != source_row or label != first_year + source_row - 1:
            msg = f"Out-of-order source row or year label in {dataset_id}"
            raise PublicCandidateError(msg)
        if record.get("source_observation_kind") != "exact":
            msg = f"Unsupported observation classification in {dataset_id}"
            raise PublicCandidateError(msg)
        if chronology == "calendar_year_labels_as_published":
            if recorded != label or label in seen_years:
                msg = f"Invalid published event calendar year in {dataset_id}"
                raise PublicCandidateError(msg)
            observed_year: int | None = label
            seen_years.add(label)
        else:
            if recorded is not None:
                msg = f"Artificial Wimmera source labels cannot be event years in {dataset_id}"
                raise PublicCandidateError(msg)
            observed_year = None
        value = _decimal(record.get("discharge"), record.get("source_discharge_literal"), "discharge")
        if value <= 0:
            msg = f"Unexpected nonpositive candidate magnitude in {dataset_id}"
            raise PublicCandidateError(msg)
        parsed.append(
            PublicMagnitude(
                source_row=index,
                source_year_label=label,
                observed_year=observed_year,
                discharge=value,
                source_discharge_literal=_text(record.get("source_discharge_literal"), "source literal"),
            )
        )
    return tuple(parsed)


def _parse_window(record: dict[str, object]) -> HistoricalWindow:
    start = _integer(record.get("start_year"), "historical start")
    end = _integer(record.get("end_year"), "historical end")
    above = _integer(record.get("count_above"), "historical above count")
    not_above = _integer(record.get("count_not_above"), "historical not-above count")
    threshold = _decimal(
        record.get("threshold_discharge"), str(record.get("threshold_discharge")), "historical threshold"
    )
    window = HistoricalWindow(start, end, threshold, not_above, above)
    if (
        _integer(record.get("inclusive_exposure_years"), "inclusive historical exposure")
        != window.inclusive_exposure_years
    ):
        msg = "Historical source window and inclusive exposure length differ"
        raise PublicCandidateError(msg)
    return window


def _parse_case(record: dict[str, object], dataset_ids: set[str]) -> PublicCase:
    _candidate(record, "case")
    case_id = _text(record.get("case_id"), "case_id")
    dataset_id = _text(record.get("dataset_id"), "dataset_id")
    if case_id not in _EXPECTED_CASES or dataset_id not in dataset_ids:
        msg = f"Unexpected case or unknown dataset: {case_id}"
        raise PublicCandidateError(msg)
    if record.get("native_rerun_performed") is not False or record.get("native_settings_sha256") is not None:
        msg = f"Unexpected native qualification evidence in {case_id}"
        raise PublicCandidateError(msg)
    acceptance = _record(record.get("ffa_acceptance"), "ffa_acceptance")
    if acceptance.get("status") != "not_admitted" or acceptance.get("adopts_source_test_tolerance") is not False:
        msg = f"Unreviewed candidate acceptance in {case_id}"
        raise PublicCandidateError(msg)
    history = tuple(_parse_window(group) for group in _rows(record.get("historical_groups"), "historical_groups"))
    if (case_id == "rmc-arr-ex4") != (len(history) == 1):
        msg = f"Unexpected historical group definition in {case_id}"
        raise PublicCandidateError(msg)
    return PublicCase(case_id, dataset_id, history)


def read_rmc_candidates(root: Path) -> PublicCandidateCorpus:
    """Check pinned staged byte custody, then load only literal observation evidence."""
    manifest, manifest_bytes = _load_json(root / "manifest.json")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("corpus_id") != _CORPUS_ID
        or manifest.get("upstream_commit") != _UPSTREAM_COMMIT
    ):
        msg = "Unsupported RMC external candidate provenance"
        raise PublicCandidateError(msg)
    _candidate(manifest, "corpus")
    _verify_staged_files(root, manifest)
    data, dataset_bytes = _load_json(root / "datasets.json")
    cases, case_bytes = _load_json(root / "cases.json")
    for kind, record in (("datasets", data), ("cases", cases)):
        if record.get("schema_version") != 1 or record.get("corpus_id") != _CORPUS_ID:
            msg = f"Unsupported RMC {kind} schema or corpus"
            raise PublicCandidateError(msg)
        _candidate(record, kind)
    datasets = tuple(_parse_dataset(record) for record in _rows(data.get("datasets"), "datasets"))
    if {item.dataset_id for item in datasets} != set(_EXPECTED_DATASETS) or len(datasets) != 2:
        msg = "Expected exactly two distinct source candidate datasets"
        raise PublicCandidateError(msg)
    case_items = tuple(
        _parse_case(record, {item.dataset_id for item in datasets}) for record in _rows(cases.get("cases"), "cases")
    )
    if {item.case_id for item in case_items} != set(_EXPECTED_CASES) or len(case_items) != 5:
        msg = "Expected exactly five distinct RMC comparison cases"
        raise PublicCandidateError(msg)
    corpus = PublicCandidateCorpus(
        source_commit=_UPSTREAM_COMMIT,
        manifest_sha256=sha256(manifest_bytes).hexdigest(),
        datasets_sha256=sha256(dataset_bytes).hexdigest(),
        cases_sha256=sha256(case_bytes).hexdigest(),
        datasets=datasets,
        cases=case_items,
    )
    for case in corpus.cases:
        if case.historical_windows:
            calendar_exposure_years(case, corpus.dataset(case.dataset_id))
    return corpus


def calendar_exposure_years(case: PublicCase, dataset: PublicDataset) -> int:
    """Count calendar exposure once; grouped history is not expanded to annual events."""
    years = [observation.observed_year for observation in dataset.observations]
    if any(year is None for year in years):
        msg = "Undated magnitudes cannot establish annual calendar exposure"
        raise PublicCandidateError(msg)
    existing = {year for year in years if year is not None}
    for window in case.historical_windows:
        historic = set(range(window.start_year, window.end_year + 1))
        if historic.intersection(existing):
            msg = "Historical window overlaps annual records or an earlier historical window"
            raise PublicCandidateError(msg)
        existing.update(historic)
    return len(existing)


def candidate_annual_series(dataset: PublicDataset) -> CandidateAnnualMapping:
    """Permit calendar-labelled Hunter evidence only; never invent Wimmera event years."""
    if dataset.chronology_status != "calendar_year_labels_as_published":
        msg = "Artificial source year labels must not be converted to observed event years"
        raise PublicCandidateError(msg)
    annual: list[AnnualObservation] = []
    for observation in dataset.observations:
        year = observation.observed_year
        if year is None or year != observation.source_year_label:
            msg = "Missing or conflicting observed calendar-year evidence"
            raise PublicCandidateError(msg)
        flow = float(observation.discharge)
        if not isfinite(flow) or flow == 0:
            msg = "Unrepresentable candidate discharge in annual contract"
            raise PublicCandidateError(msg)
        annual.append(
            AnnualObservation(
                year=year,
                kind=ContractObservationKind.EXACT,
                exposure=ContractExposureKind.SYSTEMATIC,
                value=flow,
            )
        )
    return CandidateAnnualMapping(
        series=ObservationSeries(
            discharge_unit=dataset.discharge_unit,
            annual=tuple(annual),
            groups=(),
            zero_threshold=None,
            source_file_version=None,
            year_definition="published calendar labels; original gauge archive unverified",
        ),
        source_dataset=dataset,
    )
