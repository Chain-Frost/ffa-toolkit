"""Immutable candidate-artifact custody checks and conservative pairing review."""

import json
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import cast

from .models import LegacyInputError, Observation
from .project import LegacyProject, parse_ams_csv, parse_fld
from .report import LegacyReport, parse_report


class CaseStatus(StrEnum):
    CANDIDATE = "candidate_settings_unverified"
    VERIFIED = "verified_pair"
    MISMATCHED = "mismatched"
    UNAVAILABLE = "unavailable"
    QUALIFIED = "qualified_reference"


@dataclass(frozen=True, slots=True)
class CaseReview:
    case_id: str
    status: CaseStatus
    project_observations: int
    report_observations: int
    discrepancies: tuple[str, ...]
    unresolved: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CustodyReview:
    checked_files: int
    unresolved_originals: tuple[str, ...]
    cases: tuple[CaseReview, ...]


def _safe_path(root: Path, relative: str) -> Path:
    posix = PurePosixPath(relative)
    if posix.is_absolute() or not posix.parts or any(part in {".", ".."} for part in posix.parts):
        msg = f"Unsafe manifest path: {relative}"
        raise LegacyInputError(msg)
    path = root.joinpath(*posix.parts)
    if not path.resolve().is_relative_to(root.resolve()):
        msg = f"Manifest path leaves evidence root: {relative}"
        raise LegacyInputError(msg)
    return path


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        msg = f"Manifest field {name} must be a nonempty string"
        raise LegacyInputError(msg)
    return value


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        msg = f"Manifest field {name} must be a nonnegative integer"
        raise LegacyInputError(msg)
    return value


def _hex_hash(value: object, name: str) -> str:
    result = _string(value, name)
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        msg = f"Manifest field {name} must be a SHA-256 hex digest"
        raise LegacyInputError(msg)
    return result


def _record(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in cast(dict[object, object], value)):
        msg = f"Manifest field {name} must be an object with string keys"
        raise LegacyInputError(msg)
    return cast(dict[str, object], value)


def _records(value: object, name: str) -> list[dict[str, object]]:
    if not isinstance(value, list):
        msg = f"Manifest field {name} must be a list"
        raise LegacyInputError(msg)
    return [_record(record, name) for record in cast(list[object], value)]


def verify_manifest_files(root: Path, manifest: dict[str, object]) -> tuple[int, tuple[str, ...]]:
    """Hash staged bytes; original archive hashes are custody claims, not verified here."""
    if manifest.get("schema_version") != 1:
        msg = "Unsupported provenance manifest schema"
        raise LegacyInputError(msg)
    items = _records(manifest.get("files"), "files")
    if not items:
        msg = "Empty file manifest"
        raise LegacyInputError(msg)
    seen: set[str] = set()
    originals_missing: list[str] = []
    for item in items:
        rel = _string(item.get("staged_path"), "staged_path")
        if rel in seen:
            msg = f"Repeated staged path: {rel}"
            raise LegacyInputError(msg)
        seen.add(rel)
        _verify_manifest_item(root, item, rel)
        # Archive members are not present: never claim source archive bytes were rehashed.
        originals_missing.append(f"{rel}: original archive/member bytes not available for independent rehash")
    return len(items), tuple(originals_missing)


def _report_rounding_matches(raw: Decimal, reported: Decimal) -> bool:
    """Bound a printed value by half its least significant decimal unit.

    This admits exact half-way rounding differences without assuming the
    native binary floating-point rounding mode. It is not an estimator tolerance.
    """
    exponent = reported.as_tuple().exponent
    if not isinstance(exponent, int):
        msg = "Report rounding comparison requires a finite decimal"
        raise LegacyInputError(msg)
    return abs(raw - reported) <= Decimal(1).scaleb(exponent) / 2


def _compare_report(project: tuple[Observation, ...], report: LegacyReport) -> list[str]:
    issues: list[str] = []
    by_year = {observation.year: observation for observation in project}
    if len(project) != len(report.observations):
        issues.append("Project/report gauged-observation counts disagree")
    if set(by_year) != {observation.year for observation in report.observations}:
        issues.append("Project/report years differ")
    for observation in report.observations:
        source = by_year.get(observation.year)
        if (
            source is not None
            and source.value is not None
            and not _report_rounding_matches(source.value, observation.discharge)
        ):
            issues.append(f"Report flow differs at printed precision in {observation.year}")
    at_or_below = sum(
        observation.value is not None and observation.value <= report.zero_threshold for observation in project
    )
    if at_or_below != report.reported_at_or_below_threshold:
        issues.append("Reported zero-threshold count conflicts with project observation values")
    return issues


def review_candidate(root: Path, case: dict[str, object], files: list[dict[str, object]]) -> CaseReview:
    """Review observed correspondence but NEVER upgrade a candidate to verified."""
    case_id = _string(case.get("case_id"), "case_id")
    expected_status = _string(case.get("pairing_status"), "pairing_status")
    if expected_status not in {CaseStatus.CANDIDATE, CaseStatus.MISMATCHED, CaseStatus.UNAVAILABLE}:
        msg = f"Case {case_id} requests promotion without pinned native-run evidence"
        raise LegacyInputError(msg)
    entries = [entry for entry in files if entry.get("case_id") == case_id]
    projects = [entry for entry in entries if str(entry.get("staged_path", "")).endswith(".fld")]
    csvs = [entry for entry in entries if str(entry.get("staged_path", "")).endswith(".csv")]
    reports = [entry for entry in entries if str(entry.get("staged_path", "")).endswith(".txt")]
    if not all(len(group) == 1 for group in (projects, csvs, reports)):
        return CaseReview(case_id, CaseStatus.UNAVAILABLE, 0, 0, (), ("Unpaired project/CSV/report artifacts",))
    project = parse_fld(_safe_path(root, _string(projects[0]["staged_path"], "staged_path")).read_bytes())
    csv_observations = parse_ams_csv(_safe_path(root, _string(csvs[0]["staged_path"], "staged_path")).read_bytes())
    report = parse_report(_safe_path(root, _string(reports[0]["staged_path"], "staged_path")).read_bytes())
    mismatches = _compare_case_settings(case, project, report)
    mismatches.extend(_compare_case_observations(case, project, csv_observations, report))
    unresolved = [
        "Source project flag semantics and trailing controls are undocumented",
        "Project prior/default/run switches have not been decoded",
        "No pinned native executable, native rerun, run seed, or complete settings custody",
        "No demonstrated stochastic repeatability or parameter-output comparison precision",
    ]
    if any(observation.raw_flag != 0 for observation in project.observations):
        unresolved.append("Nonzero observation flags have no verified interpretation")
    if report.unknown_sections:
        unresolved.append("Additional historical/censor sections require a separate adapter")
    if any(parameter.label_contains_nul for parameter in report.posterior_parameters):
        unresolved.append("Report contains an embedded-NUL posterior parameter label")
    if expected_status == CaseStatus.MISMATCHED:
        unresolved.append("Manifest explicitly marks this evidence as mismatched")
    if expected_status == CaseStatus.UNAVAILABLE:
        unresolved.append("Manifest marks evidence as unavailable; no automatic promotion")
    if mismatches or expected_status == CaseStatus.MISMATCHED:
        status = CaseStatus.MISMATCHED
    elif expected_status == CaseStatus.UNAVAILABLE:
        status = CaseStatus.UNAVAILABLE
    else:
        status = CaseStatus.CANDIDATE
    return CaseReview(
        case_id, status, len(project.observations), len(report.observations), tuple(mismatches), tuple(unresolved)
    )


def review_corpus(root: Path) -> CustodyReview:
    """Audit manifest hashes and two candidate triples; do not mutate evidence."""
    manifest_path = _safe_path(root, "manifest.json")
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        msg = "Unreadable legacy candidate manifest"
        raise LegacyInputError(msg) from exc
    manifest = _record(raw, "manifest")
    count, missing = verify_manifest_files(root, manifest)
    cases = _records(manifest.get("cases"), "cases")
    files = _records(manifest.get("files"), "files")
    if not cases:
        msg = "Manifest contains no cases"
        raise LegacyInputError(msg)
    if len({_string(case.get("case_id"), "case_id") for case in cases}) != len(cases):
        msg = "Duplicate manifest case identifier"
        raise LegacyInputError(msg)
    if {entry.get("case_id") for entry in files} != {case.get("case_id") for case in cases}:
        msg = "Manifest files/cases membership differs"
        raise LegacyInputError(msg)
    return CustodyReview(count, missing, tuple(review_candidate(root, case, files) for case in cases))


def _verify_manifest_item(root: Path, item: dict[str, object], rel: str) -> None:
    member = _string(item.get("source_member"), "source_member")
    if PurePosixPath(member).is_absolute() or ".." in PurePosixPath(member).parts:
        msg = f"Unsafe archive member name for {rel}"
        raise LegacyInputError(msg)
    _hex_hash(item.get("source_archive_sha256"), "source_archive_sha256")
    original_hash = _hex_hash(item.get("original_sha256"), "original_sha256")
    staged_hash = _hex_hash(item.get("staged_sha256"), "staged_sha256")
    original_size = _integer(item.get("original_size_bytes"), "original_size_bytes")
    staged_size = _integer(item.get("staged_size_bytes"), "staged_size_bytes")
    file_path = _safe_path(root, rel)
    if not file_path.is_file():
        msg = f"Staged evidence missing: {rel}"
        raise LegacyInputError(msg)
    data = file_path.read_bytes()
    if len(data) != staged_size or sha256(data).hexdigest() != staged_hash:
        msg = f"Staged evidence byte/hash mismatch: {rel}"
        raise LegacyInputError(msg)
    raw_transforms = item.get("transformations")
    if not isinstance(raw_transforms, list):
        msg = f"Invalid transformation list for {rel}"
        raise LegacyInputError(msg)
    transforms = cast(list[object], raw_transforms)
    if not transforms:
        if original_hash != staged_hash or original_size != staged_size:
            msg = f"Undocumented transformation for {rel}"
            raise LegacyInputError(msg)
    else:
        if len(transforms) != 1:
            msg = f"Unsupported transformation chain for {rel}"
            raise LegacyInputError(msg)
        transform = _record(transforms[0], "transformations")
        if (
            transform.get("type") != "source_path_redaction"
            or transform.get("replacement") != "[REDACTED_SOURCE_PATH]"
            or transform.get("occurrences") != 1
            or data.count(b"Data file: [REDACTED_SOURCE_PATH]") != 1
            or original_hash == staged_hash
        ):
            msg = f"Unverified/unsupported redaction metadata for {rel}"
            raise LegacyInputError(msg)


def _compare_case_settings(case: dict[str, object], project: LegacyProject, report: LegacyReport) -> list[str]:
    mismatches: list[str] = []
    if report.program_version != "5.0.306.0" or report.file_version != project.file_version:
        mismatches.append("Program or file-version baseline conflict")
    if report.title != project.title:
        mismatches.append("Project/report title conflict")
    if report.distribution != _string(case.get("distribution"), "distribution"):
        mismatches.append("Report distribution differs from case manifest")
    declared_threshold = case.get("zero_threshold")
    if isinstance(declared_threshold, bool) or not isinstance(declared_threshold, (float, int)):
        msg = "Case zero_threshold must be a numeric value"
        raise LegacyInputError(msg)
    if report.zero_threshold != Decimal(str(declared_threshold)):
        mismatches.append("Report zero threshold differs from case manifest")
    all_zero_flags = case.get("project_observation_flags_all_zero")
    if not isinstance(all_zero_flags, bool):
        msg = "Case project_observation_flags_all_zero must be a boolean"
        raise LegacyInputError(msg)
    if all(flag.raw_flag == 0 for flag in project.observations) != all_zero_flags:
        mismatches.append("Project observation flags disagree with case manifest")
    return mismatches


def _compare_case_observations(
    case: dict[str, object], project: LegacyProject, csv_observations: tuple[Observation, ...], report: LegacyReport
) -> list[str]:
    mismatches: list[str] = []
    if len(project.observations) != len(csv_observations):
        mismatches.append("CSV and project observation counts disagree")
    if {observation.year: observation.value for observation in project.observations} != {
        observation.year: observation.value for observation in csv_observations
    }:
        mismatches.append("CSV and project annual maximum values/years differ")
    mismatches.extend(_compare_report(project.observations, report))
    if len(project.observations) != _integer(case.get("observations"), "observations"):
        mismatches.append("Manifest project observation count differs")
    if len(report.quantiles) != _integer(case.get("quantile_rows"), "quantile_rows"):
        mismatches.append("Manifest quantile-row count differs")
    if report.reported_at_or_below_threshold != _integer(
        case.get("reported_at_or_below_threshold"), "reported_at_or_below_threshold"
    ):
        mismatches.append("Manifest threshold count differs")
    return mismatches
