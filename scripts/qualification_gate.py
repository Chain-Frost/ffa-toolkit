"""Fail-closed WP09 legacy conformance and release-readiness checker.

This is a repository verification tool, not the supported ``ffa`` user CLI.
It does not execute the Flike binary, calculate flood estimates, or publish artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import cast

PROFILE = "tuflow-bmt-flike-5.0.306.0-file3.10"
SHA256 = re.compile(r"[a-f0-9]{64}\Z")
COMMIT = re.compile(r"[a-f0-9]{40}\Z")
REQUIRED_GATES = frozenset(
    {
        "ruff",
        "ruff_format",
        "strict_pyright",
        "unit_and_integration_tests",
        "wheel_and_sdist",
        "installed_cli_linux_py314",
        "installed_cli_windows_py314",
        "legacy_cli_replay",
        "offline_method_documentation",
        "independent_statistical_review",
        "no_unresolved_correctness_defects",
        "owner_release_authorisation",
    }
)
EVIDENCE_KINDS = frozenset(
    {
        "independent_math",
        "legacy_comparison",
        "synthetic_repeatability",
        "method_documentation",
        "independent_review",
        "installed_replay",
        "target_support",
        *REQUIRED_GATES,
    }
)
AXES = ("family", "estimator", "observations", "prior", "zero_threshold", "rating_error", "output")
MINIMUM_LEGACY_ROWS = frozenset(
    {
        "KERNEL-LOGNORMAL",
        "KERNEL-LP3",
        "KERNEL-GUMBEL",
        "KERNEL-GEV",
        "KERNEL-GENERALISED-PARETO",
        "L-MOMENTS-LOGNORMAL",
        "L-MOMENTS-LP3",
        "L-MOMENTS-GUMBEL",
        "L-MOMENTS-GEV",
        "L-MOMENTS-GENERALISED-PARETO",
        "GEV-LH-1",
        "GEV-LH-2",
        "GEV-LH-3",
        "GEV-LH-4",
        "GEV-LH-OPTIMISED",
        "BAYES-LP3-EXACT",
        "BAYES-LP3-THRESHOLD",
        "BAYES-GEV-EXACT",
        "BAYES-LOGNORMAL",
        "BAYES-GUMBEL",
        "BAYES-GENERALISED-PARETO",
        "OBS-EXACT",
        "LIKELIHOOD-CENSORED",
        "LIKELIHOOD-HISTORICAL",
        "LIKELIHOOD-EXCLUDED",
        "PRIOR-SUPPLIED",
        "PRIOR-REGIONAL",
        "PRIOR-NONE",
        "PRIOR-DEFAULT",
        "RATING-OFF",
        "RATING-ON",
        "MGB-LOW-FLOW",
        "ZERO-LOGNORMAL",
        "ZERO-LP3",
        "OUTPUT-EXPECTED-PARAMETER",
        "OUTPUT-EXPECTED-PROBABILITY",
        "OUTPUT-INTERVALS",
        "OUTPUT-PLOT-POSITIONS",
        "REPORT-CLI",
        "WP09-RELEASE-GATE",
    }
)
REQUIRED_SUPPORT_QUESTIONS = frozenset(
    {
        "L_SUPPORTED_FAMILIES",
        "BAYES_SUPPORTED_FAMILIES",
        "OBS_FAMILY_COMBINATIONS",
        "PRIORS_AND_DEFAULTS",
        "RATING_ERROR_OPTIONS",
        "NATIVE_RUN_CUSTODY",
        "OUTPUT_AND_CONFIDENCE_OPTIONS",
        "FAILURE_HANDLING",
    }
)


class InvalidLedgerError(ValueError):
    """The ledger is malformed; no release decision is possible."""


def _object(value: object, where: str) -> dict[str, object]:
    if not isinstance(value, dict):
        msg = f"{where} must be an object with string keys"
        raise InvalidLedgerError(msg)
    mapping = cast("dict[object, object]", value)
    if not all(isinstance(key, str) for key in mapping):
        msg = f"{where} must be an object with string keys"
        raise InvalidLedgerError(msg)
    return cast("dict[str, object]", mapping)


def _array(value: object, where: str) -> list[object]:
    if not isinstance(value, list):
        msg = f"{where} must be an array"
        raise InvalidLedgerError(msg)
    return cast("list[object]", value)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise InvalidLedgerError(message)


def _string(value: object, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        msg = f"{where} must be a nonempty string"
        raise InvalidLedgerError(msg)
    return value


def _unique_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            msg = f"duplicate JSON key: {key}"
            raise InvalidLedgerError(msg)
        result[key] = value
    return result


def _load(path: Path) -> dict[str, object]:
    try:
        raw: object = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_pairs)
    except (OSError, ValueError) as exc:
        msg = f"cannot read {path}: {exc}"
        raise InvalidLedgerError(msg) from exc
    return _object(raw, str(path))


def _records(value: object, where: str) -> dict[str, dict[str, object]]:
    by_id: dict[str, dict[str, object]] = {}
    for index, item in enumerate(_array(value, where)):
        record = _object(item, f"{where}[{index}]")
        identifier = _string(record.get("id"), f"{where}[{index}].id")
        if identifier in by_id:
            msg = f"duplicate {where} ID: {identifier}"
            raise InvalidLedgerError(msg)
        by_id[identifier] = record
    return by_id


def _is_sha(value: object, pattern: re.Pattern[str]) -> bool:
    return isinstance(value, str) and pattern.fullmatch(value) is not None


def _is_url(value: object) -> bool:
    return isinstance(value, str) and value.startswith("https://") and len(value) > len("https://")


def _finite_number(value: object) -> float | None:
    if isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value):
        return float(value)
    return None


def numeric_comparison_error(item: object) -> str | None:
    """Check an explicitly justified output-specific comparison; do not infer a universal tolerance."""
    comparison = _object(item, "numeric comparison")
    expected = _finite_number(comparison.get("expected"))
    actual = _finite_number(comparison.get("actual"))
    if expected is None or actual is None:
        return "expected/actual must be finite numbers"
    mode = comparison.get("mode")
    if mode == "deterministic":
        abs_tol = _finite_number(comparison.get("absolute_tolerance"))
        rel_tol = _finite_number(comparison.get("relative_tolerance"))
        if abs_tol is None or rel_tol is None or abs_tol < 0 or rel_tol < 0:
            return "deterministic tolerance must specify nonnegative absolute and relative parts"
        limit = abs_tol + rel_tol * abs(expected)
    elif mode == "printed_rounding":
        resolution = _finite_number(comparison.get("last_printed_unit"))
        if resolution is None or resolution <= 0:
            return "printed precision must specify a positive last-printed-unit resolution"
        limit = 0.5 * resolution
    elif mode == "stochastic":
        ref_se = _finite_number(comparison.get("reference_standard_error"))
        new_se = _finite_number(comparison.get("candidate_standard_error"))
        multiplier = _finite_number(comparison.get("justified_standard_error_multiplier"))
        ref_runs = comparison.get("reference_repeats")
        new_runs = comparison.get("candidate_repeats")
        if (
            ref_se is None
            or new_se is None
            or multiplier is None
            or ref_se < 0
            or new_se < 0
            or multiplier <= 0
            or not isinstance(ref_runs, int)
            or isinstance(ref_runs, bool)
            or ref_runs < 2
            or not isinstance(new_runs, int)
            or isinstance(new_runs, bool)
            or new_runs < 2
        ):
            return "stochastic comparison needs nonnegative standard errors, justified multiplier and repeats >= 2"
        limit = multiplier * math.hypot(ref_se, new_se)
    else:
        return "unrecognised numerical comparison mode"
    if not math.isfinite(limit):
        return "comparison tolerance overflowed"
    if abs(actual - expected) > limit:
        return f"difference {abs(actual - expected):.9g} exceeds declared tolerance {limit:.9g}"
    return None


def _comparison_list_error(value: object) -> str | None:
    if not isinstance(value, list) or not value:
        return "qualified comparison contains no numeric observations"
    for item in cast("list[object]", value):
        why = numeric_comparison_error(item)
        if why:
            return why
    return None


def _legacy_comparison_error(entry: dict[str, object], admitted_runs: dict[str, dict[str, object]]) -> str | None:
    if entry.get("pairing_status") != "qualified_reference":
        return "reference pairing is not qualified"
    if entry.get("legacy_program_version") != "5.0.306.0" or entry.get("legacy_file_version") != "3.10":
        return "reference executable/file version not pinned to the baseline"
    if not all(_is_sha(entry.get(key), SHA256) for key in ("input_sha256", "settings_sha256", "output_sha256")):
        return "missing reference input/settings/output hashes"
    if not _is_url(entry.get("precision_rationale_url")):
        return "missing output-specific tolerance/precision justification"
    case_id = entry.get("case_id")
    run = admitted_runs.get(case_id) if isinstance(case_id, str) else None
    if run is None:
        return "reference case has not been independently admitted"
    if entry.get("compatibility_profile") != run.get("compatibility_profile"):
        return "reference compatibility profile differs from admitted native run"
    if any(entry.get(key) != run.get(key) for key in ("legacy_program_version", "legacy_file_version")):
        return "reference executable/file version differs from admitted native run"
    mismatched_hashes = [
        key for key in ("input_sha256", "settings_sha256", "output_sha256") if entry.get(key) != run.get(key)
    ]
    if mismatched_hashes:
        return f"reference custody mismatch against admitted native run: {', '.join(mismatched_hashes)}"
    try:
        return _comparison_list_error(entry.get("comparisons"))
    except InvalidLedgerError as exc:
        return str(exc)


def _repeatability_error(entry: dict[str, object]) -> str | None:
    runs = entry.get("repeated_runs")
    if not isinstance(runs, int) or isinstance(runs, bool) or runs < 2:
        return "fewer than two documented synthetic runs"
    if not _is_url(entry.get("precision_rationale_url")):
        return "no stochastic precision justification"
    return None


def _check_passed_evidence(
    entry: dict[str, object], kind: str, admitted_runs: dict[str, dict[str, object]], row_id: str | None = None
) -> str | None:
    """Return why a purportedly passing evidence record cannot support qualification."""
    if entry.get("status") != "passed" or entry.get("kind") != kind:
        return "evidence status/kind does not match a passed check"
    if row_id is not None and entry.get("matrix_row_id") != row_id:
        return "evidence was not produced for this exact combination"
    if not _is_url(entry.get("evidence_url")) or not _is_sha(entry.get("artifact_sha256"), SHA256):
        return "missing immutable evidence URL or SHA-256"
    if not _is_sha(entry.get("source_commit"), COMMIT):
        return "missing exact source commit"
    if kind == "legacy_comparison":
        return _legacy_comparison_error(entry, admitted_runs)
    if kind == "synthetic_repeatability":
        return _repeatability_error(entry)
    if kind in {"independent_math", "independent_review"} and not _is_url(entry.get("independent_source_url")):
        return "independent basis or review not referenced"
    if kind.startswith("installed_cli_") and entry.get("installed_outside_checkout") is not True:
        return "installed CLI was not checked outside the source checkout"
    return None


@dataclass(frozen=True, slots=True)
class GateResult:
    errors: tuple[str, ...]
    blockers: tuple[str, ...]
    total_rows: int
    qualified_rows: int
    gate_passes: int

    @property
    def ready(self) -> bool:
        return not self.errors and not self.blockers

    def markdown(self) -> str:
        label = "READY FOR OWNER RELEASE DECISION" if self.ready else "BLOCKED — NOT RELEASE-READY"
        lines = [
            "# WP09 legacy conformance and release-readiness report",
            "",
            f"**Decision: {label}.** No publication is performed by this tool.",
            "",
            f"- Compatibility target: `{PROFILE}`",
            f"- Qualified matrix rows: **{self.qualified_rows}/{self.total_rows}**",
            f"- Passed release checks: **{self.gate_passes}/{len(REQUIRED_GATES)}**",
            f"- Ledger errors: **{len(self.errors)}**; release blockers: **{len(self.blockers)}**",
            "",
            "## Errors (invalid qualification evidence)",
            "",
        ]
        lines.extend(f"- {value}" for value in self.errors)
        if not self.errors:
            lines.append("- None")
        lines.extend(["", "## Unmet release conditions", ""])
        lines.extend(f"- {value}" for value in self.blockers)
        if not self.blockers:
            lines.append("- None")
        lines.extend(
            [
                "",
                (
                    "Only a frozen, owner-approved, fully evidenced legacy matrix can pass. "
                    "WMAwater extensions and GUI features are excluded from this MVP."
                ),
                "",
            ]
        )
        return "\n".join(lines)


def evaluate(root: Path) -> GateResult:  # noqa: C901  # Cross-ledger gate audits all independent release conditions.
    """Validate schema and evidence independently of the implementation under test."""
    errors: list[str] = []
    blockers: list[str] = []
    qualified = 0
    gate_passes = 0
    total_rows = 0
    try:
        matrix = _load(root / "qualification/legacy-matrix.json")
        ledger = _load(root / "qualification/evidence.json")
        registry = _load(root / "docs/method-registry.json")
        candidates = _load(root / "reference/legacy-candidates/manifest.json")
        _require(
            matrix.get("schema_version") == 1 and ledger.get("schema_version") == 1,
            "unsupported matrix/evidence schema version",
        )
        _require(
            matrix.get("compatibility_profile") == PROFILE
            and registry.get("compatibility_profile") == PROFILE
            and ledger.get("compatibility_profile") == PROFILE,
            "compatibility profile mismatches source method register or evidence ledger",
        )
        _require(candidates.get("schema_version") == 1, "unsupported candidate manifest schema")
        archived_candidate_ids: set[str] = set()
        for item in _array(candidates.get("cases"), "reference cases"):
            case = _object(item, "reference case")
            case_id = _string(case.get("case_id"), "reference case ID")
            _require(case_id not in archived_candidate_ids, f"duplicate reference case: {case_id}")
            archived_candidate_ids.add(case_id)
            _string(case.get("pairing_status"), "reference pairing")
            if (
                case.get("pairing_status") == "qualified_reference"
                or case.get("status") in ("qualified", "qualified_reference")
                or case.get("flike_qualified") is True
            ):
                errors.append(f"{case_id}: archived candidate manifest cannot establish qualified native status")
        # Only a separately identified, independently admitted native rerun can
        # qualify a reference. Never promote a staged candidate via its manifest.
        admitted_runs: dict[str, dict[str, object]] = {}
        reference_runs = _records(ledger.get("qualified_reference_cases"), "qualified_reference_cases")
        for run_id, run in reference_runs.items():
            if run_id in archived_candidate_ids:
                errors.append(f"{run_id}: qualified reference ID collides with source candidate")
            elif (
                run.get("legacy_program_version") != "5.0.306.0"
                or run.get("legacy_file_version") != "3.10"
                or run.get("compatibility_profile") != PROFILE
                or not all(
                    _is_sha(run.get(key), SHA256)
                    for key in ("legacy_executable_sha256", "input_sha256", "settings_sha256", "output_sha256")
                )
                or not _is_url(run.get("native_run_log_url"))
                or not _is_url(run.get("independent_admission_url"))
            ):
                errors.append(f"{run_id}: incomplete source/version/configuration or independent admission custody")
            else:
                admitted_runs[run_id] = run
        methods = {
            _string(method.get("method_id"), "method ID"): method.get("scope")
            for item in _array(registry.get("methods"), "methods")
            for method in [_object(item, "method")]
        }
        evidence = _records(ledger.get("evidence"), "evidence")
        release_source_commit = matrix.get("release_source_commit")
        if matrix.get("matrix_frozen") is True and not _is_sha(release_source_commit, COMMIT):
            errors.append("frozen matrix lacks a pinned release source commit")
        for proof_id, proof in evidence.items():
            if (
                proof.get("status") == "passed"
                and _is_sha(release_source_commit, COMMIT)
                and proof.get("source_commit") != release_source_commit
            ):
                errors.append(f"{proof_id}: evidence was produced against a different source revision")
        rows = _records(matrix.get("rows"), "rows")
        missing_baseline = MINIMUM_LEGACY_ROWS - set(rows)
        if missing_baseline:
            errors.append(f"Minimum legacy coverage rows missing: {', '.join(sorted(missing_baseline))}")
        quality = _records(ledger.get("quality_gates"), "quality_gates")
        total_rows = len(rows)
        included_methods: set[str] = set()
        for identifier, row in rows.items():
            method_id = _string(row.get("method_id"), f"{identifier}.method_id")
            if methods.get(method_id) != "mvp":
                errors.append(f"{identifier}: method {method_id} is missing from MVP register")
            included_methods.add(method_id)
            for axis in AXES:
                _string(row.get(axis), f"{identifier}.{axis}")
            status = row.get("status")
            admission = row.get("admission")
            if status not in {"pending", "blocked", "qualified", "not_supported"}:
                errors.append(f"{identifier}: invalid status {status!r}")
                continue
            if admission not in {"required", "unresolved", "target_unsupported"}:
                errors.append(f"{identifier}: invalid admission {admission!r}")
                continue
            required_checks = [
                _string(x, f"{identifier}.required_check")
                for x in _array(row.get("required_checks"), f"{identifier}.required_checks")
            ]
            if not required_checks or len(set(required_checks)) != len(required_checks):
                errors.append(f"{identifier}: required checks missing or duplicated")
            mandatory = {"legacy_comparison", "method_documentation", "independent_review"}
            if method_id != "legacy-report-and-method-guidance":
                mandatory.add("independent_math")
            if row.get("estimator") == "bayesian_importance_sampling":
                mandatory.add("synthetic_repeatability")
            if method_id in {"legacy-report-and-method-guidance", "legacy-conformance-verification"}:
                mandatory.add("installed_replay")
            if not mandatory.issubset(required_checks):
                errors.append(
                    f"{identifier}: mandatory checks omitted: {', '.join(sorted(mandatory - set(required_checks)))}"
                )
            errors.extend(
                f"{identifier}: unknown check {check}" for check in required_checks if check not in EVIDENCE_KINDS
            )
            proofs = _object(row.get("evidence"), f"{identifier}.evidence")
            if status == "qualified":
                if admission != "required" or any(row.get(axis) == "unresolved" for axis in AXES):
                    errors.append(f"{identifier}: unresolved/unsupported combination cannot be qualified")
                missing = [check for check in required_checks if not isinstance(proofs.get(check), str)]
                if missing:
                    errors.append(f"{identifier}: qualification missing evidence for {', '.join(missing)}")
                for check in required_checks:
                    proof_id = proofs.get(check)
                    if not isinstance(proof_id, str):
                        continue
                    proof = evidence.get(proof_id)
                    why = (
                        _check_passed_evidence(proof, check, admitted_runs, identifier)
                        if proof is not None
                        else "evidence ID not found"
                    )
                    if why:
                        errors.append(f"{identifier}/{check}: {why}")
                if not missing and not any(error.startswith((f"{identifier}:", f"{identifier}/")) for error in errors):
                    qualified += 1
            elif status == "not_supported":
                proof_id = proofs.get("target_support")
                proof = evidence.get(proof_id) if isinstance(proof_id, str) else None
                why = (
                    _check_passed_evidence(proof, "target_support", admitted_runs, identifier)
                    if proof is not None
                    else "support proof missing"
                )
                if admission != "target_unsupported" or why or not _is_url(row.get("scope_approval_url")):
                    errors.append(f"{identifier}: excluded target feature lacks source proof and scope approval")
            else:
                blockers.append(f"{identifier}: {status} ({admission})")
            if admission == "unresolved":
                blockers.append(f"{identifier}: legacy support for this combination is unresolved")
        missing_methods = {name for name, scope in methods.items() if scope == "mvp"} - included_methods
        if missing_methods:
            errors.append(f"MVP method register entries missing matrix rows: {', '.join(sorted(missing_methods))}")
        if matrix.get("matrix_frozen") is not True:
            blockers.append("Scope matrix is not frozen")
        if not _is_url(matrix.get("scope_approval_url")):
            blockers.append("Owner approval of frozen legacy scope is not recorded")
        questions = _records(matrix.get("unresolved_support"), "unresolved_support")
        missing_questions = REQUIRED_SUPPORT_QUESTIONS - set(questions)
        if missing_questions:
            errors.append(f"Required legacy support questions missing: {', '.join(sorted(missing_questions))}")
        for question_id, question in questions.items():
            if question.get("status") != "resolved":
                blockers.append(f"Unresolved target support: {question_id}")
            elif not _is_url(question.get("evidence_url")):
                errors.append(f"{question_id}: support resolution lacks source evidence URL")
        if frozenset(quality) != REQUIRED_GATES:
            errors.append(
                f"Quality gate IDs differ: missing={sorted(REQUIRED_GATES - set(quality))}; extras={sorted(set(quality) - REQUIRED_GATES)}"
            )
        for gate_id in REQUIRED_GATES:
            gate = quality.get(gate_id)
            if gate is None:
                continue
            if gate.get("status") == "passed":
                proof_id = gate.get("evidence_id")
                proof = evidence.get(proof_id) if isinstance(proof_id, str) else None
                why = (
                    _check_passed_evidence(proof, gate_id, admitted_runs)
                    if proof is not None
                    else "evidence ID not found"
                )
                if why:
                    errors.append(f"{gate_id}: {why}")
                else:
                    gate_passes += 1
            elif gate.get("status") in {"pending", "failed", "blocked"}:
                blockers.append(f"Release check {gate_id}: {gate['status']}")
            else:
                errors.append(f"Release check {gate_id}: invalid status")
    except InvalidLedgerError as exc:
        errors.append(str(exc))
    return GateResult(tuple(errors), tuple(sorted(set(blockers))), total_rows, qualified, gate_passes)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate WP09 evidence; release mode blocks unqualified MVPs")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--validate-only", action="store_true", help="check ledger integrity without requiring release approval"
    )
    parser.add_argument("--report", type=Path, help="write a human-readable readiness report")
    args = parser.parse_args(argv)
    result = evaluate(args.root)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(result.markdown(), encoding="utf-8")
    print(result.markdown())
    if result.errors:
        return 2
    if not args.validate_only and not result.ready:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
