"""Proof-of-policy tests using deliberately synthetic evidence, never native Flike benchmarks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from scripts.qualification_gate import PROFILE, REQUIRED_GATES, evaluate, main, numeric_comparison_error

SHA = "a" * 64
COMMIT = "b" * 40
URL = "https://example.org/synthetic-evidence"
SOURCE = Path(__file__).resolve().parents[1]


def _read(path: Path) -> dict[str, object]:
    return cast("dict[str, object]", json.loads(path.read_text(encoding="utf-8")))


def _write(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """Create a standalone matrix/references fixture without using private source material."""
    for relative in ("qualification/legacy-matrix.json", "qualification/evidence.json"):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((SOURCE / relative).read_bytes())
    matrix = _read(tmp_path / "qualification/legacy-matrix.json")
    ids: set[str] = {cast("str", row["method_id"]) for row in cast("list[dict[str, object]]", matrix["rows"])}
    registry: dict[str, object] = {
        "compatibility_profile": matrix["compatibility_profile"],
        "methods": [{"method_id": mid, "scope": "mvp"} for mid in sorted(ids)]
        + [{"method_id": "future-procedure", "scope": "future"}],
    }
    method_file = tmp_path / "docs/method-registry.json"
    method_file.parent.mkdir(parents=True, exist_ok=True)
    _write(method_file, registry)
    manifest_file = tmp_path / "reference/legacy-candidates/manifest.json"
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    _write(
        manifest_file,
        {
            "schema_version": 1,
            "cases": [
                {"case_id": "case-a", "pairing_status": "candidate_settings_unverified"},
                {"case_id": "case-b", "pairing_status": "candidate_settings_unverified"},
            ],
        },
    )
    return tmp_path


def test_pending_baseline_is_valid_but_not_release_ready(repo: Path) -> None:
    outcome = evaluate(repo)
    assert not outcome.errors
    assert not outcome.ready
    assert outcome.qualified_rows == 0
    assert outcome.total_rows == 40
    assert outcome.gate_passes == 0
    assert main(["--root", str(repo), "--validate-only"]) == 0
    assert main(["--root", str(repo)]) == 1


def test_candidate_legacy_case_cannot_be_promoted(repo: Path) -> None:
    matrix_path = repo / "qualification/legacy-matrix.json"
    evidence_path = repo / "qualification/evidence.json"
    matrix = _read(matrix_path)
    rows = cast("list[dict[str, object]]", matrix["rows"])
    row = rows[0]
    row["status"] = "qualified"
    row["evidence"] = {check: f"{row['id']}-{check}" for check in cast("list[str]", row["required_checks"])}
    ledger = _read(evidence_path)
    ledger["evidence"] = [
        _proof(check, cast("str", row["id"]), "case-a") for check in cast("list[str]", row["required_checks"])
    ]
    _write(matrix_path, matrix)
    _write(evidence_path, ledger)
    result = evaluate(repo)
    assert any("reference case has not been independently admitted" in problem for problem in result.errors)
    assert result.qualified_rows == 0


def _proof(kind: str, row_id: str | None = None, case_id: str = "verified-new") -> dict[str, object]:
    proof: dict[str, object] = {
        "id": f"proof-{kind}" if row_id is None else f"{row_id}-{kind}",
        "kind": kind,
        "status": "passed",
        "evidence_url": URL,
        "artifact_sha256": SHA,
        "source_commit": COMMIT,
    }
    if row_id is not None:
        proof["matrix_row_id"] = row_id
    if kind == "legacy_comparison":
        proof.update(
            {
                "case_id": case_id,
                "pairing_status": "qualified_reference",
                "compatibility_profile": PROFILE,
                "legacy_program_version": "5.0.306.0",
                "legacy_file_version": "3.10",
                "input_sha256": SHA,
                "settings_sha256": SHA,
                "output_sha256": SHA,
                "precision_rationale_url": URL,
                "comparisons": [
                    {"mode": "printed_rounding", "expected": 10.4, "actual": 10.4, "last_printed_unit": 0.1}
                ],
            }
        )
    if kind == "synthetic_repeatability":
        proof.update({"repeated_runs": 3, "precision_rationale_url": URL})
    if kind in {"independent_math", "independent_review"}:
        proof["independent_source_url"] = URL
    if kind.startswith("installed_cli_"):
        proof["installed_outside_checkout"] = True
    return proof


def _prepare_synthetic_ready_gate(repo: Path) -> None:
    """Construct an entirely synthetic, independently admitted reference control."""
    matrix_file = repo / "qualification/legacy-matrix.json"
    ledger_file = repo / "qualification/evidence.json"
    matrix = _read(matrix_file)
    ledger = _read(ledger_file)
    matrix["matrix_frozen"] = True
    matrix["scope_approval_url"] = URL
    matrix["release_source_commit"] = COMMIT
    for question in cast("list[dict[str, object]]", matrix["unresolved_support"]):
        question.update({"status": "resolved", "evidence_url": URL})
    evidence: list[dict[str, object]] = []
    for row in cast("list[dict[str, object]]", matrix["rows"]):
        row_id = cast("str", row["id"])
        row["admission"] = "required"
        row["status"] = "qualified"
        for axis in ("family", "estimator", "observations", "prior", "zero_threshold", "rating_error", "output"):
            if row[axis] == "unresolved":
                row[axis] = "documented_exact_option"
        checks = cast("list[str]", row["required_checks"])
        row["evidence"] = {check: f"{row_id}-{check}" for check in checks}
        evidence.extend(_proof(check, row_id) for check in checks)
    for gate_id in REQUIRED_GATES:
        proof = _proof(gate_id)
        evidence.append(proof)
    ledger["qualified_reference_cases"] = [
        {
            "id": "verified-new",
            "compatibility_profile": PROFILE,
            "legacy_program_version": "5.0.306.0",
            "legacy_file_version": "3.10",
            "legacy_executable_sha256": SHA,
            "input_sha256": SHA,
            "settings_sha256": SHA,
            "output_sha256": SHA,
            "native_run_log_url": URL,
            "independent_admission_url": URL,
        }
    ]
    ledger["evidence"] = evidence
    for check in cast("list[dict[str, object]]", ledger["quality_gates"]):
        check.update({"status": "passed", "evidence_id": f"proof-{check['id']}"})
    _write(matrix_file, matrix)
    _write(ledger_file, ledger)


def test_signed_and_evidenced_synthetic_gate_can_be_ready(repo: Path) -> None:
    """Positive test proves the gate isn't hard-coded to reject everything."""
    _prepare_synthetic_ready_gate(repo)
    ready = evaluate(repo)
    assert ready.errors == ()
    assert ready.blockers == ()
    assert ready.total_rows == ready.qualified_rows
    assert ready.gate_passes == len(REQUIRED_GATES)
    assert main(["--root", str(repo)]) == 0


def test_missing_registry_method_is_invalid(repo: Path) -> None:
    file = repo / "docs/method-registry.json"
    registry = _read(file)
    cast("list[dict[str, object]]", registry["methods"]).append({"method_id": "new-required", "scope": "mvp"})
    _write(file, registry)
    assert any("new-required" in error for error in evaluate(repo).errors)


def test_duplicate_ledger_key_fails_closed(repo: Path) -> None:
    path = repo / "qualification/evidence.json"
    path.write_text('{"schema_version":1,"schema_version":1}', encoding="utf-8")
    result = evaluate(repo)
    assert any("duplicate JSON key" in error for error in result.errors)
    assert main(["--root", str(repo), "--validate-only"]) == 2


def test_unsupported_method_or_removed_required_proof_fails(repo: Path) -> None:
    path = repo / "qualification/legacy-matrix.json"
    matrix = _read(path)
    row = cast("list[dict[str, object]]", matrix["rows"])[0]
    row["method_id"] = "future-procedure"
    row["required_checks"] = ["method_documentation"]
    _write(path, matrix)
    assert any("future-procedure" in error for error in evaluate(repo).errors)
    assert any("mandatory checks omitted" in error for error in evaluate(repo).errors)


def test_report_is_written_but_does_not_publish(repo: Path) -> None:
    report = repo / "out/readiness.md"
    assert main(["--root", str(repo), "--report", str(report)]) == 1
    content = report.read_text(encoding="utf-8")
    assert "BLOCKED" in content
    assert "0/40" in content
    assert "owner_release_authorisation" in content


def test_resolved_support_requires_source(repo: Path) -> None:
    path = repo / "qualification/legacy-matrix.json"
    matrix = _read(path)
    cast("list[dict[str, object]]", matrix["unresolved_support"])[0]["status"] = "resolved"
    _write(path, matrix)
    assert any("source evidence URL" in error for error in evaluate(repo).errors)


def test_qualified_reference_requires_separate_native_custody(repo: Path) -> None:
    path = repo / "qualification/evidence.json"
    ledger = _read(path)
    ledger["qualified_reference_cases"] = [{"id": "fake-native"}]
    _write(path, ledger)
    assert any("incomplete source/version/configuration" in error for error in evaluate(repo).errors)


def test_quality_pass_does_not_accept_candidate_evidence(repo: Path) -> None:
    path = repo / "qualification/evidence.json"
    ledger = _read(path)
    cast("list[dict[str, object]]", ledger["quality_gates"])[0].update(
        {"status": "passed", "evidence_id": "CANDIDATE-CASE-A"}
    )
    _write(path, ledger)
    assert any("evidence status/kind" in error for error in evaluate(repo).errors)


@pytest.mark.parametrize(
    ("comparison", "failure"),
    [
        (
            {
                "mode": "deterministic",
                "expected": 100,
                "actual": 100.1,
                "absolute_tolerance": 0.2,
                "relative_tolerance": 0.0,
            },
            False,
        ),
        (
            {
                "mode": "deterministic",
                "expected": 100,
                "actual": 101,
                "absolute_tolerance": 0.2,
                "relative_tolerance": 0.0,
            },
            True,
        ),
        ({"mode": "printed_rounding", "expected": 12.34, "actual": 12.344, "last_printed_unit": 0.01}, False),
        ({"mode": "printed_rounding", "expected": 12.34, "actual": 12.35, "last_printed_unit": 0.01}, True),
        ({"mode": "printed_rounding", "expected": 12.34, "actual": 12.34, "last_printed_unit": 0}, True),
        (
            {
                "mode": "stochastic",
                "expected": 100,
                "actual": 101,
                "reference_standard_error": 1,
                "candidate_standard_error": 1,
                "justified_standard_error_multiplier": 2,
                "reference_repeats": 3,
                "candidate_repeats": 3,
            },
            False,
        ),
        (
            {
                "mode": "stochastic",
                "expected": 100,
                "actual": 110,
                "reference_standard_error": 1,
                "candidate_standard_error": 1,
                "justified_standard_error_multiplier": 2,
                "reference_repeats": 3,
                "candidate_repeats": 3,
            },
            True,
        ),
        (
            {
                "mode": "stochastic",
                "expected": 100,
                "actual": 100,
                "reference_standard_error": 1,
                "candidate_standard_error": 1,
                "justified_standard_error_multiplier": 2,
                "reference_repeats": 1,
                "candidate_repeats": 3,
            },
            True,
        ),
    ],
)
def test_numerical_policies_do_not_use_one_universal_tolerance(comparison: dict[str, object], failure: bool) -> None:
    assert (numeric_comparison_error(comparison) is not None) == failure


def test_release_source_revision_is_pinned(repo: Path) -> None:
    matrix_file = repo / "qualification/legacy-matrix.json"
    matrix = _read(matrix_file)
    matrix["matrix_frozen"] = True
    _write(matrix_file, matrix)
    assert any("release source commit" in error for error in evaluate(repo).errors)


def test_mismatched_evidence_profile_is_invalid(repo: Path) -> None:
    path = repo / "qualification/evidence.json"
    ledger = _read(path)
    ledger["compatibility_profile"] = "unrelated-vendor-profile"
    _write(path, ledger)
    assert any("compatibility profile mismatches" in error for error in evaluate(repo).errors)


def test_minimum_coverage_cannot_be_silently_deleted(repo: Path) -> None:
    path = repo / "qualification/legacy-matrix.json"
    matrix = _read(path)
    cast("list[dict[str, object]]", matrix["rows"]).pop(0)
    _write(path, matrix)
    assert any("Minimum legacy coverage rows missing" in error for error in evaluate(repo).errors)


def test_required_support_questions_cannot_be_deleted(repo: Path) -> None:
    path = repo / "qualification/legacy-matrix.json"
    matrix = _read(path)
    cast("list[dict[str, object]]", matrix["unresolved_support"]).pop(0)
    _write(path, matrix)
    assert any("Required legacy support questions missing" in error for error in evaluate(repo).errors)


def test_real_wp09_evidence_remains_valid_but_blocked() -> None:
    """The repository's actual ledger is not replaced by synthetic approvals."""
    result = evaluate(SOURCE)
    assert not result.errors
    assert not result.ready
    assert result.qualified_rows == 0
    assert result.total_rows == 40
    assert result.gate_passes == 0


@pytest.mark.parametrize(
    "tampered_hashes",
    [
        pytest.param(("input_sha256",), id="input"),
        pytest.param(("settings_sha256",), id="settings"),
        pytest.param(("output_sha256",), id="output"),
        pytest.param(("input_sha256", "settings_sha256", "output_sha256"), id="all-three"),
    ],
)
def test_native_comparison_hashes_must_match_the_admitted_run(repo: Path, tampered_hashes: tuple[str, ...]) -> None:
    _prepare_synthetic_ready_gate(repo)
    path = repo / "qualification/evidence.json"
    ledger = _read(path)
    proofs = cast("list[dict[str, object]]", ledger["evidence"])
    comparison = next(proof for proof in proofs if proof.get("kind") == "legacy_comparison")
    for field in tampered_hashes:
        comparison[field] = "f" * 64  # Different but syntactically valid SHA-256.
    _write(path, ledger)
    result = evaluate(repo)
    custody = next(error for error in result.errors if "reference custody mismatch" in error)
    assert all(field in custody for field in tampered_hashes)
    assert result.qualified_rows == result.total_rows - 1
    assert not result.ready
    assert main(["--root", str(repo), "--validate-only"]) == 2


@pytest.mark.parametrize(
    ("field", "promotion"),
    [
        pytest.param("pairing_status", "qualified_reference", id="pairing-status"),
        pytest.param("status", "qualified_reference", id="candidate-status"),
        pytest.param("flike_qualified", True, id="local-qualification-flag"),
    ],
)
def test_archived_candidate_cannot_gain_admission_by_manifest_promotion(
    repo: Path, field: str, promotion: object
) -> None:
    _prepare_synthetic_ready_gate(repo)
    path = repo / "qualification/evidence.json"
    ledger = _read(path)
    ledger["qualified_reference_cases"] = []
    for proof in cast("list[dict[str, object]]", ledger["evidence"]):
        if proof.get("kind") == "legacy_comparison":
            proof["case_id"] = "case-a"
    _write(path, ledger)
    manifest_path = repo / "reference/legacy-candidates/manifest.json"
    manifest = _read(manifest_path)
    cast("list[dict[str, object]]", manifest["cases"])[0][field] = promotion
    _write(manifest_path, manifest)
    result = evaluate(repo)
    assert any("archived candidate manifest" in error for error in result.errors)
    assert any("reference case has not been independently admitted" in error for error in result.errors)
    assert result.qualified_rows == 0
    assert not result.ready
    assert main(["--root", str(repo), "--validate-only"]) == 2


@pytest.mark.parametrize("case_id", ["rmc-arr-ex3", "lp3-uniform-law"])
def test_external_only_fixture_cannot_satisfy_native_admission(repo: Path, case_id: str) -> None:
    _prepare_synthetic_ready_gate(repo)
    path = repo / "qualification/evidence.json"
    ledger = _read(path)
    comparison = next(
        proof
        for proof in cast("list[dict[str, object]]", ledger["evidence"])
        if proof.get("kind") == "legacy_comparison"
    )
    comparison["case_id"] = case_id
    _write(path, ledger)
    result = evaluate(repo)
    assert any("reference case has not been independently admitted" in error for error in result.errors)
    assert result.qualified_rows == result.total_rows - 1


@pytest.mark.parametrize(
    "missing_field",
    [
        "compatibility_profile",
        "legacy_executable_sha256",
        "input_sha256",
        "settings_sha256",
        "output_sha256",
        "native_run_log_url",
        "independent_admission_url",
    ],
)
def test_incomplete_native_run_custody_cannot_qualify(repo: Path, missing_field: str) -> None:
    _prepare_synthetic_ready_gate(repo)
    path = repo / "qualification/evidence.json"
    ledger = _read(path)
    run = cast("list[dict[str, object]]", ledger["qualified_reference_cases"])[0]
    run.pop(missing_field)
    _write(path, ledger)
    result = evaluate(repo)
    assert any("incomplete source/version/configuration" in error for error in result.errors)
    assert any("reference case has not been independently admitted" in error for error in result.errors)
    assert result.qualified_rows == 0


def test_native_run_id_cannot_collide_with_archived_candidate(repo: Path) -> None:
    _prepare_synthetic_ready_gate(repo)
    path = repo / "qualification/evidence.json"
    ledger = _read(path)
    run = cast("list[dict[str, object]]", ledger["qualified_reference_cases"])[0]
    run["id"] = "case-a"
    for proof in cast("list[dict[str, object]]", ledger["evidence"]):
        if proof.get("kind") == "legacy_comparison":
            proof["case_id"] = "case-a"
    _write(path, ledger)
    result = evaluate(repo)
    assert any("qualified reference ID collides with source candidate" in error for error in result.errors)
    assert any("reference case has not been independently admitted" in error for error in result.errors)
    assert result.qualified_rows == 0


def test_duplicate_native_run_ids_are_rejected(repo: Path) -> None:
    _prepare_synthetic_ready_gate(repo)
    path = repo / "qualification/evidence.json"
    ledger = _read(path)
    runs = cast("list[dict[str, object]]", ledger["qualified_reference_cases"])
    runs.append(dict(runs[0]))
    _write(path, ledger)
    assert any("duplicate qualified_reference_cases ID" in error for error in evaluate(repo).errors)


@pytest.mark.parametrize("field", ["compatibility_profile", "legacy_program_version", "legacy_file_version"])
def test_comparison_profile_and_versions_must_match_the_admitted_run(repo: Path, field: str) -> None:
    _prepare_synthetic_ready_gate(repo)
    path = repo / "qualification/evidence.json"
    ledger = _read(path)
    comparison = next(
        proof
        for proof in cast("list[dict[str, object]]", ledger["evidence"])
        if proof.get("kind") == "legacy_comparison"
    )
    comparison[field] = "different-version-or-profile"
    _write(path, ledger)
    result = evaluate(repo)
    assert any("profile differs" in error or "version not pinned" in error for error in result.errors)
    assert result.qualified_rows == result.total_rows - 1


def test_evidence_artifact_hash_is_not_the_native_output_hash(repo: Path) -> None:
    _prepare_synthetic_ready_gate(repo)
    path = repo / "qualification/evidence.json"
    ledger = _read(path)
    comparison = next(
        proof
        for proof in cast("list[dict[str, object]]", ledger["evidence"])
        if proof.get("kind") == "legacy_comparison"
    )
    comparison["artifact_sha256"] = "e" * 64
    _write(path, ledger)
    result = evaluate(repo)
    assert result.errors == ()
    assert result.ready
