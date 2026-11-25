"""Pinned evidence identity, not a claim of implemented Flike conformance."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CompatibilityProfile:
    """Observed vendor/version identity; method and package versions are separate."""

    identifier: str
    product: str
    legacy_program_version: str
    legacy_file_version: str
    evidence_status: str


LEGACY_FLIKE_PROFILE = CompatibilityProfile(
    identifier="tuflow-bmt-flike-5.0.306.0-file3.10",
    product="TUFLOW/BMT Flike",
    legacy_program_version="5.0.306.0",
    legacy_file_version="3.10",
    evidence_status="observed_artifact; input-output pairings unverified",
)
