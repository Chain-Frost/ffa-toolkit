"""Ensure the archive verifier rejects path-based and filename-based evidence leakage."""

import pytest

from scripts.verify_distribution import check_member_names


@pytest.mark.parametrize(
    "filename",
    [
        "ffa_toolkit-0.0.0.dev0/reference/legacy-candidates/project.fld",
        "ffa_toolkit-0.0.0.dev0/reference/legacy-candidates/manifest.json",
        "ffa_toolkit-0.0.0.dev0/src/ffa_methods/data/source.7z",
        "ffa_methods/samples/case-b_Bayes_Out.txt",
        "ffa_methods/samples/AMS_case-a.csv",
        "ffa_methods/../outside",
    ],
)
def test_refuse_unqualified_evidence_in_distributions(filename: str) -> None:
    with pytest.raises(ValueError, match=r"disallowed archive member|Flike input"):
        check_member_names([filename])


def test_allow_documentation_and_typed_marker() -> None:
    check_member_names(
        [
            "ffa_methods/resources/references.md",
            "ffa_methods/resources/method-registry.json",
            "ffa_methods/py.typed",
        ]
    )
