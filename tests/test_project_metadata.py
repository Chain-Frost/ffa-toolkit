"""Constrain the development-only version and evidence profile to checked-in policy."""

import json
import tomllib
from pathlib import Path
from typing import cast

from ffa_methods import __version__
from ffa_methods.compatibility import LEGACY_FLIKE_PROFILE

ROOT = Path(__file__).resolve().parents[1]


def test_metadata_and_legacy_profile_agree_with_repository() -> None:
    with (ROOT / "pyproject.toml").open("rb") as stream:
        metadata = cast("dict[str, object]", tomllib.load(stream)["project"])
    assert metadata["version"] == __version__
    assert metadata["requires-python"] == ">=3.11"

    registry = cast("dict[str, object]", json.loads((ROOT / "docs/method-registry.json").read_text()))
    assert registry["compatibility_profile"] == LEGACY_FLIKE_PROFILE.identifier
    assert registry["implementation_status"] == "not_implemented"
