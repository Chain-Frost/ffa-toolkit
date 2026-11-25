"""Independent ordinary CLI integration and failure/custody tests (not native Flike)."""

import json
from hashlib import sha256
from math import isclose
from pathlib import Path
from typing import cast

import pytest

from ffa_methods.cli import main
from ffa_methods.deterministic_workflow import calculate_from_bytes


def _settings() -> dict[str, object]:
    source = Path(__file__).resolve().parents[1] / "docs/examples/ordinary-settings.json"
    return cast(dict[str, object], json.loads(source.read_bytes()))


def _encoded(values: dict[str, object]) -> bytes:
    return json.dumps(values).encode("utf-8")


@pytest.mark.parametrize(
    ("estimator", "extra"),
    [
        ("gumbel_lmoments", {}),
        ("lognormal_lmoments", {"reference_discharge": 2.0}),
        ("lp3_lmoments", {"reference_discharge": 2.0}),
        ("gp_lmoments", {"gp_location_mode": "fixed", "gp_location": 0.25}),
        ("gp_lmoments", {"gp_location_mode": "free"}),
    ],
)
def test_consistent_flow_unit_rescaling_preserves_fit(estimator: str, extra: dict[str, object]) -> None:
    original = {**_settings(), "estimator": estimator, **extra}
    changed = {**original, "discharge_unit": "scaled flow"}
    for key in ("reference_discharge", "gp_location"):
        if key in original:
            changed[key] = 8.0 * cast(float, original[key])
    a, _ = calculate_from_bytes(b"Year,AMS\n2000,1\n2001,2\n2002,4\n", _encoded(original))
    b, _ = calculate_from_bytes(b"Year,AMS\n2000,8\n2001,16\n2002,32\n", _encoded(changed))
    for left, right in zip(a.values, b.values, strict=True):
        assert isclose(right.value, 8.0 * left.value, rel_tol=2e-11)
        assert right.unit == "scaled flow"
    for left, right in zip(a.parameters, b.parameters, strict=True):
        factor = 8.0 if str(left.coordinate).endswith((".location", ".scale")) else 1.0
        assert isclose(right.mean, factor * left.mean, rel_tol=2e-11, abs_tol=2e-12)
        assert right.coordinate == left.coordinate


@pytest.mark.parametrize(
    "override",
    [
        {"estimator": "lognormal_lmoments"},
        {"estimator": "lp3_lmoments", "reference_discharge": True},
        {"estimator": "lp3_lmoments", "reference_discharge": 0},
        {"estimator": "gp_lmoments", "gp_location_mode": "fixed"},
        {"estimator": "gp_lmoments", "gp_location_mode": "free", "gp_location": 0.0},
        {"estimator": "gp_lmoments", "gp_location_mode": "unknown"},
        {"estimator": "gp_lmoments", "gp_location_mode": "fixed", "gp_location": 8.0},
        {"estimator": "gumbel_lmoments", "lh_shift": 0},
        {"estimator": "gumbel_lmoments", "reference_discharge": 1},
        {"estimator": "lognormal_lmoments", "reference_discharge": "2"},
        {"estimator": "gev_lh_fixed_h"},
        {"source_classification": "censored"},
    ],
)
def test_ordinary_unsupported_settings_have_no_output(tmp_path: Path, override: dict[str, object]) -> None:
    ams = tmp_path / "ams.csv"
    ams.write_bytes(b"Year,AMS\n2000,1\n2001,2\n2002,4\n")
    config = tmp_path / "settings.json"
    config.write_bytes(_encoded({**_settings(), **override}))
    destination = tmp_path / "report"
    with pytest.raises(SystemExit) as error:
        main(["fit", "--ams", str(ams), "--settings", str(config), "--output", str(destination)])
    assert error.value.code == 2
    assert not destination.exists()


@pytest.mark.parametrize("row", [b"2001,\n", b"2001,0\n", b"2000,2\n"])
def test_ordinary_reuses_exact_record_admission(row: bytes) -> None:
    with pytest.raises(ValueError):
        calculate_from_bytes(b"Year,AMS\n2000,1\n" + row + b"2002,4\n", _encoded(_settings()))


def test_signed_gumbel_quantile_and_exact_source_bundle(tmp_path: Path) -> None:
    ams = tmp_path / "ams.csv"
    # BOM, unsorted source years, original decimal tokens and CRLF survive untouched.
    raw = b"\xef\xbb\xbfYear,AMS\r\n2002,4.000\r\n2000,1.0\r\n2001,2.00\r\n"
    ams.write_bytes(raw)
    config = tmp_path / "settings.json"
    settings_bytes = _encoded({**_settings(), "non_exceedance_probabilities": [0.0001, 0.9]})
    config.write_bytes(settings_bytes)
    destination = tmp_path / "report"
    assert main(["fit", "--ams", str(ams), "--settings", str(config), "--output", str(destination)]) == 0
    doc = json.loads((destination / "results.json").read_text(encoding="utf-8"))
    assert doc["results"][0]["value"] < 0
    assert any("not a physically possible discharge" in warning for warning in cast(list[str], doc["warnings"]))
    assert doc["results"][0]["meaning"] == "deterministic_fitted_quantile"
    assert doc["provenance"]["input_sha256"] == sha256(raw).hexdigest()
    assert doc["provenance"]["settings_sha256"] == sha256(settings_bytes).hexdigest()
    assert (destination / "source-ams.csv").read_bytes() == raw
    assert (destination / "source-settings.json").read_bytes() == settings_bytes
    assert ams.read_bytes() == raw
    assert config.read_bytes() == settings_bytes
