"""Exercise actual fixed-H fits and all three report representations."""

import csv
import json
from hashlib import sha256
from pathlib import Path
from typing import cast

import pytest

from ffa_methods.cli import main


@pytest.mark.parametrize(
    ("flows", "shift", "kind", "outside_years"),
    [
        ((1, 100, 101, 102, 103, 104, 105, 106, 107, 108), 0, "finite_upper", (2008, 2009)),
        ((1, 2, 3, 4, 5, 6, 7, 8, 9, 100), 4, "finite_lower", (2000, 2001, 2002, 2003, 2004)),
    ],
)
def test_support_violations_preserve_fit_and_source_identity(
    tmp_path: Path, flows: tuple[int, ...], shift: int, kind: str, outside_years: tuple[int, ...]
) -> None:
    # The expected cases follow independent normalized-PWM GEV moment inversion:
    # upper endpoint about 106.56; lower endpoint about 5.62.
    ams = tmp_path / "ams.csv"
    rows = tuple((2000 + i, q) for i, q in enumerate(flows))
    ams.write_text("Year,AMS\n" + "".join(f"{year},{q}\n" for year, q in rows), encoding="utf-8")
    settings = tmp_path / "settings.json"
    source = Path(__file__).resolve().parents[1] / "docs/examples/fixed-gev-settings.json"
    config = cast(dict[str, object], json.loads(source.read_text(encoding="utf-8")))
    config["lh_shift"] = shift
    settings.write_text(json.dumps(config), encoding="utf-8")
    output = tmp_path / "report"
    before = ams.read_bytes()
    assert main(["fit", "--ams", str(ams), "--settings", str(settings), "--output", str(output)]) == 0
    doc = cast(dict[str, object], json.loads((output / "results.json").read_text(encoding="utf-8")))
    support = cast(dict[str, object], doc["gev_fitted_support"])
    assert support["kind"] == kind
    assert support["observation_count"] == len(rows)
    assert support["outside_count"] == len(outside_years)
    assert support["at_endpoint_count"] == 0
    assert support["outside"] == [{"year": year, "discharge": q} for year, q in rows if year in outside_years]
    assert len(cast(list[object], doc["results"])) == 4
    assert cast(dict[str, object], doc["provenance"])["input_sha256"] == sha256(before).hexdigest()
    assert ams.read_bytes() == before
    assert any("outside fitted GEV support" in warning for warning in cast(list[str], doc["warnings"]))
    with (output / "results.csv").open(encoding="utf-8", newline="") as stream:
        csv_rows = list(csv.DictReader(stream))
    assert len(csv_rows) == 4
    assert all(json.loads(row["gev_fitted_support_json"]) == support for row in csv_rows)
    report = (output / "report.txt").read_text(encoding="utf-8")
    assert "FITTED GEV SUPPORT" in report
    assert all(f"outside: year={year}," in report for year in outside_years)
