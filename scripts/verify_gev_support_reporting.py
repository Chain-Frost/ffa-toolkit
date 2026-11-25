"""Targeted GEV-only support policy audit of the existing pinned external input cohort."""

import json
import sys
from hashlib import sha256
from math import isfinite
from pathlib import Path

from ffa_methods.estimators.gev_lh import fit_gev_lh
from ffa_methods.gev_support import GEVSupportObservation, diagnose_gev_support
from scripts.audit_pyflood2022 import (
    ASSETS,
    REVISION,
    archive_csv,
    flood_table,
    group_series,
    pair_audit,
    parse_table,
    verify_asset,
)
from scripts.validate_pyflood2022_fits import cohort


def run(directory: Path) -> None:
    assets = {name: verify_asset(directory / name, size=size, blob=blob) for name, (size, blob) in ASSETS.items()}
    header, metadata, _ = parse_table(assets["sites_info.csv"])
    index = header.index("STATIONID")
    sites = {row[index] for row in metadata}
    if not sites or "" in sites or len(sites) != len(metadata):
        msg = "invalid metadata station IDs"
        raise ValueError(msg)
    first = flood_table(archive_csv(assets["flood_data_v5.zip"])[1])
    second = flood_table(archive_csv(assets["flood_data_censored_v5.zip"])[1])
    if not pair_audit(first, second)["exact_unique_raw_pair"]:
        msg = "pinned archives do not have identical unique raw observations"
        raise ValueError(msg)
    cases, _ = cohort(first, sites)
    if not cases:
        msg = "no eligible station records"
        raise ValueError(msg)
    rows = group_series(first)
    outside_counts = [0] * 5
    equality_counts = [0] * 5
    rounded_disagreements: list[tuple[str, int]] = []
    for site, values, _ in cases:
        records = tuple(
            GEVSupportObservation(year, float(value)) for year, value in sorted(rows[site]) if value is not None
        )
        if tuple(record.discharge for record in records) != values:
            msg = "source observations do not match the admitted cohort"
            raise ValueError(msg)
        for h in range(5):
            fitted = fit_gev_lh(values, shift=h)
            support = diagnose_gev_support(
                records, location=fitted.location, scale=fitted.scale, shape=fitted.shape, unit="source flow unit"
            )
            outside_counts[h] += bool(support.outside)
            equality_counts[h] += bool(support.at_endpoint)
            # Independent ordinary-float comparison, explicitly not authoritative at rounding boundaries.
            if fitted.shape:
                bound = fitted.location + fitted.scale / fitted.shape
                if isfinite(bound):
                    rounded = tuple(
                        row for row in records if (row.discharge > bound if fitted.shape > 0 else row.discharge < bound)
                    )
                    if rounded != support.outside:
                        rounded_disagreements.append((site, h))
    print(
        json.dumps(
            {
                "scope": "GEV fitted-support policy only; not native or MCMC evidence",
                "revision": REVISION,
                "sha256": {name: sha256(data).hexdigest() for name, data in assets.items()},
                "eligible_sites": len(cases),
                "station_variant_fits": len(cases) * 5,
                "outside_station_variant_counts_by_h": outside_counts,
                "endpoint_equality_station_variant_counts_by_h": equality_counts,
                "rounded_endpoint_disagreements": rounded_disagreements,
            },
            allow_nan=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        message = "usage: python -m scripts.verify_gev_support_reporting PINNED_ASSET_DIRECTORY"
        raise SystemExit(message)
    run(Path(sys.argv[1]))
