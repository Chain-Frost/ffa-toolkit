# Independent Australian peak-flow evidence — pyflood2022 v5

[development requirements](https://github.com/Chain-Frost/ffa-toolkit/blob/main/docs/development.md), under #40/#2.
This read-only study is **not native Flike qualification** (#39/#12).

## Source pin and permissions

- Lerat, J. and Vaze, J. (2025), *How February 2022 redefines extreme floods in
  Australia*, Communications Earth & Environment 6, 378,
  [doi:10.1038/s43247-025-02307-z](https://doi.org/10.1038/s43247-025-02307-z).
- [pyflood2022 pinned commit d603936](https://github.com/jlerat/pyflood2022/tree/d603936ae642feef94228b39c33b7307dc236b8e).
  Version 5 archive names and plotting scripts are inspected, not executed.
- [Zenodo DOI 10.5281/zenodo.14271026](https://doi.org/10.5281/zenodo.14271026)
  is an archived release linked by the README; do not treat it as byte-identical
  to the selected Git commit. Its listed download is 8.8 MB; 105.7 MB was
  the cumulative download volume, not the archive size.

| Asset under source data/ | Bytes | Git blob SHA-1 |
| --- | ---: | --- |
| floods/flood_data_v5.zip | 1388679 | d739e149d92c9569b68250881098b83b226d1b47 |
| floods/flood_data_censored_v5.zip | 1384695 | d7bc53bb04975e2a7524fb588e9b2071cbba52de |
| sites_info.csv | 73895 | 75cdd56a08c6d075ac8974b746b28d478c3a6ec3 |

The [source licence](https://github.com/jlerat/pyflood2022/blob/d603936ae642feef94228b39c33b7307dc236b8e/LICENSE)
is MIT (software and associated documentation). The article identifies multiple
state/territory agencies and BoM as original streamflow providers. The licence
for re-distributing the particular derived annual-flow records must be reviewed
separately. No third-party code or dataset bytes are committed or packaged.
The source CSV includes a research-data accuracy/change disclaimer. Read-only
local evaluation is distinct from redistribution or golden-fixture admission.

## Audit and bounded numerical checks

The separate workflow `pyflood2022-audit.yml` fetches only those three fixed
public source assets into a temporary directory. It validates byte lengths
and Git blob digests, computes SHA-256 for archives and extracted CSV bytes,
checks bounded single-member ZIPs without extraction, records exact headers,
counts raw keys/missing/nonpositive/unrepresentable values, water-year gaps,
station metadata correspondence and raw peak differences across the two
archives. Site IDs remain strings, including leading zeros; the source
WATERYEAR field is not converted to calendar year or used to invent exposure.
No archive scripts run. The audit logs aggregate counts, hashes and only
selected public station identifiers; no raw data rows or persisted artifacts.
The source bytes are discarded at the end. Ordinary Python quality CI uses
synthetic tests and does not retrieve the public archives.

For a local read-only run, place the three exact pinned assets in a directory
outside the checkout and run `python scripts/audit_pyflood2022.py <directory>`
from an environment with the existing package installed. No additional
dependency is required. The pinned source object digest is a byte identity
check, not scientific evidence of validity.

Select three distinct eligible station series deterministically: shortest,
median and longest by record count and site ID. Require at least 30 finite
strictly positive nonconstant peaks, a matching metadata site, and no duplicate
site/year records. Gaps remain gaps; no claim of complete record exposure
or verified stationarity is made. Independently recompute ordinary sample
L-moments from unbiased ascending-rank PWMs; compare to the H=0 fit's
theoretical moments. Independently compare reported quantiles at F=0.5,
0.9 and 0.99 with SciPy GEV at the fitted parameters and same shape convention.
These are numerical consistency tests on real records, **not published
MCMC result comparisons**. A rejected fit is counted, not replaced by a
more favourable station. Tolerances are explicit in the script; no native
or engineering quantile acceptance claim follows.

## Comparator boundary and follow-up

The paper's Methods, equations (1)–(3), use GEV in the ARR shape-sign
convention, flat location and raw scale priors, Normal shape prior
(mean zero, SD 0.2), Stan NUTS and 10,000 posterior parameter sets.
The 1% AEP quantity is the posterior **mean of the individual 0.99 quantiles**,
not the inverse of the averaged CDF and not the quantile at the mean
parameters. Sensitivity analyses use LP3 and a lower-20% left-censored
fit. The public scripts read archived results and produce figures;
they do not themselves establish exact Stan fitting settings, prior
bounds, seed, year completeness rules, censor threshold/tie convention
or fitted-run correspondence.

Do not compare a deterministic fit or differently specified prior to
the paper's posterior mean as a pass/fail oracle. Before independent
MCMC-vs-importance comparisons, establish source provenance, exact
row/variable/subset and published output semantics, equivalent prior
density including coordinate Jacobians, censoring/ties, the LP3
parameterization, and per-estimand numerical precision. Retain results
as external mathematical evidence only. More families, scale and
censor stress cases, and larger regression matrices can follow this
bounded audit under #53 and #40. None closes the native qualification
gate; root Sustainable Use License 1.0 is unchanged.

## First actual source audit receipt

[development requirements](https://github.com/Chain-Frost/ffa-toolkit/blob/main/docs/development.md)
completed on 24 November 2026 against implementation head
`0c85a55a03163cc13179ed3e89d84b84a9f8dd4a`. The byte-size and Git-blob
checks passed for all three assets. Recorded SHA-256 digests:

| Asset | SHA-256 |
| --- | --- |
| flood_data_v5.zip | d40746161efd4c96e4ece741b702ab2f908946e9bde9e73bebcfab28d1d97497 |
| flood_data_censored_v5.zip | a4b0fd54f9a6c783deb27b12042d2b18ea79acaef8bc54af4cd626895e26f9cb |
| sites_info.csv | 3a8484e1efd94d118d8e331747f3f0cc96b587270f7d8a546b5c890a446f9d59 |
| flood_data_v5.csv (member) | 8de976d2795e6f546d638c3a9d3039730a2b166e4a97fbe4bb21f26529afcec2 |
| flood_data_censored_v5.csv (member) | 9edcf364e79180ffae0b2c07d8f7898603eabab45b0f9dd8e436473cdd021833 |

Each archive contains 42,946 unique station/water-year keys and the
same raw peak values, all finite strictly positive and representable as
binary64. There are 1,094 flood-series sites, 1,108 metadata entries and
no unmatched flood sites; 497 sites have water-year gaps. The archive
water-year labels span 1885–2022, with 8/40/130 minimum/median/maximum
records per site. These are counts of the actual pinned assets, not
theoretical complete exposure counts. The censored file is an alternative
fitting result over the same raw annual peaks; it is not a separate
population of censored measurements. Source schema contains distinct
instantaneous peak [m3/sec], specific peak [m3/sec/km2], runoff [mm],
GEV and LOGPEARSON3 ALL/JACKKNIFE/SURPRISE/CI5/CI95 fields. Mapping
the latter to fitted-run metadata remains a separate acceptance task.

| Deterministically selected station | Observations | Gaps | H=0 moment and quantile consistency |
| --- | ---: | ---: | --- |
| 19202-1 | 30 | 21 | PASS |
| 225223 | 51 | 0 | PASS |
| 403200 | 130 | 8 | PASS |

This receipt is for the actual source audit and three real-record
numerical consistency checks. It does not establish either a matched
published MCMC comparison or native equivalence. The original run's
quality matrix found a formatter issue; refreshed exact-head quality
and independent review are required before merge.
