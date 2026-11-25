# WP02: file3.10 records, report extraction and reference custody

**Status:** bounded parser and evidence-only annual mapping / **not a verified native-Flike replica**. Owner: [development requirements](https://github.com/Chain-Frost/ffa-toolkit/blob/main/docs/development.md); cross-package tracker [development requirements](https://github.com/Chain-Frost/ffa-toolkit/blob/main/docs/development.md). Project target: TUFLOW/BMT Flike 5.0.306.0, project file3.10, CLI only. The original corpus and admission limits are described in [the sample audit](sample-data-audit.md) and [the compatibility contract](compatibility-contract.md).

## Supported extraction and preserved evidence

`src/ffa_methods/legacy/project.py` accepts the explicitly evidenced **file3.10 three-column project layout**: magic/version, title/subtitle, original seven primary control tokens, variable label and units, an uninterpreted pre-observation integer, observation count, and ordered discharge / numeric flag / year records. The entire trailing control/prior block remains an ordered tuple of source lines. The parser never converts an undocumented control into a fitting choice. Invalid version, column count, finite-number syntax, flow, count or duplicated year fails with an explicit `LegacyInputError`.

`parse_ams_csv` accepts strict `Year,AMS` with exact `Decimal` values. A blank AMS cell is *missing*; a written zero is `reported_zero_unverified`, **not proven physical zero**. The project source flag remains intact. Zero flags on the currently staged examples do not prove the semantics of nonzero flag codes. Nonzero flags produce `flag_unresolved` and must not be silently interpreted as exclusion, censoring or a fitting instruction.

`report.py` extracts the printed gauged observation table, posterior parameter labels/coordinates/correlation triangles, return-period quantiles with printed 90% bounds and log10(q) moments, expected-probability-at-flow table with printed return periods, threshold and threshold counts, and report/version headers. Numeric tokens are parsed as `Decimal` and raw source lines remain available. The original report headings are preserved: a printed `AEP plot position` or `Expected probability` is not by itself proof of an annual *exceedance* probability. The Case B report's corrupted fourth-parameter label (NUL characters and a spurious number) is retained; the separate `Parameter 4 = loge(P/(1-P))` declaration is captured without rewriting the corrupt line.

The known data types preserve exact/missing/excluded/reported-zero/verified-true-zero/left- and right-censor/interval-censor/unresolved flag separately. Exclusions require a stated reason. `ExposurePeriod` distinguishes systematic and historical years, requiring explicit year membership and a perception threshold for historical groups. `audit_exposure` rejects overlapping exposures, duplicate observations, unknown groups, invalid counts and group years already represented as observations. A grouped count without known member years remains auditable but **cannot be qualified as likelihood-ready**; it is not assigned to guessed missing years.

## Custody and pairing

`provenance.py` checks admitted sanitized bundle members against manifest sizes and SHA-256 values. Original pre-redaction custody and exact transformations remain private. Public original/member fields refer to the sanitized admission bundle; no verification of original archives is claimed. Reference data stay excluded from wheels and sdists.

Pair review compares each explicit triple's project/CSV year-flow identity, report years/printed discharge precision, title, file version and observed threshold count. A difference greater than half a printed unit is a **data pairing discrepancy**, not a numerical-estimator acceptance tolerance. The reader does not match by filename stem, change a project switch or auto-promote any candidate. Status vocabulary: `candidate_settings_unverified`, `verified_pair`, `mismatched`, `unavailable`, and `qualified_reference`. Automatic review is deliberately limited to candidate, mismatched and unavailable; verified/qualified require separate authenticated run evidence. The current two triples must remain candidates even after successful parsing.

## WP01 / WP03 / WP05 / WP07 integration contract

- **WP01:** These modules live under `src/ffa_methods/legacy/` without adding a competing root package skeleton, CLI, pyproject, tool config, CI or publication. WP01 owns source distribution exclusion (`reference/**`), Python 3.14 environment, Ruff 0.16.6 / strict Pyright and the common public-facing domain contract. Adapt/import as appropriate when those shared interfaces are frozen.
- **WP03:** The observed LP3 posterior **source order** is index 1 mean of natural-log flow, 2 natural logarithm of standard deviation of natural-log flow, 3 skew of natural-log flow; index 4 when present is the separate `logit(P(flow <= zero threshold))` coordinate, evidenced by explanatory report text but subject to a corrupted summary label. This is not raw log-standard-deviation. Report `Mean(log10(q))` is a *different* output coordinate and must not be used for a parameter prior. For GEV, the audit records `u`, `ln(a)` and `k`; precise sign/scaling mapping still needs independent target evidence.
- **WP05/WP07:** Do not treat the source `flag` numeric as an exclusion or a grouped-threshold declaration without controlled legacy evidence. Excluded rows and group-count censoring are separate evidence types. A threshold of 9.889 appears among the opaque Case B project tail controls while the paired candidate report prints zero threshold 0.000; **do not assume the last project token is the active threshold** or silently rewrite it. The effect of that field is unknown.
- **WP06/WP08:** Preserve observed 19/21-row quantile grids, original labels and raw probability values. The return period column is consistent with `T = 1/(1-F)` for a non-exceedance F; that consistency is not proof of every plotting position or confidence-limit method. No Bayesian numerical acceptance follows from matching report extraction.

## Initial reader/adapter acceptance — issues #5 and #40

The bounded parser/adapters are merged in PR #17. Continue the #29 RMC candidate-reader stack after its #22 evidence dependency, preserving undated magnitudes, source labels and hashes. Verify supported conversions/classifications and exposure with retained-file and independent synthetic tests. Reuse the merged #30/#31 likelihood/censor integration instead of reimplementing those boundaries. Complete agreed reader/adaptation scope, guidance, current quality checks and independent review; new native runs are not required to close the defined implementation issue.

## Deferred native evidence — issue #39

Unknown file3.10 control/flag/prior/rating/output meanings, complete executable/input/settings/output run custody and native examples proving the target's option combinations are transferred to #39. Preserve unknown fields rather than assign meaning to unlabelled numbers. The existing six-file corpus does not establish controlled native inference or every option. These limitations constrain the later compatibility claim, not independent testing of supplied records and explicit supported adapters. See [development stages](development-stages.md).

No GUI, modern WMAwater model or invented inference default is part of WP02.

## Quality repair validation (23 November 2026)

Starting from PR #17 head `33690c98309004c5e0203112e05b22a7d3815b34`, split observation, exposure, project/CSV, report and provenance validation into typed internal helpers to satisfy the existing complexity limit. Correct import ordering, Decimal constructor style and formatting; type JSON containers explicitly and check the finite Decimal exponent before rounding comparison. No lint/type configuration or source fixtures changed.

Local Windows Python 3.14.6 validation: Ruff 0.16.6 lint and format, strict Pyright 1.1.414 (zero errors/warnings), all 43 tests (including all 24 WP02 tests and the real six-file candidate corpus), wheel/sdist build, distribution-content verification, clean installed-wheel/CLI smoke outside checkout and patch whitespace passed. These are engineering checks, not independent statistical review or native Flike qualification. Both existing evidence triples retain candidate status.

## Explicit WP01 observation mapping (review continuation)

The internal adapter `src/ffa_methods/legacy/adapter.py` maps **evidence**, not
active likelihood inputs. It requires a caller-supplied source row, SHA-256
file identifier and explicitly classified systematic or historical exposure.
Its `MappedAnnualObservation` returns both the WP01 float
`contracts.AnnualObservation` and the **unmodified** WP02 Decimal record,
original line and source hash. It does not interpret unknown file3.10 switches
or provide a public API.

| WP02 kind | WP01 kind | Admission |
| --- | --- | --- |
| `exact` | `exact` | Numeric conversion only; no native-run acceptance |
| `verified_true_zero` | `exact` with 0 | Only if independently classified; raw project/CSV parsing never asserts this |
| `missing` | `missing` | No value is fabricated; remains distinct from zero |
| `excluded` | `excluded` | Original value and exclusion reason remain in wrapper |
| `left_censored`, `right_censored`, `interval_censored` | Corresponding WP01 censored kind | Caller must supply explicit bounds; no numeric project flag is interpreted |
| `reported_zero_unverified`, `flag_unresolved` | **No mapping** | Reject, never classify automatically |

A nonzero original numeric flag is rejected even if a caller attempted to
reclassify its kind. A project flag of zero is retained as `source_flag="0"`,
not interpreted as a verified control. Decimal-to-float conversion rejects
non-finite or overflow results **and underflow of a nonzero reading to zero**.
A valid finite conversion can still lose source precision, which is why the
original Decimal and raw line are retained.

`map_annual_series` audits explicit exposure membership before conversion.
Unaccounted years, group/observation overlap, and undated records are not
converted to annual rows. The current common `CensorGroup` contract lacks
explicit member-year custody; grouped evidence is **rejected** by this
annual-only adapter rather than silently flattened or double-counted.
The resulting series carries `zero_threshold=None`; controls/priors/rating
options, reported zero semantics, native-run correspondence and likelihood
rules remain unresolved. `MappedAnnualSeries.ready_for_likelihood` is always
false until a separately reviewed admission mechanism exists.

### Independent review and validation history

Original PR17 head `a00536e` passed hosted Python 3.14 Ubuntu/Windows
quality/build/install checks (run 45170884532) and its six-file custody
tests, but the initial independent Codex review found that a malformed
later numerical table row could terminate a parse silently. The review
continuation now raises on malformed numeric rows and tests truncated
quantile and probability records. The final branch CI and a new Codex
review must be recorded separately: previous-head checks do not prove
this change correct. Both legacy candidate triples remain unqualified.

Independent Codex review of `10ba1557` found two additional fail-closed
boundaries, corrected in the subsequent reviewed candidate: malformed later
gauged-observation rows must raise instead of ending the table, and distinct
Decimal censor interval bounds must remain strictly ordered **after** float
conversion. Regression tests exercise both cases. Exact final-head CI and
independent review are required again; no earlier-head review certifies the
latest functional changes.

The final-head review also identified that the annual adapter initially
discarded historical `ExposurePeriod` identifiers, calendar membership and
perception thresholds. `MappedAnnualSeries.source_periods` now retains the
**original immutable exposure periods** alongside the mapped annual and
Decimal records. Separate synthetic historic periods with thresholds 10 and 20
remain distinct after mapping; neither is admitted to likelihood without
external control qualification.
