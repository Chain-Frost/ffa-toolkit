# WP03 LP3 report-internal replay — source-custodied evidence (not distributed)

**Evidence status:** Report-internal mathematical consistency only;
`candidate_settings_unverified`; **not** a native Flike conformance case,
a verified `.fld` project/output match, or a numerical inference acceptance case.

This document deliberately resides under `reference/` and is excluded from
both wheel and sdist. The site-specific fitted parameter values below must **not**
be duplicated in packaged guidance, CLI resources or public example payloads.
Both source reports are already staged and hash-pinned in
[manifest.json](manifest.json); no source file is modified by these tests.

The [development requirements](https://github.com/Chain-Frost/ffa-toolkit/blob/main/docs/development.md)
uses report-only expected-parameter values and never launches a vendor
executable or infers missing project controls.

## Pinned report-internal numerical evidence

Two preserved **FLIKE 5.0.306.0 / file 3.10** reports contain both LP3 posterior
expected-parameter coordinates and an "Exp parameter quantile" table. The
read-only regression in
[development requirements](https://github.com/Chain-Frost/ffa-toolkit/blob/main/docs/development.md)
checks **all 40 rows** by replaying the scalar LP3 kernel directly from the
**reported posterior parameter means**, without reproducing inference or
pretending that a similarly named .fld file produced the report.

| Candidate report | Coordinates printed in the report | Upper bound | Expected-parameter rows |
| --- | --- | ---: | ---: |
| [development requirements](https://github.com/Chain-Frost/ffa-toolkit/blob/main/docs/development.md) | mean ln(Q) = 5.45975, ln[SD(ln Q)] = 0.51810, skew = -0.75235 | 20386.9 | 21 |
| [development requirements](https://github.com/Chain-Frost/ffa-toolkit/blob/main/docs/development.md) | mean ln(Q) = 4.51784, ln[SD(ln Q)] = 0.73566, skew = -0.71925; mean threshold logit = -2.02714 | 30354.4 | 19 |

The staged report SHA-256 values are
`7372edd66bf5d9b919c344841a091c15d6ac9cacb0d637760c9214018b5c796c`
(Case A) and
`7f5acd4cf52ad08de5e9f3fa724be1f25a5996758d43923e00515065b64e3dfe`
(Case B). The tests require these exact unmodified bytes. The
[development requirements](https://github.com/Chain-Frost/ffa-toolkit/blob/main/docs/development.md)
records the original hashes, permitted path redaction and
`candidate_settings_unverified` status.

For positive discharge, the reports are consistent with the **expected-parameter**
LP3 quantile `q_T = LP3.isf(1/T)` for Case A. Case B explicitly reports
"Parameter 4 = loge(P/(1-P)) where P is probability of flow <= flow threshold".
For its **positive** expected-parameter quantiles, the matching numerical
relationship is `q_T = LP3.isf((1/T)/(1-p0))` where
`p0 = logistic(-2.02714) ≈ 0.116382715583`. Substituting the distinct
reported **most probable** `P=0.1250` gives incorrect expected-parameter
quantiles. This is a report-output identity above the declared threshold,
**not** a new zero-flow likelihood, prior or unconditional atom model (WP07
owns those decisions). The LP3 upper bounds equal the negative-skew finite
support endpoints to the precision implied by the printed parameters.

The source parameters are printed to five decimal places, quantiles to two
and support upper bounds to one. Regression checks propagate each printed
parameter's **±0.000005 rounding interval** through all corner combinations
and compare the resulting interval with the output's ±0.005 (quantile) or
±0.05 (upper bound) printing interval. This is a
**report-internal, parameter-to-quantile consistency test**, not an arbitrarily
selected percentage tolerance or a matching native-run test. The posterior
sampler, original .fld control settings, report pairing and zero-threshold
likelihood have **not** been qualified. It does not establish legacy GEV
`k` sign or LN/Gumbel/GP coordinate mappings.
### Evidence limits and rounding strategy

The test enumerates all endpoint corners of the five-decimal printed
parameter rounding cells to construct an *evaluated* enclosure and intersects
it with the two-decimal quantile or one-decimal endpoint output interval.
This is reproducible finite-grid numerical evidence for these narrow cases,
not certified interval arithmetic for arbitrary parameters and not a flood
frequency uncertainty bound. It is deliberately separate from stochastic
posterior repeatability and all native-run admission gates.

No redistribution permission for the original client archive is inferred.
This note references existing source-custodied candidate reports only;
no original report excerpt is copied into distributable package resources.
