# Historical and censored annual-maximum likelihood

**Method ID:** `censored-historical-likelihood`. **Legacy target:** Flike 5.0.306.0, file 3.10. **Qualification:** no integrated, settings-matched legacy likelihood is yet established.

## Practical purpose and required evidence

An observed year may provide an exact annual maximum, only a lower/upper limit, an interval, no measurement, or an explicit exclusion. Historical threshold information can record years with no floods above an understood perception threshold. It supplies *exposure*, not invented exact annual values. Do not turn missing years into zeros or combine a historical count with annual records describing the same years.

Inputs must retain source year labels, discharge units, source flags, exact/censored/excluded/missing kinds, lower and upper limits, historical time span, threshold/count grouping and any overlap. Supported distribution, zero threshold, prior/no-prior and rating-error settings require separate qualification.

## General mathematical foundation — not an asserted Flike default

For a continuous annual-maximum law with density `f(q|theta)` and CDF `F(q|theta)=P(Q<=q|theta)`, the conventional contributions are:

| Evidence | Single-year likelihood contribution |
| --- | --- |
| Exact q | `f(q|theta)` |
| Left-censored at U | `F(U|theta)` |
| Right-censored at L | `1-F(L|theta)` |
| Interval L to U | `F(U|theta)-F(L|theta)` |

The last expression must be computed stably for narrow or extreme-tail intervals. An independently established zero-threshold atom or discrete bound convention changes these formulas and must not be silently ignored.

For a disjoint exchangeable group with `n_low` at/below threshold T and `n_high` above, the independent product is `F(T|theta)^n_low * [1-F(T|theta)]^n_high`, with a grouping coefficient where the observed sampling definition requires it. Which constants the legacy engine retains is a separate conformance question. Verify independence and nonoverlap before multiplying any group with individual years.

## Plotting, uncertainty and current status

ARR Book 3 §2.6.2 equations 3.2.35–3.2.36 give the plotting starting point, **not permission to re-rank a reduced uncensored sample**. Historical perception duration, excluded years and distinct count groups change the available ranking evidence. A later report must identify input provenance, accepted count structure, nonoverlap check, likelihood model, posterior or bootstrap uncertainty as applicable, plus integration error separately. The current package does not expose this fit.

**Scientific basis:** ARR 2019 Book 3 Chapter 2 §§2.6.2, 2.6.3.1–2.6.3.4 and equations 3.2.35–3.2.36, 3.2.40–3.2.49 (`ARR3`); Kuczera (1999), DOI 10.1029/1999WR900012 (`KUCZERA1999`).

**Observed legacy evidence:** some samples retain excluded individual rows, others grouped counts; source-run pairings and rules are not yet established (`LEGACY_SAMPLES`). **Implementation rule:** preserve original evidence, reject double-counting, do not fabricate undated years. **Unresolved:** matched native likelihood, multiplicity constants, historical exposure settings, measurement/rating effects and numerical diagnostics. Bundled citations are available through `ffa references`.
