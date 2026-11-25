# WP05 #30 — Internal synthetic record-to-likelihood integration

**Status:** independently specified mathematical checks only. This document concerns
the internal `ffa_methods.likelihood_integration` boundary and does **not** qualify
a TUFLOW/BMT Flike 5.0.306.0 fit, prior, censor-group interpretation, rating
error model, zero-threshold law or executable run.

## Purpose and components

The merged, separately reviewed foundations are:

- **WP02:** `legacy.adapter.map_annual_series` audits exposure and produces a
  `MappedAnnualSeries` with its original Decimal `Observation`, raw source line,
  source digest, source period and WP01 `AnnualObservation` together. Its
  `ready_for_likelihood` property remains **False**: a parser is not a solver.
- **WP03:** explicit bound continuous `DistributionKernel` instances, with
  discharge-coordinate `logpdf`, `logcdf`, `logsf` and stable endpoint methods.
  A private, pass-through `_WP03KernelContract` bridges only the parameter
  *names* of the WP03 `ppf/isf(probability)` methods to WP01's declared
  `ppf(non_exceedance_probability)` and `isf(survival_probability)` keyword
  signatures. It delegates all seven methods without changing their mathematics;
  no parallel kernel interface or distribution family is introduced.
  No source-parameter or discharge unit is inferred from a filename.
- **WP05:** `log_likelihood` with audited exact/left/right/interval-censored
  observations and source-labelled disjoint historical periods; normalised
  source-coordinate `JointGaussianPrior` with full covariance and Jacobian;
  joint `marginal_shared_rating_log_likelihood` using supplied whole-epoch
  scenarios. The historical binomial multiplicity is retained.
- **WP07:** original screening/audit evidence remains immutable. A separate
  [development requirements](https://github.com/Chain-Frost/ffa-toolkit/blob/main/docs/development.md) owns
  verified, explicitly approved low-flow-to-censor transformation. The
  independent synthetic adapter in this issue does not duplicate that code,
  apply an MGBT decision, substitute a zero atom or append negative-year
  synthetic censor-group placeholders to positive calendar-year exposure.

## Explicit synthetic-only admission

`SyntheticEvidenceDeclaration` requires the purpose
`independent_synthetic_mathematics`, an exact `synthetic:` source ID,
lowercase SHA-256 source identifier, the original and kernel discharge units,
ordered source period IDs and any independently supplied historical source
record IDs. The digest is *caller-supplied identity*, **not** verification that
raw files with that hash have been admitted.

For this bounded component route, WP02 input must declare
`source_file_version="synthetic-component-v1"` and original observation
sources must match the declared `synthetic:` label. Every period must be
fully observed **systematic** exposure without a perception threshold.
The adapter checks row-to-record/year identity, source lines, recorded flags,
units and periods. It rejects project file3.10 input, unresolved/undocumented
numeric flags (including project flag zero), reported or verified zeros,
ambiguous exposure, grouped WP02 counts, a zero threshold, and tampered annual
classification. It never sets WP02 `ready_for_likelihood=True`.

Historical count/known-magnitude evidence is supplied separately through
`HistoricalPeriod`, with an explicit source record and a fully stated,
**positive-calendar-year, inclusive** time span, threshold, labelled measured
magnitudes and additional unmeasured exceedance count. The declaration must
identify each source record. WP05's exposure ledger rejects overlapping annual
years and history; no unrecorded non-exceedance is fabricated. Historical
WP02 periods with unknown perception/selection rules are not automatically
converted into this form. Independent count-only evidence is not automatically
converted from WP02 `CensorGroup`.

There is deliberately **no general units converter**. Each bound kernel's
declared discharge convention must match the mapped observation convention;
a rating scenario must use the same recorded convention in every epoch.
Explicit shared-rating error scenarios are mathematical examples, not
evidence of an operational Flike rating-error distribution or settings.
The prior helper requires source order, explicit transforms and supplied or
regional source-density provenance; legacy `default` and `no_prior` still
raise `PriorEvidenceError`. It does not infer the coupling between a kernel's
parameters and an unknown vendor prior, nor implement the WP06 sampler.

**Security and interpretation:** `SyntheticLikelihoodEvaluation` and
`SyntheticRatingEvaluation` always retain `legacy_qualified=False`.
These are internal functions for scientific component verification, **not**
a new supported public API or an activated `ffa fit` CLI mode. The source
declaration is a restriction and an audit record, not cryptographic proof
that a caller created a synthetic dataset or verified legacy controls.

## Reproducible mathematical example

Test `test_wp02_wp03_wp05_exact_censors_history_and_skipped_years` in
`tests/test_wp05_integration.py` constructs source-preserved synthetic annual
records for 2000–2005: exact 2, left-censored at 1, right-censored at 3,
interval [1,2], missing, excluded (retaining value and reason). A separate
historical period 1890–1893 has threshold 1.5, one *year-labelled* measured
exceedance 4 and one unmeasured exceedance. The bound unit Gumbel kernel has
`F(q)=exp(-exp(-q))` and `f(q)=exp(-q-exp(-q))`.

Its independent joint log-likelihood target is

```text
log f(2) + log F(1) + log[1-F(3)] + log[F(2)-F(1)]
+ log f(4) + log C(3,1) + log[1-F(1.5)] + 2 log F(1.5)
```

The missing/excluded annual terms stay in the ledger as excluded contributions,
rather than being silently treated as floods or threshold counts. The known
historical year is not counted again inside the combinatorial term.

Further maintained tests exercise all five WP03 kernel classes, microscopic
interval integration and NaN-tail rejection, impossible-support observations,
source/year and flag tampering, period overlaps and identity mismatches,
shared rating errors and mixed-unit rejection, full-covariance Gaussian source
coordinates and scale/logit Jacobian semantics, `no_prior`/`default` refusal,
and an advisory-only WP07 screening audit. Existing WP05 three pinned Normal
interval oracles and historical/rating/prior tests remain unchanged.

## Handoff and remaining qualification

- **WP02 #5:** authentic source controls, project flags, historical membership,
  raw evidence and real grouped counts remain its source/record owners.
- **WP07 #31:** separate source-provenance-preserving PILF/censor integration
  (including pair/no-double-counting), not duplicated in this work package.
- **WP06 #9:** can use the internal likelihood/explicit prior pieces after
  independently binding source-parameter coordinates to a kernel. No solver,
  posterior predictive result, simulation control, proposal or native
  importance sampler is selected here.
- **Deferred native #39 and final conformance #12:** actual legacy default/no-prior densities, rating
  laws, zero-threshold rules, exact project/report matching and settings-pinned
  native comparisons remain blocked pending authoritative Flike information.
  These unresolved items do not invalidate the independent synthetic checks
  or prevent completion of the defined WP05 mathematical/software scope.
  Initial development is tracked in #40 under `docs/development-stages.md`.

**Acceptance record:** source branches and tests must be run on the final PR
head under the repository's Python 3.14 Windows/Ubuntu quality workflow
(Ruff, strict Pyright, pytest, wheel/sdist, distribution content,
installed-wheel smoke, whitespace), then receive a **separate independent
Codex review of that head**. Independent synthetic and mathematical
verification is not a substitute for native compatibility qualification.
