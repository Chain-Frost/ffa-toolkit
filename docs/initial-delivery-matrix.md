# Supported calculations and initial delivery acceptance

Integration profile: 26 November 2026. Planning: [#14]; delivery and exact accepted-head/CI record: [#1].
This crosswalk describes the integrated source tree and remaining requirements; it is not a new
method registry, result schema, automated gate or implementation prerequisite. Use the live linked
issues to determine activation and ownership. Independent mathematical verification and native
qualification are different: **no calculation below is qualified as Flike 5.0.306.0 / file3.10
equivalent**. See the [compatibility contract](compatibility-contract.md) and
[optional external evidence](external-evidence.md).

## Currently installed deterministic calculations

The CLI supports fixed-H GEV (H=0–4) and ordinary L-moment Gumbel, LN2, LP3 and GP
(explicit fixed/free location) fits. Inputs are complete, exact, strictly positive annual maxima
with explicit family-specific settings. Run the generated installed examples from a repository copy:

```console
ffa fit --ams docs/examples/fixed-gev-ams.csv --settings docs/examples/fixed-gev-settings.json --output example-gev-result
ffa explain lh-moments-gev --format markdown
ffa fit --ams docs/examples/ordinary-ams.csv --settings docs/examples/ordinary-settings.json --output example-ordinary-result
ffa explain l-moments --format markdown
```

The GEV settings schema is `ffa-gev-fixed-h/1`; units, year definition, caller source classification,
H and requested non-exceedance probabilities are explicit. The three reports are JSON, CSV and text.
They preserve source/settings byte hashes, parameter/quantile definitions and F/AEP. The fourth-moment
ratio residual is not a standardized goodness-of-fit statistic. No bootstrap or Bayesian inference
is invoked by this command. The output directory must not exist. See
[reporting and settings](wp08-cli-reporting.md) and
[fixed-H evidence](wp04-fixed-h-gev-evidence.md). Ordinary settings use `ffa-ordinary-lmoments/1`,
with explicit positive references for LN/LP3 and explicit GP location choice. Ordinary reports
include original source/settings byte snapshots checked against their SHA-256 provenance. GEV reports
preserve exact endpoint/outside diagnostics and warnings. Signed mathematical quantiles remain signed.
No bootstrap, Bayesian fit, POT exposure or native default is selected by these commands.

## Workflow matrix

Exact AMS below means caller-classified complete positive systematic annual maxima, not censored or
historical observations. Deterministic rows use no prior or rating correction. Bayesian rows require
explicit source classification, a named same-coordinate prior/density and numerical settings;
unknown controls never become defaults. Each installed calculation must preserve original source
labels, units, observation/exposure decisions, actual file/settings identity, method version,
outputs and limitations.

The components retain their [baseline validation](validation.md) and scoped independent numerical
references, with refreshed integrated acceptance recorded in [#1]. “Component” denotes bounded
mathematical evidence, not a complete installed workflow or native qualification. “Pending” denotes
required work rather than a selectable command. Use the public method pages and
[reference register](references.md) for the mathematical basis.

| Family / estimator | Record / exposure | Prior / rating | Output / uncertainty meaning | Internal status and independent evidence | Installed fit | Remaining owner / work |
| --- | --- | --- | --- | --- | --- | --- |
| GEV ordinary L and fixed LH, H=0–4 | Exact AMS | None | Fitted point parameters, F/AEP quantiles, raw fourth-ratio residual and exact fitted-support diagnostics; no interval | Implemented; [fixed-H tests](../tests/test_gev_lh.py), [support tests](../tests/test_gev_support.py), CLI/report tests and installed smoke | Yes | [#12] bounded support reporting complete; [#7] broader reports; [#2] standardized diagnostics/uncertainty |
| Gumbel ordinary L, raw Q | Exact AMS | None | Fitted location/scale and signed quantiles | Implemented; [ordinary tests](../tests/test_ordinary_lmoments.py), [CLI tests](../tests/test_wp08_ordinary_cli.py), independent installed reference | Yes | [#11] bounded point-fit CLI complete; [#7] broader scientific reports |
| LN2 ordinary L, raw Q | Exact positive AMS | None; positive physical reference discharge | Fitted natural-log parameters and quantiles | Implemented; independent moment/unit and installed CLI checks | Yes | [#11] bounded point-fit CLI complete; [#7] broader reports |
| LP3 ordinary L, ln(Q/reference) | Exact positive AMS | None; positive reference discharge | Fitted log-coordinate parameters, skew and quantiles; support admission | Implemented; independent inversion, limit/support/unit and installed CLI checks | Yes | [#11] bounded point-fit CLI complete; [#7] broader reports |
| GP ordinary L, raw Q; explicit fixed/free location | Exact AMS (not POT exposure) | None | Point parameters/quantiles with explicit location and shape convention | Both variants implemented; independent inversion/support and installed CLI references | Yes | [#11] bounded point-fit CLI complete; [#7] broader reports; no POT claim |
| GEV standardized diagnostics and optimized H | Exact AMS; eligible candidates H=0–4 | None | Conditional variance, standardized z, explicit selection/failure/tie policy; not raw residual | Fixed-H core only; [optimized-H specification](wp04-optimized-h-specification.md), conditional variance/selection verification pending | Pending | [#2], coordinate [#12] |
| Fixed/selected-H GEV and ordinary L families, declared uncertainty | Exact AMS and admitted model | None | Sampling-based bootstrap intervals, declared refit/reselection and attempted/failed replicates; not Bayesian credible limits | Fixed-H research primitive only (explicit seed/counts/interval/type-7 interpolation) | Pending | [#2], [#7], [#1]; specify required family/selection uncertainty and installed integration |
| LN Bayesian, fixed or estimated log-SD | Classified disjoint systematic exact positive and positive left/right/interval censor records; missing/excluded retained | Supplied named Gaussian prior; positive physical reference; no rating | Distinct posterior estimands; divergent estimated-scale upper means omitted; output-specific precision separate | Explicit binding, actual inference, gamma scaling, bounded fixed-budget precision and [CLI workflow](wp08-bayesian-cli.md); [analytic/2D-quadrature tests](../tests/test_wp06_lognormal_posterior.py), replay, unit/censor/prior checks | Implemented, acceptance pending | [#3]/[#4] bounded components complete; [#6] installed acceptance; [#5] broader precision; [#8]/[#9]/[#10] broader bindings |
| GEV and LP3 Bayesian, fixed scale and fixed shape | Classified positive systematic exact/censored records; missing/excluded retained | Supplied one-coordinate Gaussian prior, explicit LP3 reference; no rating | Location/log-mean and weighted probability; quantity-specific precision pending | Internal binding with independent regression cases; [scope and validation status](wp06-fixed-family-posterior.md) | Pending | [#8], [#5]/[#6] |
| Other LP3, Gumbel, GEV and GP Bayesian combinations | Classified exact/censored systematic exposure where the law admits it | Named supplied prior; density-preserving transformations, no rating | Same distinct posterior and precision quantities | [Kernel](../tests/test_distributions.py), [likelihood](../tests/test_likelihood.py), prior/sampler components only; additional family binding and posterior reference checks pending | Pending | [#8], [#4]/[#5]; clarify GP annual-max interpretation and admitted pairings |
| Admitted Bayesian families with supplied regional prior | Explicit classified records per family | Joint supplied prior, named coordinates/covariance, regional training-gauge provenance and Jacobian | Posterior conditional on supplied regional information, not a guessed default | [Prior component tests](../tests/test_priors.py); actual regional-prior workflow pending | Pending | [#3]/[#6]/[#8]; preserve regional provenance |
| Admitted Bayesian families with historical/grouped exposure | Complete disjoint counted groups, perception periods/thresholds, year-labelled exact exceedances and unmeasured counts | Supplied prior; rating separate | Posterior and precision with exposure and counts conserved | [Likelihood](wp05-likelihood-priors-rating.md) and [paired censor integration](../tests/test_wp07_paired_censor_integration.py) components; posterior/CLI missing | Pending | [#9] |
| Admitted Bayesian families with shared rating | Classified observations/exposure with explicit rating epochs and consistent units | Joint weighted scenario vectors; declared measured-flow law and whole-series marginal likelihood | Posterior conditional on rating model, with separate numerical error | [Shared-rating component tests](../tests/test_rating_errors.py); actual posterior/CLI missing | Pending | [#10], coordinate [#9] |
| Low-flow screening and adopted exclusion/censor treatment | Systematic AMS, source years, thresholds, tie/zero/count distinctions; paired source when established | No prior for screening; explicit prior in downstream posterior | Screen decision and audit, before/after outputs where supported; not automatic removal | Independent USGS MGBT 1.0.7 [statistics](../tests/test_wp07_mgbt_statistic.py), [conditional probabilities](../tests/test_wp07_mgbt_pvalues.py), sweeps and paired-censor integration | Pending screening-to-fit path | [#6]/[#8]/[#9]/[#7]; reuse existing screen, retain analyst decision and exposure |
| LN/LP3 zero-threshold inference | Zero/threshold observations need an identified observation law | Explicit law/prior; no improvised atom, offset or prior | Threshold-logit inference and uncertainty when specified | Threshold coordinates only; actual fitted law unresolved | Unsupported; reject | [#1]/[#6]/[#8], explicit scope/evidence disposition; native law separate |
| Legacy no-prior/default mode | Classified exposure as admitted by specified law | Unidentified native density/bounds/defaults | Posterior only with identified law | Unsupported, no inference of prior from absent settings | Unsupported; reject | [#1]/[#6]/[#8], scope/evidence disposition; native law separate |
| Unknown-year historical magnitudes and unmapped controls | Retain undated magnitudes and unresolved flags, no invented chronology | Explicit model and settings required | No inference until admitted | Reader/record representation only; defined historical likelihood does not cover all unknown-year processes | Unsupported; reject | [#9]/[#1], identify model or explicit unresolved limitation |
| POT, new production MCMC, regional model fitting, nonstationarity, WMAwater extensions | Separate future scope | Not specified | No initial delivery claim | Outside current workflow | No | See [compatibility contract](compatibility-contract.md); GP distribution fit is not POT |

See [ordinary L-moment evidence](wp04-ordinary-lmoments-evidence.md) for raw-vs-log inputs and GP
shape/location conventions; [likelihood integration](wp05-integration.md) for source admission;
[weighted outputs](wp06-weighted-output-evidence.md) and [reporting](wp08-cli-reporting.md) for estimands.

## Shared components, evidence and ownership

| Requirement | Accepted foundation | Remaining requirement |
| --- | --- | --- |
| Python, quality and installation | Distribution `ffa-toolkit`, import `ffa_methods`, CLI `ffa`; Python 3.11–3.14 supported; public baseline and main qualification passed | Use [current development commands](development.md); publication remains disabled |
| Source readers and chronology | Legacy/Year,AMS parsing, observation adapters and admitted neutral case-a/case-b fixture/manifest; synthetic custody and tamper coverage | Workflow-specific classification, unit/exposure conservation and settings validation [#9]/[#6]/[#11]; do not alter fixture identity or infer a matching run |
| Five distribution kernels | Probability/quantile/support/log-tail and coordinate tests; [kernel documentation](distribution-kernels.md); admitted report-internal replay | Family binding [#8]; native matching separately |
| Likelihood, prior and shared rating | Exact/left/right/interval, defined historical/counted terms, named densities/joint rating composition; explicit LN binding and executed references | Remaining family/exposure/rating posterior bindings [#8]/[#9]/[#10], regional-prior admission and installed paths [#6]; no assumed native model |
| Real importance sampling | Full-covariance draws/self-normalized weights, local inverse-curvature builder and explicit gamma-squared covariance scaling; bounded mean MCSE/ESS and actual LN binding | [#4] scaling complete; [#5] output-specific runtime precision, tail/moment admission, replicates and controls; [#6] installed orchestration. Global ESS is not a tail certificate |
| Scientific basis | ARR/primary references and independently checked equations/tests for admitted components; proposal scaling and precision specification reviewed | [#2]/[#5]/[#8]/[#9]/[#10] sufficient published mathematics and runtime verification for remaining scope; no restricted material redistribution |
| Independent external validation | Read-only comparators and [archived result identity audit](pyflood2022-archive-identities.md), aggregate-only receipts, synthetic tests and temporary external inputs | [#13] matched posterior comparator, unresolved publication/export precision/provenance and statistical suitability; archived consistency is not a refit or native comparison |
| Reports, plots and guidance | Typed reports, Bayesian-output bridge, seven offline method pages, generated GEV/ordinary examples, exact support diagnostics and source snapshots | [#7] broader scientific reports, per-workflow examples and static plots/exposure conventions; [#5] precision distinctions and [#6] Bayesian installed path |
| Native qualification | Gate machinery and synthetic rejection/admission tests, pending public ledger | Native evidence externally managed; no compatibility or release claim |

For Bayesian output, distinguish quantile at expected source-coordinate parameters, posterior mean
and median of quantiles, inverse expected CDF (predictive quantile), expected F/direct AEP at a
flow, reciprocal expected AEP (not posterior mean return period), pointwise posterior quantile and
return-period limits, and log10-discharge moments when defined. These are not interchangeable.
Statistical flood uncertainty, bootstrap sampling uncertainty and numerical Monte Carlo error must
remain separate. Preserve signed mathematical outputs with appropriate warnings and report undefined
or infinite quantities explicitly; no fabricated finite sentinels.

## Owner decisions

Do not silently reduce delivery to the easiest installed combinations. Continue sufficiently
specified work independently; record the decision and evidence in [#1] before excluding a required
mathematical workflow:

- [#2]/[#1]: which ordinary-family bootstrap and selected-H uncertainty constructions are required,
  explicit controls, failure/reselection semantics and independent tests.
- [#3]/[#6]/[#8]/[#9]/[#10]: precise family × exposure × supplied/regional prior × shared-rating
  combinations, particularly GP annual maxima versus POT and rating/exposure overlap.
- [#1]/[#6]/[#8]/[#9]: whether independent, fully specified no-prior, zero-threshold or unknown-year
  historical models are required at initial acceptance. Unknown native laws and controls stay
  unsupported until established; no guessed law or fabricated dates.
- [#7]/[#9]: source-defined historical/censored plotting positions with full exposure preserved;
  any unresolved mathematical plot needs explicit disposition rather than deleting data.

## Final installed acceptance for the delivery tracker

Use the existing [method register](method-registry.json), [shared contracts](shared-contracts.md),
[development quality contract](development.md), distribution verifier, installed smoke and
qualification machinery. This is the checklist for [#1]; [#14] completes the planning crosswalk,
not the calculation work.

- [ ] Every required row has an approved explicit configuration, linked owner and accepted
  implementation/verification, or an owner-approved mathematical scope/evidence disposition with
  a linked outstanding requirement. Keep unsupported settings fail-closed.
- [ ] Independently verify family/estimator coordinates, support and endpoints, exposure/count
  conservation, units, prior density/Jacobians, rating law, source decisions, explicit bootstrap
  construction and actual posterior inference. Use analytic, synthetic, high-precision or
  appropriately admitted comparisons; quantity-specific tolerances, no universal MC error bound.
- [ ] Build wheel/sdist on a coherent head and verify resource identity, notices and dataset
  exclusions. Install the wheel in a clean environment outside checkout and execute each admitted
  workflow from files/settings to correctly labelled JSON/CSV/text and supported static plots.
  Verify source/settings hashes, version and method identity; no supplied-particle mock fit.
- [ ] Exercise unsupported/invalid data and settings (missing/excluded/zero, censor, overlaps, units,
  priors, fitted support, numerical failure and existing output directories). Preserve exposure,
  numerical and statistical uncertainty distinctions, F/AEP and each actual estimand; do not
  hide failure or publish misleading partial results.
- [ ] Supply installed help, explanations/references and original rerunnable offline examples for
  every selectable path, with declared assumptions, sources, verification and limitations.
- [ ] Record actual Ruff lint/format, strict Pyright, pytest, build, distribution, clean installed
  CLI smoke and patch whitespace under the current CI policy. Confirm applicable Python 3.11
  and 3.14 qualification and obtain independent final-head review with substantive findings
  resolved. Distinguish local, hosted, skipped and unexecuted checks.
- [ ] Report native qualification separately; externally admitted evidence and the existing native
  gate are required for any equivalence claim, not for independent mathematical delivery.
  Neither planning nor initial delivery authorizes a package release.

[#1]: https://github.com/Chain-Frost/ffa-toolkit/issues/1
[#2]: https://github.com/Chain-Frost/ffa-toolkit/issues/2
[#3]: https://github.com/Chain-Frost/ffa-toolkit/issues/3
[#4]: https://github.com/Chain-Frost/ffa-toolkit/issues/4
[#5]: https://github.com/Chain-Frost/ffa-toolkit/issues/5
[#6]: https://github.com/Chain-Frost/ffa-toolkit/issues/6
[#7]: https://github.com/Chain-Frost/ffa-toolkit/issues/7
[#8]: https://github.com/Chain-Frost/ffa-toolkit/issues/8
[#9]: https://github.com/Chain-Frost/ffa-toolkit/issues/9
[#10]: https://github.com/Chain-Frost/ffa-toolkit/issues/10
[#11]: https://github.com/Chain-Frost/ffa-toolkit/issues/11
[#12]: https://github.com/Chain-Frost/ffa-toolkit/issues/12
[#13]: https://github.com/Chain-Frost/ffa-toolkit/issues/13
[#14]: https://github.com/Chain-Frost/ffa-toolkit/issues/14
