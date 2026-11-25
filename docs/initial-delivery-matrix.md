# Supported calculations and initial delivery acceptance

Baseline: FFA Toolkit main `34cc841f45555246bbb3ac98c8166e55548678ad`. Planning: [#14]; delivery: [#1].
This crosswalk describes the accepted software and the remaining requirements; it is not a new
method registry, result schema, automated gate or implementation prerequisite. Use the live linked
issues to determine activation and ownership. Independent mathematical verification and native
qualification are different: **no calculation below is qualified as Flike 5.0.306.0 / file3.10
equivalent**. See the [compatibility contract](compatibility-contract.md) and
[optional external evidence](external-evidence.md).

## Currently installed calculation

The only accepted file-to-fit path at the baseline is fixed-H GEV (H=0–4), with complete, exact,
strictly positive annual maxima and explicit settings. Run the generated example from an installed
environment (paths below are relative to the repository):

```console
ffa fit --ams docs/examples/fixed-gev-ams.csv --settings docs/examples/fixed-gev-settings.json --output example-result
ffa explain lh-moments-gev --format markdown
```

The settings schema is `ffa-gev-fixed-h/1`; units, year definition, caller source classification,
H and requested non-exceedance probabilities are explicit. The three reports are JSON, CSV and text.
They preserve source/settings byte hashes, parameter/quantile definitions and F/AEP. The fourth-moment
ratio residual is not a standardized goodness-of-fit statistic. No bootstrap or Bayesian inference
is invoked by this command. The output directory must not exist. See
[reporting and settings](wp08-cli-reporting.md) and
[fixed-H evidence](wp04-fixed-h-gev-evidence.md).

## Workflow matrix

Exact AMS below means caller-classified complete positive systematic annual maxima, not censored or
historical observations. Deterministic rows use no prior or rating correction. Bayesian rows require
explicit source classification, a named same-coordinate prior/density and numerical settings;
unknown controls never become defaults. Each installed calculation must preserve original source
labels, units, observation/exposure decisions, actual file/settings identity, method version,
outputs and limitations.

The baseline's internal components and tests have independent review evidence in
[baseline validation](validation.md) and passing [main qualification](https://github.com/Chain-Frost/ffa-toolkit/actions/runs/45612275257).
“Component” below refers only to that bounded evidence, not a verified installed workflow.
“Pending” denotes required work, not a selectable command. Follow the cited public method pages and
[reference register](references.md) for the mathematical basis. The linked issues own the remaining
work, including integration across shared boundaries.

| Family / estimator | Record / exposure | Prior / rating | Output / uncertainty meaning | Internal status and independent evidence | Installed fit at baseline | Remaining owner / work |
| --- | --- | --- | --- | --- | --- | --- |
| GEV ordinary L and fixed LH, H=0–4 | Exact AMS | None | Fitted point parameters, F/AEP quantiles, raw fourth-ratio residual; no interval | Implemented; [fixed-H tests](../tests/test_gev_lh.py), [CLI tests](../tests/test_wp08_fixed_gev_cli.py), installed smoke | Yes, command above | [#12] fitted support; [#7] reporting; [#2] uncertainty |
| Gumbel ordinary L, raw Q | Exact AMS | None | Fitted location/scale and quantiles | Implemented, [ordinary L tests](../tests/test_ordinary_lmoments.py) | Pending | [#11] |
| LN2 ordinary L, raw Q | Exact positive AMS | None; positive physical reference discharge | Fitted natural-log parameters and quantiles | Implemented; independent moment/unit checks in ordinary L tests | Pending | [#11] |
| LP3 ordinary L, ln(Q/reference) | Exact positive AMS | None; positive reference discharge | Fitted log-coordinate parameters, skew and quantiles; support admission | Implemented; independent inversion, limit, support and unit tests in ordinary L tests | Pending | [#11] |
| GP ordinary L, raw Q; explicit fixed/free location | Exact AMS (not POT exposure) | None | Point parameters/quantiles with explicit location and shape convention | Both variants implemented; independent inversion/support tests in ordinary L tests | Pending | [#11] |
| GEV standardized diagnostics and optimized H | Exact AMS; eligible candidates H=0–4 | None | Conditional variance, standardized z, explicit selection/failure/tie policy; not raw residual | Fixed-H core only; [optimized-H specification](wp04-optimized-h-specification.md), conditional variance/selection verification pending | Pending | [#2], coordinate [#12] |
| Fixed/selected-H GEV and ordinary L families, declared uncertainty | Exact AMS and admitted model | None | Sampling-based bootstrap intervals, declared refit/reselection and attempted/failed replicates; not Bayesian credible limits | Fixed-H research primitive only (explicit seed, counts, interval level and type-7 interpolation); tests in fixed-H evidence | Pending | [#2], [#11], [#7]; specify broader family/selection uncertainty |
| LN Bayesian, fixed or estimated log-SD | Explicit classified systematic exact positive AMS and positive left/right/interval censor bounds; missing/excluded preserved, years disjoint | Supplied named Gaussian prior in source coordinates; positive physical reference; no rating | Actual posterior quantities and intervals, separately assessed integration precision | Kernels, likelihood/prior and sampler/proposal components; [independent sampler](../tests/test_wp06_importance_sampling.py) and [curvature](../tests/test_wp06_importance_proposal.py) tests; model binding pending | Pending | [#3] binding, [#6] installed; [#4]/[#5] proposal/precision |
| LP3, Gumbel, GEV and GP Bayesian, each with explicit family-specific coordinates/support | Classified exact/censored systematic exposure where the law admits it | Named supplied prior; density-preserving transformations, no rating | Same distinct posterior and precision quantities | [Kernel](../tests/test_distributions.py), [likelihood](../tests/test_likelihood.py), prior/sampler components only; family binding and posterior reference checks pending | Pending | [#8], [#4]/[#5]; clarify GP annual-max interpretation and admitted pairings |
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
| Likelihood, prior and shared rating | Exact/left/right/interval, defined historical/counted terms, named densities and joint rating-scenario composition | Actual posterior bindings [#3]/[#8]/[#9]/[#10] and installed paths [#6]; no assumed native model |
| Real importance sampling | Full-covariance explicit Gaussian draws and self-normalized weights, local mode/full inverse-curvature builder, bounded mean MCSE/ESS; [execution](wp06-importance-execution.md) and [proposal](methods/importance-sampling-reference.md) evidence | [#4] explicit covariance scaling (Kuczera 1999); [#5] output-specific error/precision, independent replicates and explicit controls. No global ESS tail certificate |
| Scientific basis | ARR and primary literature catalogued in [references](references.md); source-specific formulas, assumptions and independent expected values required for each additional method | [#2]/[#4]/[#5]/[#10] verify sufficient published mathematics before implementing; no redistribution of restricted material |
| Independent external validation | Read-only audit/comparator scripts and [external evidence instructions](external-evidence.md); ordinary tests use independent synthetic/analytic checks | [#13] whole-cohort numerical/exposure limitations, archived result identity and settings-matched posterior comparator; external data rights and correspondence remain separate. Support findings: [#12] |
| Reports, plots and guidance | Typed report formatter, weighted-output bridge, six offline method pages and generated fixed-H example | [#7] complete usable explanations, reproducible installed examples and static plots with source-defined positions/exposure; preserve [#12] support diagnostics and [#5] precision distinctions |
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

- [#2]/[#11]: which ordinary-family bootstrap and selected-H uncertainty constructions are required,
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
