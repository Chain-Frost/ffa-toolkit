# WP05 — Likelihoods, priors, historical evidence and shared rating errors

**Status:** mathematical foundation implemented and tested, but **NOT qualified as legacy Flike behaviour**. This work targets TUFLOW/BMT Flike program 5.0.306.0 / project file 3.10. It builds on WP01 merged through PR #14, and depends on WP02 #5 and WP03 #6, with zero/censor integration owned by WP07 #10. No analytical CLI mode is enabled.

## References and target evidence

- [ARR Book 3 version 4.2](https://www.arr-software.org/pdfs/ARR_190514_Book3_V4.2.pdf), Chapter 2 sections 2.6.3.1–2.6.3.5 (equations 3.2.40–3.2.51), provides the mathematical foundation for exact/censored likelihood, historical evidence, rating errors, and priors.
- Kuczera (1999), DOI 10.1029/1999WR900012, develops at-site Bayesian estimation with gauged and censored/historical information.
- Kuczera (1996), DOI 10.1029/96WR00804, investigates correlated rating-curve error. Its complete treatment and the exact target-build controls require further source investigation.

## Exact, censored and historical likelihood

Internal module: `ffa_methods.likelihood`, consuming the WP01 `ObservationSeries` and `DistributionKernel` interfaces. An exact measured annual peak contributes log f(q); a left-censored observation contributes log F(u); a right-censored observation contributes log S(l); and an interval-censored observation contributes log[F(u)−F(l)]. The implementation uses survival-function subtraction in the upper tail. When both log-CDF and log-survival endpoint values coalesce in binary64 despite a positive-width interval, it uses adaptive composite-midpoint **density integration in the original discharge coordinate**, with rescaled log-density summation to avoid numerical underflow. It accepts the integral only after two consecutive mesh refinements agree within `2e-14` in log probability; if the kernel's density cannot be resolved or convergence is not demonstrated by 1,024 subintervals, it raises `LikelihoodEvidenceError` rather than silently returning negative infinity. This is a numerical consistency test for smooth continuous kernels, not a rigorous bound for arbitrarily spiky densities. Equal endpoints remain zero probability; intervals outside support remain impossible where the kernel's tail probabilities establish this. NaN and positive log-tail probabilities are rejected **before** attempting density integration; kernel CDF/SF monotonicity violations are also rejected. No arbitrary probability floor or interval widening is used.

A count-only group with n unlabelled years, k above threshold h and n−k at/below h contributes:

`log C(n,k) + k log S(h) + (n-k) log F(h)`

The binomial multiplicity constant is retained. The group must have a complete period and both counts must sum to the period length.

The separate `HistoricalPeriod` interface is for an explicitly observed period with r **year-labelled** exact exceedances above h and k additional **unmeasured** exceedances. Its likelihood is:

`sum_known log f(q_i) + log C(n-r,k) + k log S(h) + (n-r-k) log F(h)`

The known years are not also counted as unlabelled binomial trials. Unknown-year historical magnitudes are a different observation process and are not inferred from this model. The threshold must have been perceptible throughout the exposure.

The exposure ledger rejects overlapping annual, grouped and historical periods, repeated years, incompatible counts, unresolved informative exposure and undefined historical bounds. Missing and excluded records remain visible as omitted terms; they are never silently recoded as censored counts. Zero-threshold observations raise an explicit error pending the verified WP07 model. Mathematically impossible exact observations yield negative infinite log likelihood rather than an arbitrary positive flow offset.

## Source-coordinate supplied and regional priors

Internal module: `ffa_methods.priors`. `JointGaussianPrior` accepts a named parameter vector, mean, **full** covariance matrix, source metadata and supplied/regional mode. Cholesky factorization checks positive definiteness and evaluates the normalized multivariate Gaussian density including off-diagonal terms.

The observed LP3 coordinates are `(mean ln Q, ln SD[ln Q], skew[ln Q])`, with an optional logit probability at/below the zero threshold. The prior stays in these *original* coordinates. If an evaluator uses raw positive scale s instead of eta=ln(s), it applies `log p(ln s)−ln s`. If it uses probability p instead of z=logit(p), it applies `log p_source(logit p)−ln(p)−ln(1−p)`. These Jacobians modify the joint density without discarding covariance.

The supplied/regional selection demands explicit source evidence, matching coordinate order and covariance. Regional metadata records whether the analysed gauge contributed to construction of the prior. `no_prior` and `default` selections are **intentionally blocked** until their legacy densities, bounds and numerical treatment are verified. No improvised flat, broad Gaussian or Beta prior is introduced.

## Correlated rating errors — explicitly unqualified composition

Internal module: `ffa_methods.rating_errors`. `RatingEpoch` groups annual observations sharing one rating-error variable. A `JointRatingScenario` supplies a **whole vector** of epoch offsets and a normalized integration weight, permitting correlations across epochs.

For the **declared test measurement model** `measured_y = true_q × exp(b_epoch)`, a fixed error b gives `f_Y(y | b) = f_Q(y exp(−b)) exp(−b)`; censor thresholds transform consistently. Each rating epoch must use the **same recorded discharge unit** (after removing surrounding whitespace); mixing `m3/s` and `ft3/s` is rejected before any scenario is evaluated. There is deliberately no implicit conversion of annual flows, censor thresholds, historical thresholds or kernel parameters. The marginal joint likelihood is the log-sum-exp of conditional whole-series likelihoods weighted by **joint** scenario probabilities. Integrating each observation independently would change the model and is not done.

This composition is independently testable, but **must not** be represented as the exact Flike rating-error facility. The legacy error law, grouping controls, systematic/residual components, covariance, thresholds, defaults and numerical integration are currently unknown. There are no matched enabled/disabled rating-error reference cases.

## Current verification

Focused, independent synthetic checks cover: exponential exact/censor likelihoods and upper-tail intervals; historical count constants and labelled magnitudes; exposure overlap and missing/excluded records; off-diagonal Gaussian normalization; log-scale and logit coordinate Jacobians; and the difference between shared and independent rating errors.

Initial local execution: **35 focused tests passed on Python 3.13.5**. Cross-platform candidate verification on **Python 3.14** passed on both Windows and Linux in [development requirements](https://github.com/Chain-Frost/ffa-toolkit/blob/main/docs/development.md): **47 repository tests passed per OS**, strict Pyright returned zero errors and warnings, Ruff lint/format passed for all eight WP05-owned Python files, and wheel/sdist build, distribution verification and clean installed-wheel smoke tests all passed. The temporary diagnostic workflow was removed after qualification. Standard PR CI must pass at the reconciled WP05 head, and independent numerical comparisons against exact Flike settings remain outstanding. The merged WP01 foundation has a successful main-branch quality run.

## Initial mathematical/software acceptance — issues #8 and #40

The likelihood/prior/rating foundation was merged in PR #18. The source-preserving WP02/WP03/WP05 integration was merged in PR #37, completing #30; see [WP05 integration](wp05-integration.md). The 24 November closeout audit accepted this bounded mathematical/integration scope using the merged final-head CI and independent review receipts recorded in [development requirements](https://github.com/Chain-Frost/ffa-toolkit/blob/main/docs/development.md). Retain independent tests of exact/censored/historical terms, disjoint exposure, covariance/Jacobians, interval tails and declared shared rating models. Do not reopen the completed integration issue merely because native defaults remain unavailable.

Issue #8's initial mathematical/software criteria are accepted, with the following native-only requirements explicitly transferred to #39. Downstream sampler and CLI implementation remain active under #9/#11. This does not assert native compatibility or complete the overall programme.

## Deferred native evidence — issue #39

Decode still-unknown file3.10 prior/rating/observation controls, native regional/supplied formats, effective `no_prior`/`default` densities and bounds, native threshold/zero/count semantics, and the target's rating laws/grouping/defaults. Then admit controlled settings-matched disabled/enabled and mixed-observation comparisons under WP09 #12. Until evidenced, these native modes remain unknown or unsupported. Their absence does not block a fully specified independently tested mathematical model. See [development stages](development-stages.md).

### 23 November 2026 — WP05 correction evidence (PR #18 continuation)

- Addressed two independent Codex findings from review of `6e28ea9caa7e601548c47d5ea189df5d5fcb23b8`: make the three WP05 register entries accurately **partial, not native-qualified**, and reject mixed rating-epoch discharge units before evaluating joint scenarios. The same-unit identity case remains exactly `-2` for two unit exponential exact observations; the mixed-`ft3/s` case raises an error.
- Added the three mathematical Normal interval vectors from [development requirements](https://github.com/Chain-Frost/ffa-toolkit/blob/main/docs/development.md), sourced from [USACE-RMC/Numerics at `7e8e8d1`](https://github.com/USACE-RMC/Numerics/blob/7e8e8d1c5f26e045a35ec9fc09367de95ed05b02/Test_Numerics/Distributions/Univariate/Test_DistributionLikelihoodRegression.cs) under its 0BSD licence. The tests retain the source's expected log probabilities `-43.628216632280818` for [9,10] and [-10,-9], `-46.970640393085588` for [0,1e-20], and `2e-13` absolute tolerance. Values are pinned directly in test parameters while the authoritative public-candidate JSON remains owned by PR22; no duplicate distributable corpus was added.
- A local Python 3.13.5 analytic numerical probe using `erfc` and a binary64 unit-Normal density returned `-43.62821663228082` (both tails) and `-46.97064039308559` (central tiny interval). This probe tests the fallback mathematics only, **not** the full repository or Python 3.14 compatibility. Full final-head GitHub CI, independent re-review, native parameter/control mappings and qualified target-build comparisons must be recorded separately.

- The independent Codex review of `4b9b546` identified a further P2 invalid-tail bypass (NaN log-CDF/log-SF could enter quadrature). A regression now requires rejection of non-finite-invalid tail inputs prior to fallback; this is a correctness correction, not a new legacy assumption. Final-head CI/re-review evidence is tracked on PR #18.
