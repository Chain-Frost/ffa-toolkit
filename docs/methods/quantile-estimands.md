# Distinct quantile estimands and probability-limit meanings

**Method ID:** `quantile-estimands`. **Legacy target:** Flike 5.0.306.0, file 3.10. **Qualification:** the WP06 weighted-output mathematical component and WP08 internal formatter do not constitute an installed legacy fitting engine or an admitted native comparison.

## Practical purpose: what does a flood estimate mean?

Let Q be an annual-maximum discharge, `F(q|theta)=P(Q<=q|theta)`, and `AEP(q|theta)=1-F(q|theta)`. The p-AEP quantile is `q_p(theta)=F^-1(1-p|theta)`. AEP is **not** F; annual return period for nonzero p is `1/p`, not `1/F`. Some candidate legacy plot headings say “AEP plot position” while their numbers are consistent with F. Preserve both **raw heading** and **interpreted probability kind**, and do not claim this mapping applies to every source file without validation.

## Four estimands that must not be silently substituted

Let posterior parameter distribution be `pi(theta|D)`. The expected-parameter operation uses the **reported source parameter coordinates** (for LP3, natural-log mean, log of natural-log standard deviation, and skew), not a convenient transformed vector.

| Quantity | Mathematical definition | Meaning |
| --- | --- | --- |
| Quantile at expected parameters | `F^-1(1-p | E[theta|D])` | Quantile of the law at mean source parameters; **not** mean quantile |
| Mean/median posterior quantile | `E[q_p(theta)|D]` / posterior median | Summary of the random quantile across parameter uncertainty |
| Expected F at fixed q | `E[F(q|theta)|D]` | Average non-exceedance probability; expected AEP is its complement |
| Predictive inverse | `inf{q: E[F(q|theta)|D] >= 1-p}` | Quantile from the averaged CDF, generally different from every row above |

A central or one-sided probability limit for `q_p(theta)` must identify its **level, lower/upper side, construction and target random quantity**. A Bayesian pointwise posterior limit is not automatically a bootstrap interval, frequentist confidence interval or simultaneous band. Monte Carlo integration error is separate from flood uncertainty.

Moments of `log10(q_p)` are **not** the natural-log LP3 parameter coordinates. A report must identify whether the output is the **mean, variance or standard deviation** of the base-10 logarithm and declare the discharge reference unit.

## Required inputs, status and diagnostics

The input probability must carry an explicit F/AEP kind. Expected F requires the **input discharge and its unit**, not an input AEP. Quantile outputs require a probability, output discharge unit and explicit estimand. For a probability limit, record side, level and method; for a log10 moment, record moment kind. Where available, inference should report source settings, parameter weights/ESS, convergence and precision separately from posterior flood variability.

The current CLI cannot fit source files; formatter output is always explicitly **not legacy-qualified** even when supplied provenance calls a case “qualified”. The RMC ModeCurve comparison-side interpretation does not establish historical Flike target estimands. Historical/censored plots require actual exposure and ARR §2.6.2, not re-ranking only uncensored years.

**Mathematical basis:** ARR 2019 Book 3 Chapter 2 §2.5 equations 3.2.25–3.2.34 and §§2.6.3.6–2.6.3.7 equations 3.2.52–3.2.56; Kuczera (1999), DOI 10.1029/1999WR900012 (`ARR3`, `KUCZERA1999`).

**Observed candidate tables:** expected-parameter rows, 90% Monte Carlo limits and expected-probability/return-period columns (`LEGACY_SAMPLES`), without verified settings/interval interpolation. **Project decision:** preserve estimand, source-coordinate and base-10/probability distinctions. **Remaining:** fitted engine, interval interpolation, native output matching and independent acceptance. Citations are included in the installed `ffa references` command.

## Explicit lognormal Bayesian CLI examples

The versioned `ffa-lognormal-bayesian/1` workflow composes the existing
classified lognormal binding, local curvature, full covariance scaling,
actual independent importance sampler and typed WP06 output bridge. It
does not choose native Flike defaults. From a checkout, run:

```console
ffa fit --ams docs/examples/bayesian-fixed-ams.csv --settings docs/examples/bayesian-fixed-settings.json --output fixed-bayes
ffa fit --ams docs/examples/bayesian-censored-ams.csv --settings docs/examples/bayesian-censored-settings.json --output censored-bayes
```

The four example files are also packaged in `ffa_methods/resources/examples/`.
The full versioned settings contract, derivation and synthetic reference
checks are in [Bayesian CLI workflow](../wp08-bayesian-cli.md).
The fixed example audits its expected AEP only, with explicit example
tolerances; all other results remain unassessed. The estimated-scale
censored example has no output precision admission. For the full Gaussian
estimated-log-SD prior the mean design quantile at F > 0.5 is divergent,
so that particular estimand is explicitly omitted rather than replaced by
a finite simulated average. Other estimands are distinct, not substitutes.
Neither example establishes native equivalence, engineering suitability
or general tail precision.
