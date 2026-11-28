# Explicit lognormal Bayesian file-to-report workflow

Tracking: [#6](https://github.com/Chain-Frost/ffa-toolkit/issues/6).
This implementation composes the existing WP06 target, curvature proposal,
explicit gamma scaling, weighted sampler and WP08 reports.
No native Flike comparison or engineering qualification is claimed.

## Source and settings contract

Use `ffa fit --ams INPUT.csv --settings SETTINGS.json --output NEW_DIRECTORY`.
The exact shared JSON decoder rejects duplicate keys and nonstandard constants.
All nested settings keys are explicit; no hidden defaults are selected.

- `schema_version`: `ffa-lognormal-bayesian/1`; `estimator`: `lognormal_bayesian`.
- `source_format`: `year_ams_csv` uses strict Year,AMS with positive exact records;
  `classified_systematic_csv` uses exactly `Year,Kind,Value,Lower,Upper`.
  Each row is a distinct positive integer systematic year. Exact records need
  a positive Value and empty bounds. Left/right censoring needs only the
  appropriate positive bound; interval censoring needs two ordered positive
  bounds. Optional retained positive measurement must be consistent with bounds.
  Explicit missing/excluded classifications carry no active censor bounds;
  missing carries no value. No inferred zero, historical exposure, grouped
  records, threshold law, native flags or rating correction.
- `source`, `discharge_unit`, `year_definition`, and the literal
  `caller_asserted_classified_systematic_annual_maxima` classification are
  caller declarations. Reference discharge is positive in the declared units.
- `fixed_log_sd` is the **natural log of the standard deviation** of
  ln(Q/reference), or explicit JSON null for estimated log-SD. A value of zero
  fixes the actual log-flow SD to one, not zero.
- `prior`: mode supplied, source, ordered coordinate names, means and full
  positive-definite covariance. Free coordinates are mean log-flow and, when
  estimated, log of the SD of log-flow. No default prior or prior Jacobian
  outside these declared coordinates is inferred.
- `proposal`: starting vector in prior coordinate order, gamma, scales and
  explicit local Newton / central-difference controls. Gamma multiplies
  proposal SD; covariance changes by gamma squared, target unchanged.
- `sampling`: exact RNG identifier python_random_gauss, count, uint64 seed,
  minimum ESS and maximum single weight. Those gates are not output precision.
- `outputs`: strictly increasing non-exceedance levels and positive fixed
  discharges (one or both tables; at most 200 combined), central interval mass.
- `precision`: either {"mode":"not_assessed"} or an explicit fixed_budget
  object as in the fixed example. Requests identify a meaning, tolerance in
  that output's unit, and all applicable axes; unused axes are JSON null.
  Parameter mean requests apply to the single known-scale free coordinate.
  No adaptive stopping. The declared total draw and scalar-evaluation budgets
  reserve a report run in addition to the unchanged audit schedule (the audit
  replays the same primary seed; independent seeds are never pooled). The
  report reservation is a conservative scalar-output accounting rule, not a
  wall-time limit or a cap on optimization work.

The report includes exact input/settings snapshots verified against SHA-256,
source coordinates, separate estimands, seed and actual stream description,
proposal mean/covariance and local diagnostics. For selected requests the
JSON contains the full precision attempt ledger, admission, criteria and
per-request states. CSV and text show result-point precision states. JSON and text
also show parameter-mean precision states. The ledger shows whether the primary
replay was attempted. A conditional
pass applies only to the audited primary estimate; a different numerical
result path cannot inherit it. Unrequested outputs remain not_assessed.
Existing deterministic report schemas and contents are retained. Existing
destinations, parent traversal, symbolic links and Windows reparse points are
rejected before calculation and checked again at publication. Publish into an
output parent you control. The staged rename is not a portable atomic no-replace
operation against a directory inserted between the final check and rename.

## Mathematical basis and independent references

See ARR Book 3 v4.2 Chapter 2 Â§Â§2.6.3.6â€“7, Eqs 3.2.52â€“54 and 3.2.56;
Kuczera (1999), p. 1554, Eqs (16)â€“(17);
[importance sampling](methods/importance-sampling-reference.md),
[lognormal binding](wp06-lognormal-binding.md),
[output precision](wp06-output-precision.md), and the
[reference register](references.md). These are the repository's previously
reviewed mathematical sources. The orchestration adds no new inference law.
With y = ln(Q/reference), the likelihood for an exact positive flow has
normal density in y divided by Q. A left/right/interval-censored year uses
normal CDF, survival, or interval probability, respectively. A supplied
Gaussian prior is defined directly in the free coordinates. The normalized
weights are proportional to likelihood times prior divided by the scaled
proposal density. Quantile at the mean coordinates, mean/median of per-draw
quantiles, inverse expected CDF, expected AEP and reciprocal expected AEP
remain distinct.

In the synthetic known-scale example, log flows 1, 2, 3, SD 1 and prior
N(0,1) imply posterior N(1.5, 0.25) by completing the square. Predictive log
flow is N(1.5, 1.25). At non-exceedance p, z = Phi^-1(p), the expected-coordinate
quantile is exp(1.5+z), the posterior mean is exp(1.5+z+0.125) and the predictive
quantile exp(1.5+sqrt(1.25)*z). AEP at exp(2) is 1-Phi((2-1.5)/sqrt(1.25)).
The example audits only the expected AEP, with deliberately synthetic
criteria and a broader proposal, not an engineering tolerance.

The estimated-scale censored example has exact log flows -0.4 and 0.5,
left censor at -0.2, right censor at 0.1, independent N(0,0.16) mean and
N(0,0.0144) log-SD priors. Independent midpoint quadrature of the product
of the two Normal densities, left CDF and right survival probabilities
and prior densities, with 81x81 and 161x161 grids over +/-9 prior SDs,
gives posterior means 0.003598304886871672 and -0.01338564667751898 and
expected AEP at exp(0.35) of 0.3686203803796388 (grid agreement better than
1e-12). Independent approximate inverse-curvature covariance
[[0.102946298,-0.000059604],[-0.000059604,0.0141626883]] gives quadrature
SNIS variance coefficients 0.106215, 0.014182 and 0.013989. Six reference
asymptotic standard errors at 4000 draws, rounded outward, are 0.032,
0.012 and 0.012. These fixed synthetic comparison tolerances are not
precision admission, universal guarantees or native tolerances. The
existing binding tests also integrate the independent reference with the
actual proposal. Under the full Gaussian estimated-log-SD prior, for F>0.5
the design quantile's posterior mean diverges; the CLI omits it explicitly
without calculating a finite simulated replacement. Other estimated/censored
precision remains unadmitted by the existing audit.

## Example and acceptance

The four files under [examples](examples/) are generated mathematical data,
also included in wheel and sdist. Run both command examples from
[quantile estimands](methods/quantile-estimands.md). Endpoint
tests cover the above independent references, source hashes and exact bytes,
distinct estimands, reproducibility, undefined mean omission, invalid
settings and input, fixed-budget acceptance and existing output rejection.
See [PR #37](https://github.com/Chain-Frost/ffa-toolkit/pull/37) for exact-head
acceptance evidence.

## Integration coordination

Base `affc0315e96978c22930cdb0776f90fca2f9528b` includes #31. Concurrent PRs #32/#34/#35 update the registry, #35 updates the calculation matrix, and #36 updates the CLI, formatter, packaging, installed smoke and guidance. Reconcile these shared files against the actual latest main before validation. Preserve #36's deterministic-only plot admission, the Bayesian output-specific audit and both sets of example resources and tests. No existing branch or main has been overwritten.
