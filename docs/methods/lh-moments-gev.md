# Fixed-H GEV ordinary L and LH mathematical fitting

**Method ID:** `lh-moments-gev`. Fixed H=0–4 point fitting, provisional direct-simulation selected H and optional parametric bootstrap are available for complete exact annual maxima. Native Flike qualification remains open.

## Purpose and required inputs

The installed `ffa fit` command fits one GEV distribution to a *complete, strictly positive, exact, equally weighted* annual-maximum series. It uses only the documented `Year,AMS` CSV layout already parsed by WP02 and an **explicit** `ffa-gev-fixed-h/1` JSON configuration. The caller must affirm that every row is a valid annual maximum for the declared `year_definition` and `discharge_unit`. That declaration is recorded, not externally authenticated.

Missing cells, reported zeroes, unknown native flags, excluded observations, historical/interval/threshold censoring and grouped exposure are **not** silently discarded or fitted. A file3.10 `.fld` project is not a valid input to this bounded CLI path. The choice of `lh_shift` and desired non-exceedance probabilities is required; no vendor/default estimator, H optimization, prior or bootstrap selection is inferred.

## Mathematical coordinates and estimator

For sorted exact annual peaks `q(1) <= ... <= q(n)`, define sample LH moments from unbiased combinations of order statistics on subsets of `H+r` data (`r=1,2,3,4`; `H=0` yields ordinary L moments). The production estimator accumulates adjacent data spacings with exact combinatorial counts, preserving meaningful zero and one-ULP dispersion. It fits the first three sample moments to the corresponding GEV population LH moments (ARR Book 3, Table 3.2.4; equations 3.2.64–3.2.77; Wang 1997).

The explicitly documented `k` convention is

```text
F = non-exceedance probability,  AEP = 1 - F
Q(F) = u + alpha/k * [1 - (-ln F)**k],  alpha > 0,  k > -1
k -> 0: Q(F) = u - alpha * ln(-ln F)
```

Parameters `u`, `alpha` and `k` are **deterministic point estimates**, not posterior means or Monte Carlo summaries. The fixed `H` is 0–4. No transform into unverified vendor shape sign, log-scale or zero atom is applied.

## Reproducible installed example

The package includes original, synthetic `resources/examples/fixed-gev-ams.csv` and `resources/examples/fixed-gev-settings.json` data. Copy those two files to a writable directory before running:

```console
ffa fit --ams fixed-gev-ams.csv --settings fixed-gev-settings.json --output fixed-gev-report
ffa explain lh-moments-gev --format markdown
```

The output directory must not exist and its parent must already exist. It contains `results.json`, `results.csv` and `report.txt`. These files include input/settings SHA-256 digests; explicit units, year definition, fixed H and probability choices; named point estimates; fitted quantiles; the actual sample LH moments and observation count; and, when available, the **unstandardized** fourth-to-second LH ratio residual. That raw residual is **not** Wang (1998)'s standardized optimized-H z statistic.

The sample is a synthetic mathematical example, **not** a native Flike run or flood-design recommendation. SHA-256 records the actual file bytes but does not authenticate the caller's source classification.

## Fitted support and interpretation

In ARR coordinates, the endpoint is `b = u + alpha/k` for nonzero k.
For k>0, the support is (-infinity, b]; for k<0 it is [b, infinity);
for k=0 both directions are unbounded. The finite endpoint is in the
closure of the support. Equality is distinct from strictly outside and
is not evidence of an interior observation or a finite likelihood density.

Every admitted deterministic fixed-H fit reports the endpoint direction,
exact rational endpoint of the fitted binary64 coordinates, approximate
binary64 endpoint (null if outside its representable range), unit, observation
count, and separate outside/at-endpoint counts and year/discharge identities.
Classification compares the exact mathematical endpoint to the actual
binary64 values consumed by the estimator; it does not compare to a rounded
display bound, add a tolerance, or round-trip scale through logarithms.
This is a diagnostic of the computed fit, not of uncertainty in the
parameters or original decimal-to-binary64 conversion. A finite endpoint
whose approximation is null is not an unbounded distribution.

The JSON field `gev_fitted_support`, the GEV-only CSV column
`gev_fitted_support_json` (the same JSON object on each quantile row), and
the fitted-support section of report.txt preserve the same finding. No
observation is removed, modified, censored or refitted. A valid algebraic
L/LH estimate remains reportable, even if some observations are strictly
outside and have zero exact continuous likelihood density. At an endpoint,
density limits depend on the shape; likelihood admission must separately
evaluate the chosen likelihood and its numerical/regularity conditions.
Wang (1998), printed p.3497, discusses the limitation of fitted-distribution
Anderson–Darling and Kolmogorov–Smirnov tests when observations fall beyond
bounds. His LH-kurtosis comparison is a different statistic. This diagnostic
does not redefine that statistic, select H, reject bootstrap replicates,
or certify engineering suitability. The existing bootstrap primitive is
not an installed reporting workflow and its rules remain unchanged.

For a synthetic upper-bound warning, create a Year,AMS file with consecutive
years 2000–2009 and flows 1,100,101,102,103,104,105,106,107,108; use the
bundled explicit H=0 settings above. The expected upper endpoint is about
106.56, with 2008 and 2009 outside. For a lower-bound warning use flows
1,2,3,4,5,6,7,8,9,100 with H=4; the endpoint is about 5.62, with years
2000–2004 outside. These are synthetic cases, not native examples.

## Uncertainty and limitations

## Selected H and parametric bootstrap

In fixed-H settings, replace `lh_shift` with `critical_z` and `calibration_seed`, set `schema_version` to `ffa-gev-selected-h/1` and set `estimator` to `gev_lh_selected_h`. All five fits use the same annual maxima. Wang's statistic divides the raw fourth-ratio residual by the simulated conditional standard deviation. Reports give each candidate's signed z, precision bounds, cutoff decision and fitted-support observations. The first passing H supplies the reported quantiles. If every candidate fails the cutoff, the report contains candidate results with no selected quantiles. An out-of-range fit or numerically unresolved decision gives an error. The provisional range is 20–60 observations and fitted ARR shape in [-0.8, 0.6] at every H.

For a reproducible synthetic example, create 50 annual maxima with year `1950+i` and discharge `20 - log(-log((i-0.35)/50.3))` for `i=1..50`. Use `critical_z` 1.96, `calibration_seed` 7 and non-exceedance probability 0.9. Run `ffa fit` with this CSV, the selected-H settings and a new output directory. The record produces five candidate diagnostics and selects H=0. The example uses regularly spaced theoretical Gumbel probabilities, so it illustrates the calculation rather than the frequency of selection in random records.

JSON and text contain the candidate diagnostics and support observations. CSV includes candidate support alongside the numerical results. An all-rejected report has one CSV row carrying support with empty result fields.

Either GEV schema accepts an optional `bootstrap` object with explicit `replications`, `seed`, `confidence`, `maximum_failure_fraction` and `endpoint_tolerance`. The last value is in the declared discharge unit. Each simulated record is generated by the original fitted GEV and refitted at the declared fixed H or with the entire five-H selection repeated. Selected-H simulations reuse the declared calibration seed inside each refit. Percentile limits use type-7 sample quantiles of the successful refits. The report lists failures and Monte Carlo bounds for each endpoint. A budget that cannot resolve an endpoint to the declared tolerance yields an error. The percentile limits express model-based sampling uncertainty, and the endpoint bounds express finite-simulation precision. See the [calibration and bootstrap specification](https://github.com/Chain-Frost/ffa-toolkit/blob/main/docs/wp04-optimized-h-specification.md) for gates and interpretation.

Negative mathematical predictions are reported with a warning. Physical support and engineering suitability require separate appraisal.

**Scientific foundations:** ARR 2019 Book 3 Chapter 2 §§2.6.4.5, 2.8.10.3, equations 3.2.64–3.2.77 and Table 3.2.4 (`ARR3`); Wang (1997), *LH moments for statistical analysis of extreme events* (`WANG1997`); Wang (1998) conditional variance for future standardized diagnostics (`WANG1998`). See `ffa references ARR3` and `ffa references WANG1997`.

See the [development requirements](../development.md) for the historical, censored, zero and Bayesian workflows and native qualification.
