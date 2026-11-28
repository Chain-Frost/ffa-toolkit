# Ordinary L-moment fitting (H=0)

**Method ID:** `l-moments`

Independent deterministic mathematical fits for stationary complete exact positive annual maxima. The point fits report fitted-family quantiles and can also request the parametric bootstrap described below. No prior or rating-error law is used. The caller asserts the year definition, physical unit and complete-record classification. No observation is silently discarded; missing, zero, negative, duplicate or otherwise unsupported records fail. Omitted years are not inferred to be non-exceedances. A GP fit to annual maxima is **not** a peaks-over-threshold workflow.

## Coordinates and mathematical meaning

| Estimator | Sample moments | Reported coordinates |
| --- | --- | --- |
| gumbel_lmoments | Raw Q, L1/L2 | Location and positive scale in declared flow units |
| lognormal_lmoments | Raw Q, L1/L2 | LN2, zero location; mean of ln(Q/reference), ln(SD of ln(Q/reference)) |
| lp3_lmoments | ln(Q/reference), L1/L2/L3 | Natural-log mean, natural log of log-flow SD, signed log-flow skew |
| gp_lmoments | Raw Q, L1/L2 for fixed location, L1/L2/L3 for free location | Location and scale in flow units; EVT xi = negative ARR/Hosking kappa |

The reference discharge must be explicitly positive and in the same units as Q; it is a coordinate choice, not a fitted location. Changing reference changes the log mean, not the physical fitted quantile. Changing discharge units consistently requires rescaling the reference and any fixed GP location. LP3 exact gamma-shape inversion has an admitted numerical bracket [0.01, 1e8]; exact zero skew is its Normal limit, not a clipping rule. GP fixed/free configurations must respect the kernel's finite-mean and fitted-support constraints; LP3/GP reject unsupported fits according to their existing mathematical policies. Gumbel mathematical quantiles may be negative: they are preserved and flagged, not clipped.

For Gumbel, scale=L2/ln(2), location=L1-EulerGamma*scale. LN2 matches L2/L1=erf(sigma_ln/2) and L1=reference*exp(mu+sigma_ln²/2). For LP3, use the exact signed gamma L-skew identity tau3=sign(g)*(6 I_(1/3)(a,2a)-3), a=4/g². GP uses ARR kappa=(L1-location)/L2-2 for fixed location, or (1-3tau3)/(1+tau3) for free location; scale=(1+kappa)(L1-location), location=L1-(2+kappa)L2 in the free case. No native shape or location default is inferred.

## Explicit settings and worked example

Run the installed CLI outside the repository using the bundled synthetic `ordinary-ams.csv` and `ordinary-settings.json`:

~~~console
ffa fit --ams ordinary-ams.csv --settings ordinary-settings.json --output gumbel-report
~~~

Copy resources from `importlib.resources.files("ffa_methods").joinpath("resources", "examples")` if using an installed wheel. The three years have discharges [1,2,4]; raw sample L1=7/3 and L2=1. The bundled JSON has estimator `gumbel_lmoments`. For each separate output directory, change only the estimator and add the applicable required fields:

| Estimator | Additional required fields |
| --- | --- |
| gumbel_lmoments | none |
| lognormal_lmoments | `"reference_discharge": 2` |
| lp3_lmoments | `"reference_discharge": 2` |
| gp_lmoments (fixed) | `"gp_location_mode": "fixed", "gp_location": 0.25` |
| gp_lmoments (free) | `"gp_location_mode": "free"` |

The schema is `ffa-ordinary-lmoments/1`; all keys are exact and family-inapplicable keys fail. H=0 is inherent; do not add lh_shift. F probabilities are strictly increasing, finite and interior to (0,1). AEP=1-F for annual maxima. The existing `ffa-gev-fixed-h/1` settings remain unchanged. At F=0.5 the GP free example has location=1/3, scale=2, xi=0 and quantile=1/3+2 ln(2); fixed-location GP has kappa=1/12 and scale=325/144. The LP3 log-geometric sample has mean=0 relative to 2, skew=0 and log-flow SD=sqrt(pi)*2 ln(2)/3. These are independent synthetic identities, not native benchmarks.

Read `results.json`, `results.csv` and `report.txt`. Parameter summaries are `point_estimate` and fitted quantiles are `deterministic_fitted_quantile`. The ordinary report also includes byte-exact `source-ams.csv` and `source-settings.json`, copied into the staged bundle. Their hashes match the analysed inputs. The output directory must be new.

An optional `bootstrap` object configures parametric percentile intervals. For example, add `"bootstrap": {"replications": 2000, "seed": 42, "confidence": 0.9, "maximum_failure_fraction": 0, "endpoint_tolerance": 5}` to the ordinary Gumbel settings. The tolerance is in the declared discharge unit and must reflect the desired endpoint precision. The original fitted family generates a complete record of the original length. Each refit repeats the same family, reference discharge and GP location mode. Failed trials count against the declared maximum fraction, and limits use the successful refits. The report provides rank-based Monte Carlo bounds on each endpoint separately from the percentile sampling interval. See the [bootstrap specification](https://github.com/Chain-Frost/ffa-toolkit/blob/main/docs/wp04-optimized-h-specification.md).

The report retains every attempted refit in simulation order. If the failure limit or endpoint precision requirement is unmet, the CLI exits with status 2 and saves the point-fit bundle and attempt ledger without interval limits. With the packaged ordinary synthetic data and these settings, the 10% AEP point fit is approximately 4.7472 m³/s. The 90% bootstrap percentile limits are approximately 1.9497 and 9.0002 m³/s, with 2,000 successful refits.

## Basis and limits

ARR 2019 Book 3 v4.2 §2.6.4, Table 3.2.3 and §2.8.10.3; Hosking (1990), DOI 10.1111/j.2517-6161.1990.tb01775.x; Hosking and Wallis (1987, 1997). Use `ffa references ARR3` and `ffa references HOSKING1990`. See repository `docs/wp04-ordinary-lmoments-evidence.md` for reviewed numerical boundaries and source provenance. Independence/stationarity and suitability require analyst assessment; native defaults and compatibility remain unqualified. Native qualification is separate from these independent mathematical calculations.
