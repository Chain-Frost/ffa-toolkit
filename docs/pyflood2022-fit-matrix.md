# Independent pyflood2022 all-eligible-record numerical stress matrix

Part A of [development requirements](https://github.com/Chain-Frost/ffa-toolkit/blob/main/docs/development.md),
bounded child [development requirements](https://github.com/Chain-Frost/ffa-toolkit/blob/main/docs/development.md).
Base `fff3208f167d07224b72395854c0a80626cb6dc8` incorporates the
accepted [#54 source audit](pyflood2022-validation.md). No production
algorithm, source licence or native qualification change.

## Population and execution

Use both pinned v5 archives and metadata exactly as in #54, including
byte size/Git blob identity. Require identical unique raw observations.
All source-ID-sorted sites with matching metadata, no duplicate years,
at least 30 finite representable positive nonconstant peaks are selected;
no post-fit replacement or removal. Record exclusion reasons and IDs.
Retain gaps; this is numerical fitting of the observed vector only, not
certification of complete exposure, stationarity or suitability for
hydrological design. Repeat each mathematical fit after multiplying
discharges and the log-model reference discharge by exactly 1024.

Ten variants are exercised: GEV H=0,1,2,3,4; Gumbel; zero-location
two-parameter lognormal fitted using raw-flow L1/L2; LP3 fitted using
natural-log flow L1–L3; free-location GP and explicitly fixed-zero GP.
GP cases are numerical AMS stress tests only, not evidence of a peaks-
over-threshold process or a recommended AMS distribution. All estimates
are deterministic moment fits, not Bayesian or optimized-H estimates.

The same isolated read-only workflow fetches the three pinned inputs
outside the checkout, runs the original audit, executes
`python -m scripts.validate_pyflood2022_fits <directory>`, and discards
the source files. If the preliminary numerical audit fails after the pinned
inputs have been fetched, the full matrix still runs its independent source
checks and records all outcomes without suppressing the earlier workflow
failure. No third-party code/data copied, no network dependency
in ordinary unit tests, no new dependencies, no raw rows in logs. The
aggregate receipt includes eligible IDs, exclusions and record counts,
per-method timings, checks, maximum error/tolerance ratios, every
nonpass station/reason and all admitted fits with observations outside
model support. All other cohort IDs are passes for the listed method.
A rejected fit is not counted as a numerical pass. A check exception is
not a fit rejection. For admitted fits, support is checked independently of
moments, the scaled refit, and each quantile/probability level. A failed
check does not erase completed tolerance ratios or known support violations.
Nonpass receipts identify check exceptions separately; support that could not
be established is explicitly unresolved. Completing the stress job means the experiment ran
and all outcomes are recorded; it does not mean every result is accepted.

## Mathematical oracles

Wang (1997), WRR 33(12), pp. 2841–2848, Appendix A1–A5
([DOI](https://doi.org/10.1029/97WR02134)) defines normalized PWMs as
expected maxima of samples of size r+1. Independently estimate each
maximum by averaging over ascending-rank combinations; take the
published alternating linear combinations for L1–L4. For GEV the
expected maximum of j independent values follows the transformed GEV
with location/scale determined by the same shape k; at unit scale and
zero location its expectation is (1-Gamma(1+k) j^(-k))/k. At k=0 it is
Euler's constant plus ln(j). Evaluate these maxima with a separate
small-k log-Gamma expansion and form the LH moments; match the first
three sample moments, compare all four reported theoretical moments,
and check the fourth ratio residual (not standardized Wang z).

For ordinary H=0 models, independent population equations are: Gumbel
L1=location+Euler*scale, L2=scale*ln(2); lognormal raw-flow L1 is its
mean, L2/L1=erf(log-flow-SD/2); log Pearson III log-space L2 is
SD/(sqrt(a)*Beta(a,1/2)) with a=4/skew^2 and signed L3/L2 from the
complementary beta identity, using the Normal limit for zero skew;
GP L1=location+scale/(1-xi), L2=scale/((1-xi)(2-xi)) and
L3/L2=(1+xi)/(3-xi). Match the moments actually consumed by the
estimator, including only two for fixed-location GP.

The sample reference is independent of the production adjacent-gap
sums. Population formulas independently reconstruct parameters'
moments; SciPy beta/gamma primitives may be shared, so they are not
independent library implementations. Compare F=0.5, 0.9, 0.99, 0.999
quantiles with direct analytic Gumbel, GP, lognormal equations and
SciPy GEV/Pearson III evaluations. The latter GEV path shares the
distribution backend used by the production bound kernel, but not
the fixed-H quantile equation. Also verify CDF and survival round-trips
and unit-scaled refit quantiles. The tests independently check known
order-statistic, exponential and Gumbel cases and inject a wrong
quantile to verify mismatch detection.

Admission tolerance is a numerical harness choice, not a published
engineering threshold: moment absolute error <= max(1e-10 times max
absolute fitting-coordinate sample, 1e-8 times absolute expected moment).
Raw fourth ratio uses absolute/relative 1e-8. Reference quantile uses
max(1e-12 times maximum raw peak, 1e-8 times absolute expected quantile).
CDF/SF use absolute 1e-11 or relative 1e-9. Unit refit uses absolute
1e-10 times maximum peak or relative 1e-8. Record ratios to these bounds;
nonfinite errors are mismatches, not missing successes. No tolerance
will be widened to make source results pass. Independently investigate
and track substantive mismatches rather than alter kernels here.

## Remaining

Published result identity, licensing/re-distribution, the documented
1,000-event count discrepancy, censored and jackknife conventions,
and matched MCMC/posterior comparison remain #55 Parts B/C. A separate
runtime correction, if warranted by this matrix, belongs to its method
owner. No native comparisons, package publication or Flike claim.

## Initial full-cohort execution receipt

[development requirements](https://github.com/Chain-Frost/ffa-toolkit/blob/main/docs/development.md)
at head `c040f3050c866fdcb1041b622d6919436a782e58` exercised 674
eligible stations containing 35,252 peaks; 369 eligible series have water-year
gaps. The other 420 stations have fewer than 30 records.

| Variant | Numerical passes | Fit rejections | Admitted fits outside support |
| --- | ---: | ---: | ---: |
| GEV H=0 | 674 | 0 | 1 |
| GEV H=1 | 674 | 0 | 5 |
| GEV H=2 | 674 | 0 | 7 |
| GEV H=3 | 674 | 0 | 11 |
| GEV H=4 | 674 | 0 | 22 |
| Gumbel | 674 | 0 | 0 |
| LN2 | 674 | 0 | 0 |
| LP3 | 563 | 111 | 0 |
| GP free location | 371 | 303 | 0 |
| GP fixed zero | 475 | 199 | 0 |
| Total (station-variant cases) | 6127 | 613 | 46 |

Every admitted fit passed the initial numerical checks; no numerical
mismatch or check exception was reported. All 613 fit rejections were
existing fitted-support checks: 111 LP3, 292 free-GP below the lower
endpoint, 11 free-GP above the upper endpoint, 199 fixed-zero GP above
the upper endpoint. The 46 GEV support exceptions are distinct
station-variant cases (not necessarily 46 distinct stations). They do
not certify satisfactory model fits; the method owner must decide how
to expose or reject them in the applicable fitting/report workflow.

The initial review identified two P2 harness issues (conditional execution
and diagnostic retention). The corrected workflow and added fault-injection
tests require a fresh complete execution and independent review; this
historical receipt is not final-head acceptance.
