# WP04: standardized LH diagnostics and optimized-H GEV — mathematical foundation

**Status: provisional direct calibration and explicit CLI workflow.**
[Issue #2](https://github.com/Chain-Frost/ffa-toolkit/issues/2) tracks qualification and independent review.

## Primary-source verification

Wang (1998), printed pp. 3498–3500, equations (3)–(7) and Section 5, was inspected from the
original article. The mathematical formulation, rather than the paper's coefficient tables,
is used here. For a fixed H in 0..4, fit GEV to the first three sample LH moments using the
existing estimator. The sample and fitted fourth-to-second moment ratios are the sample and
population LH kurtoses, respectively. The third-to-second ratios represent LH skewness.
Wang assumes approximately jointly normal sample ratios, neglects their bias, and evaluates
their conditional distribution at sample skewness equal to population skewness. Matching the
sample first three moments gives the plug-in fitted population skewness.

Let V3 and V4 denote the sampling variances of the two ratios under this fitted GEV for the
same sample size n and shift H, and C34 their covariance. Equation (3), squared, gives

    conditional_variance = V4 * (1 - correlation**2) = V4 - C34**2 / V3
    z = (sample_LH_kurtosis - fitted_LH_kurtosis) / sqrt(conditional_variance)

Equation (4) is the z statistic. The denominator is not the raw fourth-ratio residual, the
sample LH scale, an unconditional kurtosis SD or a bootstrap confidence limit. Equations
(5)–(7) provide a fitted approximation of the form b(tau3)/n + c(tau3)/n**2, where both
functions are fourth-degree polynomials in population LH skew and depend on H. The article
derived them from simulations of 100,000 complete records per size and shape, at n=20 and
50 and ARR-coordinate shape k=-0.8 to +0.6 at 0.05 increments, separately for H=0..4.
Wang evaluates nominal 5% rejection at n=30 and 60 and nominal 1% rejection at n=30,
using shapes -0.3 to +0.3. The test is approximate.
The printed table captions and the equations in the text use inconsistent cross-references;
the body of equations (6) and (7) identifies b and c unambiguously.

Neither the published tables nor the paper, observed sample data or derived native artifacts
are admitted to this repository. The alternative direct paired-ratio simulation follows the
covariance construction in equation (3); it is not a claim to reproduce the polynomial
approximation identically. It has not yet been admitted as a production calibration method.

## Independent numerical research, 28 November 2026

An independent synthetic calculation checked 30 H/n/shape combinations:
H=0..4, n in {20,50}, k in {-0.3,0,0.3}, location 0, scale 1. Each n/shape case
used two separate 100,000-record NumPy PCG64 batches with seeds 20261128 and 20261129.
The same records were evaluated for all five H; seeds were reused across n/shape cases
(common random numbers, not independent cross-cell experiments). Python 3.13.5,
NumPy 2.3.5 and SciPy 1.17.0 were used. No native executable or observed records were used.

The independent reference used direct ordered-subset rank weights rather than production
adjacent-gap weights. For zero-based rank i, r in {2,3,4}, and m=H+r, the coefficient is

    w_i = sum((-1)**j * C(r-1,j) * C(i,m-j-1) * C(n-1-i,j)
              for j in range(r)) / (r * C(n,m))

with inadmissible binomial indices treated as zero. Generate open-interval uniform draws,
transform via the ARR-coordinate GEV inverse CDF (continuous Gumbel limit at k=0), sort
each record and multiply by these weights. Divide LH3 and LH4 by LH2. Independently
integrate the population LH moments from beta order-statistic densities applied to
SciPy's genextreme quantile to obtain population skew. Calculate sample covariance with
B-1 denominator and compare its Schur complement with the centered least-squares residual
sum divided by B-1. For the private primary-paper comparison only, evaluate equations
(5)–(7) at the independently integrated population skew and compare against the average
of the two direct simulated variance estimates. The results were:

- Maximum relative discrepancy of the two algebraic covariance expressions: less than 5e-13.
- Maximum absolute relative discrepancy between published polynomial variance and average
  direct simulated variance: less than 0.93%.
- Maximum absolute batch difference divided by their mean: less than 1.45%.

These are observed finite-grid research comparisons, not an adopted tolerance, precision
certificate, independent reviewer sign-off, shape-range guarantee or test-size validation.
The restricted coefficients were used only for the private comparison; no coefficients,
source vectors or publication-derived fixtures are shipped. Reproduce the study with the
described original procedure and a separately held copy of the article. The wider grid,
independent held-out calibration and numerical precision policy remain outstanding.

## Direct fitted-GEV calibration

For each fitted H, simulate 20 batches of 10,000 complete unit-scale, zero-location
GEV records at its fitted ARR shape. Positive affine transformations leave the two
ratios invariant. NumPy PCG64 uses the declared base seed plus H. The pooled sample
covariance gives the point variance. An approximate Student t batch-means interval
allocates 1% across the two tails and five candidates. The interval is widened by
the absolute pooled versus batch-mean difference. It measures Monte Carlo precision
under this simulation design. It does not measure uncertainty about the observed flood
record or certify the normal approximation.

Require 20–60 annual maxima and fitted k in [-0.8, 0.6] for every H. These provisional
limits span the record lengths and simulation shape range in Wang's specification
and evaluation. The simulation evaluates the actual record length and fitted shape,
without interpolating the paper's polynomials. Reject a nonpositive or nonfinite
variance bracket, a variance half-width exceeding 2% of its pooled estimate, or an
absolute-z bracket that straddles the declared cutoff. A 2% variance error corresponds
to at most approximately 1.02% error in the magnitude of z. The cutoff check addresses
decisions for which even that precision is insufficient. No automatic seed retries.

The covariance implementations subtract an anchor before centering so that rounding
of a mean at a common offset does not dominate the centered ratios. The regression
uses the squared norm of the residual vector. Reject when its norm is no greater
than 32 binary64 epsilons times the centered kurtosis norm. Centering, covariance
accumulation, division and subtraction each contribute roundoff at the centered
scale. The factor of 32 gives those operations headroom and the relative rule is
invariant to rescaling kurtosis. This is a numerical rank guard, with no floor
substituted for the variance and no change to the 2% Monte Carlo requirement.
The exactly collinear dyadic record from the independent review and a one-ulp-scale
perturbation both trigger it in the scalar and simulation paths.

An independent rerun on Python 3.14.6 calculated the ratios from ordered ranks and
the population skew by integrating beta order-statistic densities. It reproduced
maximum discrepancies of 0.9175% (polynomial against direct), 1.4405% (between
batches), and 4.0e-14 (regression versus covariance). Extending the two independent
batches to n=60 gave differences up to 5.54% between the polynomial and direct
variance. The polynomial was lower in 14 of 15 n=60 cells. Thus matching the polynomial
is not the acceptance criterion for direct calibration at n=60. The 100,000-record
prototype exceeded the precision target on the predeclared seed grid, so the
production candidate doubles the budget without relaxing that target.
At 200,000 records, 199 of 200 calibration runs met the criterion on n=20,30,50,60,
k=-0.8,-0.3,0,0.3,0.6 and H=0–4 with base seeds 20261128 and 20261129. The largest
accepted relative half-width was 1.879%. The n=20, k=-0.8, H=4 result at base seed
20261128 failed at 2.040%. These runs exercised the precision gate without changing
the seed or replacing a failure.

A held-out experiment used 100,000 new records at each n/shape cell, with n=20,30,60
and true k=-0.3,0,0.3. Independent order-statistic weights and population beta-density
moments were used to refit shape. The research variance oracle interpolated 50,000-record
simulations on a 0.01 shape grid. The study used PCG64 seeds 225829 for records and
638273 for the variance oracle. It excluded records with any fitted shape outside the
provisional range and omitted finite-budget precision gating. At n=30 and true k=0,
96.88% met the shape restriction. Candidate rejection rates among those records were
3.96%, 4.08%, 4.37%, 4.62% and 4.71% at the conventional 1.96 cutoff. At n=20 and true
k=0.3, 69.86% met the shape restriction. A cutoff describes an individual approximate
check, with no nominal error guarantee after conditioning or selection.

## Selection and uncertainty

Fit every H=0..4 using the existing LH estimator. The smallest H with absolute z
at most the configured critical value is selected when all five decisions are
numerically stable. Equality passes at the point-rule level. A domain refusal
lists the out-of-range fitted candidates and prevents calibration. A precision
refusal identifies the candidate and failed Monte Carlo gate. Numerical failures
identify the candidate and fit or calibration stage. A complete all-rejected
record reports `selection=no_pass`, retains diagnostics and support observations,
and emits no selected quantile. Support findings are independent of the cutoff.

The optional complete-record parametric bootstrap uses the fitted distribution
as its generating law. Each equal-length synthetic record repeats the declared
ordinary family/configuration, fixed H, or full five-H selection. For selected H,
the calibration seed is held fixed across replicates, as part of the procedure
being repeated. The bootstrap record generator uses a separate declared MT19937
seed. Simulated discharges are drawn from the fitted law without clipping; refits
apply the estimator's own domain rules. Fit, calibration and no-pass failures are
counted against all attempted records. Each attempt appears once in simulation order
in the JSON array encoded in `settings.bootstrap_attempts`. Each row records the
selected H where applicable and any failed stage and reason.

Report central type-7 percentile intervals over the successful replicate quantiles
at each explicitly requested non-exceedance probability. A positive admitted failure
fraction changes the target to the conditional distribution of successful refits.
For each interval endpoint, use theoretical binomial ranks evaluated at the
corresponding tail probability to bracket its finite-simulation error. Allocate 5%
across both bracket tails, both endpoints and all requested probabilities. A bracket
that cannot be supported by the successful count or whose distance from the type-7
estimate exceeds the caller's absolute discharge tolerance prevents interval
publication. The CLI returns status 2 and saves the point-fit bundle and attempt
ledger on interval refusal. The report settings identify the gate as
`failure_fraction`, `endpoint_count` or `endpoint_precision`. These numerical
brackets concern the endpoint estimates, while the percentile interval expresses
model-based sampling variability.

## Qualification

The [held-out direct-calibration evidence](wp04-direct-lh-held-out-validation.md)
covers record lengths 20 and 60, all-candidate eligibility, precision refusals
and decision stability. The [bootstrap worked example](wp04-bootstrap-evidence.md)
records separate ordinary, fixed-H and selected-H executions.

The direct method retains Wang's joint-normal and negligible-bias approximation.
The operational precision rule addresses calculation stability. Test rejection rates
and independent review are separate requirements. Native Flike 5.0.306.0 selection
and bootstrap settings and settings-matched comparisons remain unqualified.

## References

- Wang, Q. J. (1998), *Approximate goodness-of-fit tests of fitted generalized extreme
  value distributions using LH moments*, Water Resources Research 34(12), 3497–3502,
  [DOI](https://doi.org/10.1029/98WR02364), especially printed pp. 3498–3500.
- Wang (1997), *LH moments for statistical analysis of extreme events*,
  [DOI](https://doi.org/10.1029/97WR02134).
- ARR 2019 Book 3 v4.2, §§2.6.4.5, 2.6.4.6 and 2.8.10.3–4, equation 3.2.78,
  [Book 3](https://www.arr-software.org/pdfs/ARR_190514_Book3_V4.2.pdf).

Independent original implementation; no third-party code ported or licence change.
Calibration imports the NumPy runtime already required by SciPy. The checked NumPy
2.4.6 wheel declares BSD-3-Clause with bundled components under 0BSD, MIT, Zlib and
CC0-1.0. Toolkit distributions depend on that runtime separately and carry no copied
NumPy code or data.
