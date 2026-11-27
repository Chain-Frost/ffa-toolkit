# WP04: standardized LH diagnostics and optimized-H GEV — mathematical foundation

**Status: partial internal research primitives; no admitted fitted-GEV calibration, automatic
fit workflow, installed selector or uncertainty workflow.** [Issue #2](https://github.com/Chain-Frost/ffa-toolkit/issues/2)
owns the remaining requirements. Fixed-H GEV point fitting and its raw fourth-ratio residual remain unchanged.

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
The evaluation of test rejection rates includes n=30 and 60, shape -0.3 to +0.3, and nominal
5% and 1% levels; the test is approximate rather than an exact level guarantee.
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

## Internal software delivered in this slice

The separate internal module `estimators/gev_lh_diagnostics.py` evaluates the empirical
covariance, conditional variance and standardized residual. Its centered least-squares form
avoids subtracting nearly equal variance terms. It accepts already-paired simulated ratios;
the caller must authenticate their H, n, GEV shape and RNG configuration and qualify
precision. The primitive rejects fewer than three simulations, nonfinite ratios,
nonpositive marginal/conditional variance and nonfinite z. Three is an algebraic minimum,
not a recommended calibration count. Near-singular and finite-range behavior needs further
admission tests before production use.

The candidate decision primitive requires all five finite diagnostics in H=0..4 order,
plus a positive explicit two-sided critical z. Section 5 illustrates selecting the smallest
H that passes a two-sided 5% test; its examples select H=0, 1 and 4. This supports the
lowest-passing rule, not an automatic highest-H choice or a minimum-absolute-z rule.
For this independent policy, equality passes, an incomplete/invalid candidate set errors,
and an entirely rejected set returns no selection. Equality and fail-closed/no-pass
behaviors are explicit proposed Toolkit design decisions, not verified vendor rules.
No default alpha or cutoff is provided. The primitive does not compute candidate fits,
certify calibration, infer a native default or publish a selected flood quantile.

The 20 isolated focused tests passed under Python 3.13.5. They verify a hand-derived
rational covariance and z, affine invariance, degenerate/invalid calibrations, both
diagnostic signs, cutoff equality, lowest-H ordering, missing/invalid candidates and no
selection. These are algebra/policy tests, not full production candidate validation.
The checkout could not be cloned into the execution environment; tests were run against
the new files in an isolated directory, not the integrated repository. Python 3.11/3.14,
Ruff, strict Pyright, full tests, build, distribution/installed CLI and independent
review are still required. Draft status deliberately does not invoke ready-PR CI.

## Required before installed admission

1. Independently review the original equations and reproduce the synthetic calculation
   from the primary pages; retain the published tables outside the repository. Extend
   shape/sample-size coverage and establish explicit, justified calibration admission
   and a quantity-specific simulation precision/failure rule. Do not extrapolate
   from the small checked grid or assume the polynomial's finite-n calibration everywhere.
2. Generate paired ratios at each fitted H, record n, fitted skew/shape, simulation
   count, RNG/seeds, variance components and precision, and fail explicitly if
   calibration is inadmissible. Sample fourth moments require n>=H+4; all five
   candidates require n>=8 but this is not a calibrated minimum sample size.
   Verify degenerate/tied samples, numerical range, shape boundaries and independent
   fitted-GEV rejection-rate experiments (including decision stability near cutoff).
3. Fit each candidate using the existing GEV estimator; preserve sample/fitted tau3
   and tau4, conditional variance/SD, signed z, decision reason, support diagnostics,
   input identity and explicitly configured selection controls. Do not silently skip
   unavailable candidates or fall back to H=0. An all-rejected set has no selected fit.
   Coordinate shared settings/report contracts with #6, #7 and #14.
4. Complete explicitly configured uncertainty for ordinary L-moment, fixed-H and
   selected-H fits. Distinguish parametric-bootstrap sampling intervals from Bayesian
   posterior intervals and numerical simulation error. Declare generation distribution,
   refit/reselection per replicate, seeds, count, interval type, interpolation, failure
   denominators and acceptance thresholds; assess interval endpoint precision separately.
   Do not silently reuse fixed-H bootstrap for the selected-H workflow.
5. Provide installed CLI/report examples and full Python 3.11/3.14 quality and
   packaging checks plus independent final-head review. Keep #2 open until all its
   requirements are completed or explicitly dispositioned by the owner.

Native Flike 5.0.306.0 settings, source-specific variants, search/tie/no-pass handling,
bootstrap RNG/count/interpolation and settings-matched comparisons remain unqualified.
ARR example 10 (Albert River at Broomfleet) and Wang's example are source illustrations,
not native fixtures. No native match, release or publication is authorized.

## References

- Wang, Q. J. (1998), *Approximate goodness-of-fit tests of fitted generalized extreme
  value distributions using LH moments*, Water Resources Research 34(12), 3497–3502,
  [DOI](https://doi.org/10.1029/98WR02364), especially printed pp. 3498–3500.
- Wang (1997), *LH moments for statistical analysis of extreme events*,
  [DOI](https://doi.org/10.1029/97WR02134).
- ARR 2019 Book 3 v4.2, §§2.6.4.5, 2.6.4.6 and 2.8.10.3–4, equation 3.2.78,
  [Book 3](https://www.arr-software.org/pdfs/ARR_190514_Book3_V4.2.pdf).

Independent original implementation; no third-party code ported or licence change.
