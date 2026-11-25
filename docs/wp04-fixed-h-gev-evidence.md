# WP04 — independent fixed-H GEV L/LH estimator evidence

**Status:** partial implementation only. This module is neither an authenticated Flike 5.0.306.0 reference match nor a complete WP04 implementation. Related to [development requirements](https://github.com/Chain-Frost/ffa-toolkit/blob/main/docs/development.md) and [development requirements](https://github.com/Chain-Frost/ffa-toolkit/blob/main/docs/development.md).

## Implemented internal calculation

`src/ffa_methods/estimators/gev_lh.py` implements:

- Unbiased ordered-subset sample L-moments (H=0) and Wang LH-moments (H=1, 2, 3, 4), including sample fourth-moment diagnostics; formulas ARR Book 3 Chapter 2 equations 3.2.64–3.2.77.
- The first four theoretical GEV LH-moments in ARR Table 3.2.4, using `location=u`, `scale=alpha>0`, `shape=k>-1`, and `Q(F)=u+alpha/k * (1-(-ln(F))**k)`; its continuous Gumbel limit is evaluated explicitly. Here `F` is **non-exceedance** and annual exceedance probability is `1-F`.
- For each explicit fixed H=0..4, inversion of the theoretical third/second LH-moment ratio by a deterministic bracketed root; matching first and second moments gives scale and location. The raw fourth-to-second LH-moment ratio discrepancy is retained, **not** substituted for Wang (1998)'s standardized z statistic.
- ARR §2.6.4.6 parametric-bootstrap *research primitive* for a fixed H: sample complete series from the fitted GEV, refit at the same H, count unsuccessful replications, and report a chosen central interval. The caller must supply the seed, replication count, confidence level and non-exceedance probabilities. The currently implemented interpolation is type 7 and **is not a verified legacy rule**. These intervals are sampling-based bootstrap limits, not Bayesian posterior/expected-probability outputs.

The estimator accepts only an explicitly supplied sequence of complete, exact, equally weighted annual maximum discharges. No legacy flags, missing years, threshold groups, censored values, rating errors, priors or source projects are consumed or discarded by this module. The installed CLI exposes an explicitly configured fixed-H GEV mathematical fitting path; native Flike equivalence remains unqualified.

## Degeneracy correction and independent reference checks (23 November 2026)

The first version summed alternating floating-point multiples of observations.
For `[0] + [10]*9` at H=1 its second LH moment was exactly zero by the
defining U-statistic, but cancellation rounded it to a positive
`~1.58e-16` and a spurious GEV fit was accepted (Codex review finding
[development requirements](https://github.com/Chain-Frost/ffa-toolkit/blob/main/docs/development.md)).

The corrected estimator expands sorted observations into **adjacent spacings**
`d_i=x_(i+1)-x_i` and accumulates exact combinatorial counts before the
final floating-point products. For order `r>=2`, the coefficient of gap
`i` among `n` ascending observations is

```text
w_i^(H,r) = sum_{k=1}^{r-1} (-1)^(k-1) C(r-2,k-1)
              * C(n-i,k) * C(i,H+r-k) / (r * C(n,H+r))
lambda_r^(H) = sum_{i=1}^{n-1} w_i^(H,r) * (x_(i+1) - x_i)
```

These weights follow by counting how many of the `H+r` subset members
lie above each gap and using the partial alternating-binomial identity.
For `r=2`, all weights are **nonnegative**:
`(n-i) * C(i,H+1) / (2*C(n,H+2))`. A gap below the smallest
`H+1` admissible order statistics has weight exactly zero. For `r=1`,
the expected subset maximum is accumulated relative to the minimum
observation to preserve translation and avoid unnecessary cancellation.
There is **no magnitude-dependent epsilon or arbitrary minimum scale**.

Exact-rational enumeration of ordered subsets verifies ties, constant
records, positive one-ULP dispersion and all H=0..4; H=1..4 correctly
rejects the outlier/ties example. Affine transformations and a controlled
bootstrap with two deliberately degenerate refits verify fitted scales and
attempted/successful/failed counts without claiming that such failure rates
are native Flike rates.

The upstream [0BSD Numerics 2.2.0 pinned sources](https://github.com/USACE-RMC/Numerics/tree/7e8e8d1c5f26e045a35ec9fc09367de95ed05b02)
and independently calculated [generated vectors](../reference/public-candidates/numerics-vectors/ordinary-l-moments.json) supply three exact rational generated samples with `n=1292, 1293, 1460`. The tests pin the
binary64 sample SHA-256 values, source raw `L1..L4` and separately
normalize `tau3=L3/L2`, `tau4=L4/L2`. The generated cases retain the
original relative `1e-12` / absolute-floor `1e-13` target criterion.
This is H=0 **only**; independent ordered-subset tests cover H>0.
No competing candidate JSON copy is added to this implementation branch.

The exact population LH-skew root differs from a truncated polynomial estimator. Public tests
verify analytic Gumbel limits, shape conventions, independent subset moments, generated exact
rational moments and population recovery. The third-party textbook observation sample and all
derived fit outputs are excluded pending observation redistribution rights.

An explicit numerical inversion bracket `-0.999999 < k < 8` is used; a data sample outside this computational interval raises an error rather than receiving a clipped shape. The bracket is **not asserted to be a legacy setting or its full admissible domain**. Sample kurtosis requires `n >= H+4`; parameter fitting requires `n >= H+3`.

## Independent checks and an illustrative calculation

The tests compare sample LH formulas against the defining order-statistic U-statistics computed by enumerating all subsets; check location/scale equivariance, published Gumbel L-moment ratios, GEV quantile inversion and shape recovery, explicit invalid-data failures, and seeded bootstrap identity/counts. The population GEV LH formulas were also compared numerically against independently integrated order-statistic expectations using `scipy.stats.genextreme.ppf`, `scipy.stats.beta.pdf` and `scipy.integrate.quad` at H=0,1,3,4 and both signs of shape; maximum absolute difference in that local investigation was about `3.4e-9` discharge units.

Example with 12 complete illustrative annual maxima `[1,2,...,12]` in arbitrary discharge units:

| Fixed H | u (location) | alpha (scale) | k (shape) | F=0.99 quantile |
| --- | ---: | ---: | ---: | ---: |
| 0 | 5.152867 | 3.825618 | 0.283776 | 14.979791 |
| 1 | 5.146799 | 4.635743 | 0.481452 | 13.724202 |
| 2 | 4.991435 | 5.269184 | 0.590984 | 13.319236 |
| 3 | 4.798150 | 5.794983 | 0.661596 | 13.139693 |
| 4 | 4.600674 | 6.242470 | 0.711149 | 13.045518 |

This table tests reproducibility of the source-defined formulas; it is **not** a Flike matching case or an engineering design recommendation. Run focused tests with `python -m pytest tests/test_gev_lh.py` once the WP01 package is installed. The initial independent local run used Python 3.13.5 (46 tests passed); Python 3.14, Ruff and strict Pyright remain subject to repository CI.

## Bootstrap overflow acceptance (independent review follow-up)

A separate [development requirements](https://github.com/Chain-Frost/ffa-toolkit/blob/main/docs/development.md)
identified a finite high-range fit whose bootstrap refit at `F=0.99`
produced an **infinite** binary64 discharge. The original code counted
that attempt as successful and inserted infinity into its confidence limits.

The corrected primitive requires **all** requested initial quantiles to
be finite (or fails before sampling), and validates each simulated sample
and all requested refit quantiles before committing a replicate. Failure to
represent any required quantile is explicitly counted as a failed attempt
with a reason; no partial set of probability buckets is appended. The
source-reproducing `[1e307*i for i in range(1,13)]`, `seed=1`,
`10`-replication case must count **9 successful and 1 failed**
refit at `F=0.99`, with finite reported limits. An initially
unrepresentable `F=0.99999` quantile must raise explicitly.
These are binary64 numerical-validity rules for the independent research
primitive, **not** verified legacy Flike bootstrap controls.

## Remaining work

Ordinary H=0 family estimators are implemented internally; their installed CLI integration remains
outstanding. Standardized conditional-variance diagnostics, optimized-H selection and declared
uncertainty require independently reviewed specifications and tests. The fourth-ratio residual is not
a standardized z statistic. Native settings and comparisons remain a separate qualification task.

## References

- Commonwealth of Australia (Geoscience Australia), *Australian Rainfall and Runoff*, 2019, Version 4.2, Book 3 Chapter 2 §§2.6.4.2–2.6.4.6, equations 3.2.57–3.2.78 and Tables 3.2.3–3.2.5. https://www.arr-software.org/pdfs/ARR_190514_Book3_V4.2.pdf
- Hosking, J. R. M. (1990), *L-moments: Analysis and estimation of distributions using linear combinations of order statistics*, https://doi.org/10.1111/j.2517-6161.1990.tb01775.x
- Wang, Q. J. (1997), *LH moments for statistical analysis of extreme events*, https://doi.org/10.1029/97WR02134
- Wang, Q. J. (1998), *Approximate goodness-of-fit tests of fitted generalized extreme value distributions using LH moments*, https://doi.org/10.1029/98WR02364

## Fitted support reporting

The installed fixed-H CLI retains the algebraic L/LH estimate and reports
strict outside and exact endpoint observations separately. The comparison
uses the exact rational endpoint of the fitted ARR binary64 coordinates,
not a tolerance or a rounded endpoint. See the [installed method guidance
and synthetic examples](methods/lh-moments-gev.md). Wang (1998), p.3497,
distinguishes limitations of fitted-distribution tests beyond a bound;
no goodness-of-fit statistic, H-selection rule, likelihood admission,
or bootstrap rule is changed.

Independent synthetic checks cover finite upper/lower support, equality,
unbounded support, adjacent floating-point values, exactly representable
unit scaling and a finite endpoint beyond binary64 range. End-to-end
synthetic Year/AMS fits exercise all three reports and retain observations,
source hashes and fitted quantiles. External-record comparison remains
optional, outside the ordinary public test suite; no external dataset or
native qualification is required or asserted by this diagnostic.
