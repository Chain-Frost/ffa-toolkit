# WP06 — Censored and estimated-scale precision: mathematical eligibility

Refs [#5](https://github.com/Chain-Frost/ffa-toolkit/issues/5), #3, #4, #6 and #7.
Base: merged PR #31, `affc0315e96978c22930cdb0776f90fca2f9528b`.
This is an additive verification candidate, not general runtime admission.

## Global sufficient moment condition

Use the model and output definitions in [the lognormal binding](wp06-lognormal-binding.md)
and [the precision contract](wp06-output-precision.md), including their ARR Book 3
§2.6.3.6–7 and Owen §9.2 source basis. The argument below is an independent
derivation, not a new vendor setting or numerical default.

Let the supplied full Gaussian prior be N(m,P), and the explicit Gaussian
proposal be N(a,G), in the same free coordinates. With n exact observations,
the likelihood is bounded above by C exp(-n eta) in the estimated-scale
case (C is finite and positive for fixed positive recorded discharges).
At fixed scale the corresponding bound is a finite constant. Every
positive fixed censor probability is at most 1, including finite interval
censoring, and excluded/missing rows contribute no factor. The target is
positive and proper because a Gaussian integrates any linear exponential.

The unnormalized squared importance ratio integrated against g is
the integral of L² prior²/g. Its log is bounded above by a constant,
a linear term and the quadratic

    -1/2 theta' (2 P^-1 - G^-1) theta.

If G - P/2 is strictly positive definite, the matrix in parentheses
is strictly positive definite by inversion order. The integrable
Gaussian quadratic dominates any linear exponential times a polynomial.
This suffices for both E_g[r²] and E_g[(r h)²] for the finite scalar
routes below; normalization by Z>0 does not change finiteness.

- CDF, direct survival and fixed-point posterior indicators are bounded.
  The reciprocal expected AEP uses a direct survival mean, not the
  potentially divergent mean of individual reciprocal survivals.
- Coordinate means are polynomial. A log10 design quantile is affine
  in mu and exp(eta); the required centered second-moment integrand
  squared grows at most polynomially in mu times a linear exponential
  in eta. The same bound establishes the relevant posterior moments.
- Fixed-scale design quantile means are proportional to exp(mu).
  For estimated scale and p<=0.5, q_p <= reference * exp(mu); their
  required posterior and importance second moments are finite under
  the stated sufficient condition.
- For estimated scale and p>0.5 the posterior mean of q_p diverges:
  retain the strip lower-bound proof in the precision contract.
  No covariance choice, sample size or finite estimate changes this.
- Quantiles at expected parameters have finite coordinate means in
  real arithmetic, but this is not a proof of nonlinear error
  propagation. Inverse outputs use fixed-point scalar integrands,
  not an asserted quantile CLT or a mean of individual inverse values.

Failure of the covariance condition establishes neither divergence nor
inadequacy of every proposal; likelihood decay may rescue a narrower
proposal. Equality is not admitted. Marginal variance tests are
insufficient with correlated coordinates. The internal
`importance_moment_domain` helper checks the exact rational signs of the
stored binary64 1x1/2x2 symmetric matrices P, G and 2G-P; it does not
round an approximately zero eigenvalue into a proof. It does not alter
the proposal, target, existing audit or precision state. It does not
establish a normalized density-ratio cap, a practical tail-resolution
threshold, representability, convergence rate or a finite-sample CI.

## Original synthetic verification

Two exact log flows (-0.4, 0.5), left censor -0.2, right censor 0.1,
interval censor (-0.5,0.5), explicit missing/excluded records; reference
discharge 1. Prior mu~N(0,0.16), and for estimated scale independently
eta~N(0,0.0144). Fixed sigma=1 otherwise. At fixed scale g is the prior.
For the estimated-scale reference g is the prior tilted by exp(-2 eta):
its eta mean is -0.0288 and covariance is unchanged. In real arithmetic
the tilted reference makes scaled unnormalized weights no greater than 1.
The proof of second moments does not rely on exact floating-point tilt
equality or on that stronger bound.

Reference integrates independent Normal density/censor formulas and
the ratio variance coefficient E_g[W²(h-E_pi[h])²]/E_g[W]². The
81/161 midpoint grids span nine proposal SDs in each coordinate.
For posterior indicator reference, split the mu integral at
log(bound) - z_0.9 exp(eta), and refine both subintervals. These finite
grids are a synthetic regression reference, not certified integration
error bounds for general models.

The committed tests exercise the actual existing sampler and scalar
diagnostic with fixed seeds 51,52,53, 4000 draws per seed per model,
checking coordinate means, direct CDF/SF, rare SF at exp(4), upper
log moments, lower design means, fixed-scale upper mean and fixed-point
predictive/upper-design indicator survival brackets. No seed filtering
or reruns are used. The reference regression checks six independently
integrated asymptotic standard errors and a deliberately wide
factor-of-two MCSE consistency guard (detects gross scale or n-factor
mistakes, not an engineering precision tolerance). The separate
runtime audit refusal and divergent-mean tests protect the existing
fail-closed policy.

Independent standalone Python/SciPy quadrature (not repository tests)
for the same smooth reference quantities gave:

| Scalar | Fixed scale | Estimated scale |
| --- | ---: | ---: |
| Posterior mean mu | 0.00321401357 | 0.00331193016 |
| Direct AEP at exp(0.35) | 0.370186139 | 0.366476998 |
| Direct AEP at exp(4) | 0.0000670145359 | 0.000137360352 |
| E[log10 q_0.9] | 0.557966602 | 0.547787734 |
| E[q_0.2] | 0.453498734 | 0.462143398 |

Grid refinement on those smooth quantities and their variance
coefficients agreed within 7e-15 absolute for that calculation.
The independent simulation of the eight smooth reference integrands
on the six fixed streams gave maximum absolute standardized errors
below 2 and observed/reference MCSE ratios between 0.89 and 1.01.
It used separately implemented density/weight calculations, not the
repository sampler; it does not substitute for executing the committed
tests. No production precision acceptance is claimed.

## Remaining gates

Review this sufficient condition and numerical reference independently.
Run the actual repository tests and endpoint quality/build/installed
checks. Extend the existing audit only after an explicit normalized
ratio envelope, independently bounded reference or separately justified
audit policy, and per-output numerical resolution are established.
Preserve current seeds/tolerances and failure/attempt handling.
A nonlinear two-coordinate expected-parameter transform, centered
log-SD interval and inverse brackets for production must be reviewed
with the existing #5 contract; never turn finite moments into an
automatic conditional pass. Coordinate the typed CLI/report bridge
with #6/#7. No historical/grouped, rating-error, other family, native,
release or publishing scope is implied.
