# WP06 — Censored and estimated-scale precision: mathematical eligibility

Refs [#5](https://github.com/Chain-Frost/ffa-toolkit/issues/5), #3, #4, #6 and #7.
Base: merged PR #31, `affc0315e96978c22930cdb0776f90fca2f9528b`.
Runtime admission covers the restricted reflection case described below.

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
log(bound) - z_0.9 exp(eta), and refine both subintervals with 161/321 grids. These finite
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

Grid refinement on all ten smooth quantities and their variance
coefficients agreed within 8e-15 absolute in the expanded standalone
calculation. The initial 81/161 indicator refinement exceeded the
predeclared 1e-4 absolute quadrature difference at some endpoints;
refining the grid to 161/321 reduced the largest difference in the
indicator means and variance coefficients to 3.4e-5 without weakening
the criterion. All twelve integrands on the six fixed streams gave
maximum absolute standardized error below 2.91, observed/reference
MCSE ratios between 0.89 and 1.02, minimum centered side effective
size above 244, and minimum side count above 877.
It used separately implemented density/weight calculations, not the
repository sampler; it does not substitute for executing the committed
tests. No production precision acceptance is claimed.


## Normalized density-ratio envelope

A separate sufficient bound is available without estimating a normalizer
from the sampled weights. Write the target as a constant times
phi_(m*,P)(theta) W(theta), where m*=m-n P e_eta (m*=m at fixed scale).
Here W is the product of the exact Gaussian residual factors
exp(-((y-mu)/sigma)^2/2) and the censor probabilities, so 0<W<=1.
The normalized density is phi_(m*,P) W / Z_W. If G-P is strictly
positive definite, completing the square gives the log supremum of
phi_(m*,P)/g as

    B = (log(det G / det P) + (a-m*)' (G-P)^-1 (a-m*))/2.

If G=P and a=m* exactly in real arithmetic, B=0. Equal covariance
with unequal means does not have a finite Gaussian ratio supremum.
The strict covariance requirement for this envelope is stronger than
the second-moment condition; never conflate them.

For a finite box A around the tilted-prior mean, use a strictly positive
lower bound for its Gaussian probability (the marginal union bound
works even with correlation). If each marginal interval spans k SD
either side, P(A) >= 1 - 2 d Phi(-k), where d is dimension. Bound each
exact residual on A by its largest absolute log-flow residual divided
by the smallest sigma on A. For each censor contribution, choose a
finite nonempty log-flow subinterval contained in the event. Its
probability on A is at least its width times the smallest Normal
density bound on A and that subinterval. Specifically, with sigma
between s_min and s_max and maximum |y-mu| equal to D, the density is
at least exp(-D²/(2 s_min²))/(sqrt(2 pi) s_max). Multiply these lower
bounds for W. Then Z_W >= P(A) inf_A W > 0, and B - log(lower Z_W)
bounds the normalized log ratio. No sampled weight or estimated Z
is used as a substitute for the lower bound.

For the synthetic examples above with k=1 and finite censor
subintervals [-1.2,-0.2], [0.1,1.1], [-0.5,0.5], independent
binary64 calculations give illustrative lower Z_W values 0.0012639283
(fixed) and 0.0001511031 (estimated scale), giving log-ratio caps
about 6.67354 and 8.79755 for the matching tilted Gaussian.
These are mathematical derivation/regression targets, not certified
outward-rounded runtime values. A runtime implementation must account
for finite representation of the tilted location, strict matrix signs,
log determinants, quadratic form, lower box mass and likelihood bounds;
withhold admission if the declared ceiling or rounding cannot be
established. This is not a change to the approved exact-record audit.

## Implemented rounding and independent reference

The envelope forms the tilted mean, determinants and quadratic form exactly
from the stored binary64 parameters. The likelihood box uses a radius of
one marginal SD rounded upwards and a Gaussian box probability lower bound
of one third. On [0,1], the alternating exponential series gives
exp(-x?/2) >= 1 - x?/2 + x?/8 - x?/48. Integrating and using
sqrt(2 pi) < 51/20 gives marginal probability greater than two thirds.
A union bound handles either dimension and permits correlation. The censor
density bound uses sqrt(2 pi) < 3 and an inward-rounded event subinterval.
A private ceiling-rounded Decimal context and adjacent Decimal values for
correctly rounded transcendental operations bound the total log ratio.
The bound applies to the real-arithmetic model and proposal specified by
the stored parameters. The sampler retains its separate numerical
resolution guards.

Reflection admission requires zero prior mean in mu and, at estimated
scale, zero cross-covariance with eta. Each included exact observation or
censor event must match its reflection Q -> R?/Q with equal multiplicity
around reference discharge R. The check uses rational representations of
stored flows. This transforms left censoring to right censoring and
reverses finite interval endpoints. The likelihood, including its exact
flow Jacobian factors, is even in mu at every eta. Thus E[mu] = 0 and the
predictive F and SF at R both equal 1/2. The reciprocal expected AEP at R
equals 2. The conditional median at the expected parameters equals
R exp(E[mu]); its scale coefficient is zero in either model. The audit
uses the existing reciprocal and monotone discharge transformations,
scalar centering, simultaneous critical value, and scheduled streams.
It requires the explicit ratio ceiling and an exact reference for the
requested quantity before running an admitted audit.

For a reproducible example use R = 1, one left and one right censored
annual maximum at R, a zero Gaussian prior mean, mu variance 0.16,
and either fixed log-SD zero or independent eta variance 0.0144. Use
the prior as the proposal. The committed reflection test executes seeds
51, 52 and 53 with 4,000 draws each, verifies the exact scalar reference
values above and checks that unsupported quantities remain unresolved.
A conditional pass describes agreement within the specified integration
tolerance for that output and schedule. The posterior flood uncertainty
is a separate quantity.

## Remaining gates

General asymmetric fixed-scale censoring needs integration of the
deterministic reference enclosures being developed in PR #33. General
estimated-scale output references require independently bounded
integration over both coordinates. These references must precede
acceptance of two-coordinate expected-parameter transforms, nonzero
scale-dependent log moments and fixed-point predictive or posterior
inverse endpoints. Exact symmetry can locate some inverse medians, but
it supplies no numerical values at the audit window endpoints. Where
reflection gives an exact median, the audit retains its two endpoint
scalar diagnostics and withholds an inverse precision pass.
Divergent free-scale upper design means retain the undefined state.
