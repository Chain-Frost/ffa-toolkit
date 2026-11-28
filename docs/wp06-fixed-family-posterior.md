# WP06 — Initial GEV and LP3 posterior bindings

Partial implementation for [issue #8](https://github.com/Chain-Frost/ffa-toolkit/issues/8).
Base: `affc0315e96978c22930cdb0776f90fca2f9528b`. This internal binding has independent mathematical regression cases. Installed family workflows, native
comparisons and admission under the output-specific precision contract in #5 remain open.

## Mathematical contract and source

ARR Book 3 v4.2, Chapter 2 §2.4.2.1–2.4.2.2 and Table 3.2.1 (printed pp. 26–28,
PDF pp. 43–45), §2.6.3.1–2.6.3.3 (printed pp. 37–38), §2.6.3.5 (printed p. 40), §2.6.3.6 equations 3.2.52–3.2.54
(printed p. 42), and §2.6.3.7 (printed p. 43) were inspected in the official
[PDF](https://www.arr-software.org/pdfs/ARR_190514_Book3_V4.2.pdf).
See [references](references.md) and [distribution kernels](distribution-kernels.md).
The following is an original derivation; the fixed scale/shape choice is a
bounded implementation decision, not an ARR recommendation or inferred native default.

| Family | Free prior coordinate | Explicitly fixed | Reference |
| --- | --- | --- | --- |
| GEV | `gev.location` (raw discharge) | ln(scale), EVT xi > -1 | No log reference |
| LP3 | `lp3.mean_loge_flow` | ln(SD of ln(Q/r)), skew g with abs(g) < 2 | r > 0 in declared discharge unit |

GEV uses t = 1 + xi*(q-mu)/sigma, sigma = exp(eta). For t > 0,
F = exp(-t^(-1/xi)) and f = F*t^(-1/xi-1)/sigma. The xi=0 limit is
Gumbel. ARR kappa = -xi by direct comparison of the CDF; no mapping from
an unverified Flike k is inferred. Positive xi has a lower endpoint;
negative xi has an upper endpoint. The untruncated GEV may put probability
below zero. Selecting positive exact records does not condition the
distribution on positivity; the source must actually represent the annual
maxima, not a selected positive subset. Outside-support observations have
zero density. The restriction xi > -1 removes the endpoint-singular and
discontinuous-density cases from this initial location-only binding.
Nonzero subnormal xi is rejected because the shared GEV density and
probability operations can lose the defining product xi*z there.

LP3 uses Y = ln(Q/r), with mean mu, SD s = exp(eta) and skew g. For
nonzero g, a = 4/g^2, edge = mu - 2*s/g, b = s*abs(g)/2, and
Y = edge + sign(g)*b*G where G is Gamma(a,1). The sign reverses the
CDF and survival for negative skew; the raw-discharge density divides
the transformed density by Q. For g=0 use the exact Normal limit.
Abs(g)<2 gives a>1 and a nonsingular log-flow density at its finite
endpoint. Nonzero abs(g)<1.6e-5 is rejected because the existing
kernel approximates this band by Normal and its extreme tails have
not been admitted for inference. No automatic substitution of families.
See the distribution-kernel document for binary64 limitations.

For each model, the explicitly supplied univariate Gaussian prior is on
the **free** coordinate itself; no parameter Jacobian is needed. A prior
on physical LP3 mean discharge, a conditional of a three-dimensional
regional prior, or a raw-scale prior is not silently substituted.
Scale and shape are fixed assumptions, not parameters with zero uncertainty
estimated from data. The fixed-scale LP3 quantile at any specified interior
probability is proportional to exp(mu); the analogous GEV quantile is
affine in mu. That does not qualify the numerical precision of their
posterior means or probability limits.

## Observation and execution boundary

Reuse the exact Year,AMS parser, source classification pattern, shared
WP03 kernels and WP05 likelihood. Each exact positive maximum contributes
log f(q); the three censor kinds contribute log F(upper), log S(lower)
or log(F(upper)-F(lower)), evaluated by existing stable likelihood code.
Missing/excluded rows are retained without contribution. Exact CSV is
hashed from input bytes; a typed record series has caller-asserted identity.
Require explicit units, year definition, disjoint systematic years,
positive exact values and positive censor bounds. Native flags,
historical/grouped/rating/zero-threshold models and unknown priors are
not admitted. No positive truncation, automatic low-flow censoring or
POT exposure inference occurs.

The same kernel factory is used by likelihood and weighted outputs. The
existing local curvature constructor, explicit gamma and seeded Normal
importance sampler retain source, proposal and weighted particle evidence.
A zero likelihood is not an evaluation failure; numerical/representability
errors and failed local curvature are not repaired with guessed settings.
ARR p. 42 specifically identifies endpoint sensitivity in numerical
curvature. Global ESS and concentration checks do not prove model-wide
quantity-specific integration precision.

## Independent reference and status

Synthetic exact/censored examples: one exact, one left, one right and one
interval-censored year; the remaining two years are missing/excluded.
GEV observations use location center 3 and offsets 0.2, 0.8, -0.2,
[-0.3, 1.1], scale 1 and xi = +/-0.2. LP3 uses the same offsets on ln(Q/2),
SD 1 and g = +/-1. Both priors have SD 0.25. Expected AEP is evaluated
at center+1 in the GEV case and 2*exp(1) in LP3.

For LP3 g=+/-1, Gamma shape 4 permits the independent survival
exp(-x)*(1+x+x^2/2+x^3/6) and density exp(-x)*x^3/6, followed by the
scale/reference transform. The GEV oracle uses its direct CDF and PDF
rather than SciPy. Integrate likelihood times the Normal prior by
midpoint quadrature on +/-9 prior SDs (801, 1601 and 3201 points).
The independent standard-library calculation executed on Python 3.13.5
gave the following refined values; no Toolkit implementation was invoked.

| Family/shape | Mean of free coordinate | Expected AEP |
| --- | ---: | ---: |
| GEV +0.2 | 3.042621350955445 | 0.344735210136293 |
| GEV -0.2 | 3.003612598949103 | 0.285046606890051 |
| LP3 +1 | 0.099631883791495 | 0.175092737225691 |
| LP3 -1 | -0.012192784371186 | 0.147699344282007 |

The regression tests compare individual density/censor terms
for both signs and run the actual curvature/sampler for positive GEV
and negative LP3. They compare weighted estimates to independently
integrated posterior values using six reference asymptotic standard
errors, integrating p^2/g times the centered squared integrand for the
actual proposal g. The finite integration interval and tolerance are
specific to these synthetic cases, not a general precision criterion.
At fixed observed discharge, the fixed-scale nonsingular density is bounded
in the free location coordinate; the proposal variance
is checked to exceed half the Gaussian prior variance. Tests also cover
source/coordinate rejection, zero limits, nonunit scales, intervals
intersecting support and out-of-support exact data.

The GEV censor tests also cover a subnormal survival probability, deep
representable log survival for both shape signs and the Gumbel limit, and
a threshold above the finite upper endpoint. See the log-survival
calculation in [distribution kernels](distribution-kernels.md). Exact
check results and independent review are recorded on [PR #35](https://github.com/Chain-Frost/ffa-toolkit/pull/35).

## Estimated scale and shape assessment

The [estimated-scale implementation and verification](wp06-estimated-scale.md) use fixed shape and estimated `eta = ln(scale)` in an explicitly
correlated two-coordinate prior, using independent two-dimensional
quadrature for exact and censored reference cases. For shape estimation,
define a joint density and its support in the exact model coordinates
before choosing a sampler. A prior supplied in physical scale gains
the log-Jacobian `eta`. Converting ARR kappa to EVT xi reverses its
covariances with other coordinates and has unit absolute Jacobian.
Fixing a coordinate of a regional joint prior requires an explicit
conditional or other declared prior law. A first estimated-shape model
can use bounded nonsingular shape support. A continuous LP3 skew prior
crossing zero needs a verified near-normal kernel across the current
nonzero approximation band.

At a moving finite endpoint, the density is proportional locally to
`d^(a-1)` for distance `d` inside support, with `a=-1/xi` for upper-endpoint
GEV and `a=4/g^2` for LP3. With `m` exact observations tied there and
remaining likelihood factors, prior and proposal tending to positive
finite limits, the local posterior integral requires `m*(a-1)>-1`.
The local second moment of the importance weight requires
`2*m*(a-1)>-1`. Censor factors that vanish at the same endpoint change the
total exponent. Finite-difference curvature also needs boundary tests
for nonsingular shapes. These local criteria follow from the density
forms above. Posterior propriety requires a global check for the chosen
prior and data. See [Northrop and Attalides (2016)](https://www3.stat.sinica.edu.tw/statistica/oldpdf/A26n215.pdf)
for GEV prior-specific propriety results under their assumptions.

- LP3 fixed-probability quantile `q_p = r*exp(mu + exp(eta)*k_p(g))` has
  divergent positive raw posterior moments under a joint Gaussian with
  unbounded log-scale if an admitted interior skew interval has positive
  standardized quantile `k_p(g)`. On a bounded location strip and such
  an interval, the systematic likelihood decays at most exponentially
  in eta as eta grows, and the quantile grows double-exponentially.
  A bounded log-scale prior changes this result and defines a different
  model. LP3 raw predictive order `k>0` separately requires
  `1-k*exp(eta)*g/2>0` wherever the posterior has mass, plus integrability
  of the conditional moment under the mixture.
- GEV fixed-probability quantile `q_p = mu + exp(eta)*k_p(xi)` has finite
  absolute polynomial posterior moments for compact nonsingular shape
  support and Gaussian free coordinates, assuming positive posterior
  normalizing mass. The standardized density is uniformly bounded on
  such support, so exact observations cost at most an exponential
  factor in eta and censor probabilities are bounded by one. Predictive
  absolute order `k>0` also needs `xi<1/k` and an integrable mixture near
  that boundary. Log-discharge quantile summaries need posterior
  positivity and separate log-moment checks.

Issue #5 owns per-output precision and the proposal-weight moment
checks. With a Gaussian base prior covariance P, compact nonsingular
shape support and Gaussian proposal covariance G, a sufficient tail
condition for order-r weights is positive definiteness of
`r*P^(-1) - (r-1)*G^(-1)`, using the exact-density bound above.
This permits bounded AEP integrands and single-exponential log-scale
growth. It cannot establish LP3 raw quantile moments in the divergent
case. The installed issue #6 lognormal workflow provides CLI schema and report routing. A future family path
should carry named prior coordinates, declared support and explicit
availability per estimand through that shared workflow.

## Remaining #8 scope

- Free shape/skew, shape priors and transformations, singular
  or near-zero shape regimes, independent inference and moment existence.
- Gumbel and GP explicit bindings; GP annual-maxima versus event/POT,
  fixed/free location, and unresolved settings require disposition.
- Family-specific precision admission (#5), installed family adapters (#8) and
  reporting (#7), historical/grouped (#9), rating (#10), and separately
  controlled native comparison. No release or merge requested.
