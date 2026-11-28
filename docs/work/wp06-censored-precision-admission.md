# WP06 — Censored lognormal precision admission: Gaussian envelopes

**Proposed scientific extension, for independent review before runtime
admission.** Refs [#5](https://github.com/Chain-Frost/ffa-toolkit/issues/5)
and [#1](https://github.com/Chain-Frost/ffa-toolkit/issues/1).
Coordinate the installed workflow #6 and reporting #7 without changing
their ownership. PR #31 is the accepted exact-record, fixed-log-SD audit;
its controls, criteria, estimands and verification are not replaced here.
No native qualification, adaptive stopping or new prior is proposed.

## 1. Mathematical basis and specific problem

The [explicit lognormal model](../wp06-lognormal-binding.md) admits
disjoint, explicitly classified systematic positive exact, left-censored,
right-censored and interval-censored records, and missing/excluded rows
that have no likelihood contribution. No grouped, historical,
zero-threshold or rating-error law is inferred. The source-coordinate
Gaussian prior is supplied; sigma = exp(eta), Y = ln(Q/r) has conditional
Normal mean mu and SD sigma. Discharge reference r is positive.

Source foundations: [ARR Book 3 v4.2](https://www.arr-software.org/pdfs/ARR_190514_Book3_V4.2.pdf)
Chapter 2 sections 2.6.3.1–2.6.3.7 (likelihood, prior, importance integration
and outputs), as admitted in the binding; [Owen, Monte Carlo theory,
methods and examples, chapter 9](https://artowen.su.domains/mc/Ch-var-is.pdf),
section 9.2, equations 9.8–9.9 (ratio-estimator variance). The accepted
[precision specification](../wp06-output-precision.md) gives the centered
SNIS derivation, fixed independent audit schedule and inverse-bracket
rules. The Gaussian factorizations, inequalities and deterministic
reference constructions below are original derivations; no external
code, data or unpublished evidence is needed.

PR #31 uses the analytic Normal posterior of exact observations with a
known scale to establish moments, a bounded importance ratio and
independent reference quantities. Censoring destroys that conjugate
posterior. Substituting the exact-only Normal distribution for the
censored posterior, or merely observing good ESS, is incorrect.
The objective is to retain the existing sampling estimator and prove
sufficient target/proposal conditions for its precision audit.

## 2. Fixed log-SD: base Normal times bounded censor likelihood

Let the supplied mean prior be N(m0,v0), v0>0. Write y_i=ln(Q_i/r)
for the n exact included observations. Complete the square using
only the exact density contributions and prior:

    v = 1/(1/v0 + n/sigma²)
    m = v*(m0/v0 + sum(y_i)/sigma²)
    b(mu) = Normal(mu; m,v).

The raw-flow Jacobians and other constants cancel on normalization.
For n=0 use m=m0 and v=v0. This base distribution is a computational
factor, not the posterior when any censor record is included.

For each left event with log upper bound u, right event with log lower
bound l, or interval (l,u], define respectively

    C_left(mu) = Phi((u-mu)/sigma)
    C_right(mu) = Phi((mu-l)/sigma)
    C_interval(mu) = Phi((u-mu)/sigma)-Phi((l-mu)/sigma).

Use the existing stable likelihood evaluator in production. Each
factor is strictly positive and at most one in exact real arithmetic
for finite positive source bounds and sigma>0. With C their product,

    A = integral b(mu) C(mu) dmu,   0 < A <= 1
    pi(mu) = b(mu) C(mu)/A.

This proves propriety; it also handles all-censored admitted models.
It does not imply that A is numerically resolved or large enough for
a useful importance proposal. Do not delete censor contributions,
approximate them by exact observations or manufacture a different prior.

### 2.1 Normalized ratio envelope

For an explicit proposal g=N(a,s²), s²>v, maximizing the quadratic
log(b/g) by differentiation gives

    log B = (log(s²/v) + (a-m)²/(s²-v))/2,
    b(mu)/g(mu) <= B.

When s²=v, the ratio is bounded only if a=m, in which case B=1.
Do not admit an equal-variance displaced proposal on this proof.
Keep the existing conservative comparison s²>=v0 for the first
extension; require a resolved strict difference s²-v or a verified
identical-base proposal. A narrower proposal is not thereby proved
invalid, but is outside this bounded-ratio admission route.

Any established positive lower bound A_L<=A gives

    pi/g <= K = B/A_L,
    log K = log B - log A_L.

Here K is the upper bound B/A_L, not an estimate of the largest
observed normalized weight. For admission compare a conservative
upper enclosure of log(B/A_L) against the caller's predeclared
maximum_log_ratio. Record both terms and their derivation. Never
substitute a sampled maximum, fitted mode or unverified normalization
constant for this global bound.

For any h with E_b[|h|^j] finite,

    E_pi[|h|^j] <= E_b[|h|^j]/A_L,
    E_g[(pi/g)^2 h^2] <= K E_pi[h^2],
    E_g[(pi/g)^k] <= K^(k-1), k>=1.

Thus bounded CDF/SF and indicator integrands, parameter mean,
exp(mu + constant) design quantiles, and affine log10 quantiles
have the required moments. The centered log10 second-moment audit
also has a finite product second moment (requires the fourth log
moment). The ratio bound is a sufficient model-specific tail
condition, not a guarantee of adequate finite sample coverage.
Keep the explicit weight, contribution-side count/ESS, nondegeneracy
and replicated fixed-budget gates from PR #31.

### 2.2 Construct a positive normalizer enclosure

On a finite closed cell J=[a_j,b_j], C_left is decreasing, C_right
increasing. A positive finite interval probability is unimodal as a
function of mu: its derivative is

    [phi((l-mu)/sigma)-phi((u-mu)/sigma)]/sigma,

positive before (l+u)/2 and negative after it. Its minimum on J
is the smaller endpoint value, its maximum the value at the
midpoint of the censor bounds clamped to J. Multiply the individual
minima/maxima (all nonnegative) to obtain C_j^- and C_j^+.
These product bounds need not be tight: the separate extrema may
occur at different mu. They must not be confused with optimized
joint likelihood values.

For a predeclared finite partition of [L,U] into cells J_j let
w_j=integral_Jj b, and T=P_b(mu outside [L,U]). Then

    A_L = sum(w_j C_j^-)
    A_U = min(1, sum(w_j C_j^+) + T).

In exact arithmetic any finite cell with strictly positive mass and
censor factors gives A_L>0. Use disjoint cells up to endpoints of
zero base probability. For reference integration, omitted tails
are not zero; they enter the upper bound. Refinement on an explicitly
budgeted deterministic schedule can narrow the cells. No random
draws are used in these bounds, and no resource escalation after
observing the stochastic audit is allowed.

### 2.3 Numerical admission must not confuse a formula with an enclosure

The inequalities above are in exact real arithmetic. Ordinary
binary64 CDF subtraction, multiplication of small probabilities,
rounding log ratios, or nextafter around an unbounded special-
function approximation do not automatically produce rigorous bounds.
A stable CDF/SF/log-CDF implementation is not by itself a certified
interval evaluator. The runtime extension must establish conservative
evaluation bounds or withhold the affected admission/reference.
The numerical normalizer lower bound must never be rounded upward
and then treated as a mathematical lower bound.

One constructive route, independent of the sign or saturation of a
special-function result, is to use finite density rectangles. For
Y~N(mu,sigma²), select a finite nonzero log-discharge subinterval
[a,b] contained in a censor event. For mu in J=[l_j,u_j], let
D=max{|y-mu|: y in [a,b], mu in J} (the four corners suffice).
Then

    P(a<=Y<=b | mu) >= (b-a)/(sqrt(2*pi)*sigma)
                            * exp(-D²/(2*sigma²)).

For a left event use a finite subinterval ending at its upper bound;
for a right event one starting at its lower bound; for an interval
event use its own finite bounds. The choice and positive represented
width must be specified before sampling. The same rectangle bound
applies to the base Normal mass of a finite cell. Summed subdivisions
can improve the conservative density bounds. Use an explicitly
justified outward evaluation of elementary operations, constants,
exp/log and tail bounds, or another independently verified interval
method; preserve evidence and resource limits. Failure to resolve
a positive bound, overflow/underflow, a ratio ceiling exceedance,
or insufficient enclosure resolution is indeterminate. Do not
relabel an approximate exploratory enclosure as interval-certified.

## 3. Independently bounded censored reference outputs

Define N_h=integral b C h. For each finite cell, bound h by h_j^-,
h_j^+ and take the minimum/maximum of the four products C_j^-/+
times h_j^-/+. Multiply by w_j. Sum with valid lower/upper bounds
on omitted-tail contributions. Divide the numerator interval by
[A_L,A_U], using all four quotient corners for signed h. For
nonnegative h use [N_L/A_U,N_U/A_L]. Intersect with proven support
only. This general approach avoids false precision from replacing
an independently computed reference interval with its midpoint.

In particular:

- Scalar parameter means: h=mu. For a finite interval containing
  zero, the two omitted-tail contributions are bounded by the
  signed truncated-Normal first moments (and zero); C<=1.
  More generally use an absolute-tail moment bound.
- Flood-quantile means: h=r exp(sigma*z_p+mu), monotone increasing.
  Use the exact exponential-tilt identity
  exp(mu)b(mu)=exp(m+v/2) Normal(mu;m+v,v) to bound the omitted
  positive expectation. Multiply by the fixed physical scale
  r exp(sigma*z_p), or calculate in log space.
- Expected F and direct AEP at finite positive flow: h is bounded
  in [0,1], decreasing/increasing in mu respectively. Outside
  [L,U] the numerator contribution is between zero and T. Use
  direct SF, not 1 minus a rounded CDF.
- Log10 design-quantile mean and SD: h=(ln(r)+mu+sigma*z_p)/ln(10).
  Affine and squared-centered affine integrands have cell extrema
  at endpoints and, for the square, at the clamped zero. Evaluate
  the truncated Normal first/second moments or a conservative
  absolute-moment tail envelope. Do not require the exponentiated
  flow to be representable. Retain the primary centering constant
  for the two audit moments.
- Quantile at expected parameters: monotone transform of a reference
  mean-coordinate enclosure, distinct from mean design quantile.
- Reciprocal expected AEP: invert only a strictly positive reference
  probability interval and a strictly positive audit interval.
- Inverse outputs: evaluate independent reference posterior CDF/SF
  (threshold indicator) or predictive CDF/SF (conditional probability)
  at the fixed primary-window endpoints. Posterior design quantiles
  are strictly increasing in mu; per-draw return periods are strictly
  decreasing in mu. Split an integration cell at a represented
  threshold or use its exact indicator range [0,1] when crossing.
  For b>1/2 use direct complementary probabilities, including the
  correct strict greater-than indicator. Require the full reference
  enclosures to establish the relevant strict bracket inequalities;
  do not introduce an estimated quantile-density denominator.

Truncated Normal moment identities for Z~N(m,v), s=sqrt(v),
z=(c-m)/s, and standard phi/Phi/S:

    E[Z 1_(Z<c)]   = m Phi(z) - s phi(z)
    E[Z 1_(Z>c)]   = m S(z) + s phi(z)
    E[Z² 1_(Z<c)]  = (m²+v)Phi(z) - (2ms+v z)phi(z)
    E[Z² 1_(Z>c)]  = (m²+v)S(z) + (2ms+v z)phi(z).

These follow from integrating the standard Normal density derivative.
Apply them after an appropriate coordinate shift for centered moments.
A practical conservative alternative uses
(Z-c0)² <= 2(Z-m)²+2(m-c0)². Numerical evaluation must protect signs
and cancellation; an unresolved negative upper bound on a nonnegative
quantity is an error, not evidence of a zero tail.

For the primary estimate, a scalar reference enclosure must lie
wholly inside the predeclared tolerance window, not merely have its
midpoint inside. For inverse estimates the independently bounded
CDF/SF at window endpoints must establish the strict crossing.
Each independent audit retains the PR #31 conditional asymptotic
one-sided/two-sided confidence requirements and replicate stability.
For a scalar reference enclosure require its containment in the
corresponding audit interval, and that the audit interval (transformed
if needed) lies in the primary tolerance window. For one-sided inverse
checks require the entire relevant probability reference enclosure
on the safe side of the computed audit bound; also require the
strict crossing relative to the requested probability level.
If reference width prevents a decision, mark indeterminate; a valid
reference proving an estimate or audit interval outside tolerance
is criteria_not_met. Do not select a favourable midpoint or widen
the declared window. These additions are conservative numerical
reference checks, not new stochastic coverage guarantees.

Introduce typed censored admission/reference results rather than
calling NormalAdmission's conjugate formulas with base m,v. Preserve
the exact-record branch and tests unchanged. Record model/proposal
identity, partition/tail/evaluator provenance, explicit maximum
reference evaluations/cells, resolved ratio envelope and per-output
reference enclosures. Count the reference work separately from
the existing sampler/output budget. A deterministic reference may
be evaluated at primary-selected points under the predeclared budget;
it never becomes the primary sampling estimator. CLI and reporting
consume the reviewed typed evidence rather than reconstruct a pass
from a numeric value or ESS. A complete implementation still needs
independent scientific and exact-functional-head review.

### 3.1 Closed-form normalizer and exact all-censored checks

One censor event has an additional independent identity: under the
base b, Y has marginal distribution N(m,v+sigma²). Therefore A is
exactly its probability of that event. This applies with any number
of included exact records. With one event the numerator for E[exp(t mu)]
is exp(t m + t²v/2) times the same event probability under a base
mean shifted to m+t v. Divide by A. The identity follows by completing
the Gaussian square and does not require quadrature in mu.

An especially useful exact all-censored control uses no exact rows,
prior N(0,1), sigma=1, r=1, and distinct systematic years censored at
Y=0. With L left and R right events the variable V=Phi(-mu) is
uniform on (0,1) under the base. The posterior density of V is
proportional to V^L(1-V)^R. Hence

    A = Beta(L+1,R+1) = L! R!/(L+R+1)!
    V | data ~ Beta(L+1,R+1)
    E[F(1|mu)] = (L+1)/(L+R+2)
    E[S(1|mu)] = (R+1)/(L+R+2).

Since mu=-Phi^(-1)(V), its posterior b-quantile is the
negative standard Normal inverse of the (1-b)-quantile of V.
All probabilities and exponentiated output mappings must still
satisfy their declared numerical representation checks.

For one left event, A=1/2 exactly. The joint Gaussian
representation Y=mu+independent N(0,1) gives
E[mu|Y<=0]=-1/sqrt(pi) and Var(mu|Y<=0)=1-1/pi by the
one-sided truncated Normal moments and Gaussian conditional
variance. The two equalities verify centered log-moment admission
without using Toolkit's likelihood or sampler as an oracle.

Independent numerical integration of the original density on the
real line verified the following all-censored identities:

| L,R | Exact A | Exact expected F at flow 1 | Posterior mu 90th percentile (Beta inverse) |
| --- | ---: | ---: | ---: |
| 1,0 | 1/2 | 2/3 | 0.478273532376 |
| 3,2 | 1/60 | 4/7 | 0.431109472925 |
| 10,0 | 1/11 | 11/12 | -0.882071138844 |

These are useful exact/analytic controls for the first functional
extension, including the no-exact-record branch and decreasing
posterior quantile as left censor evidence accumulates. They are
not assumptions about arbitrary unequal censor thresholds.
The one-event and Beta identities can be used as independent
verification oracles; do not introduce a special production
posterior in place of the actual supplied censor likelihood.

## 4. Estimated log-SD: further analytical envelope (separate runtime gate)

The admitted untruncated Gaussian prior on theta=(mu,eta) has mean
m0 and full SPD covariance V0. For n exact observations, factor the
raw-flow likelihood as a theta-independent constant times

    exp(-n eta) * exp[-exp(-2 eta) sum((y_i-mu)^2)/2].

The remaining censor likelihood is bounded between zero and one.
With e=(0,1)^T, completing the square gives the exponentially tilted
prior

    phi(theta;m0,V0) exp(-n eta)
      = exp(-n e^T m0 + n² e^T V0 e/2) * phi(theta;m*,V0),
    m* = m0 - n V0 e.

Thus the exact posterior is again b R/A, now with two-dimensional
Gaussian base b=N(m*,V0), and

    R(theta) = exp[-exp(-2 eta) sum((y_i-mu)^2)/2] * C(theta),
    0 < R <= 1,   A=E_b[R].

This proves propriety independently of a local Hessian and clarifies
that the computational envelope is not a modification of the prior.
An admitted model with no exact records has m*=m0.

For explicit Gaussian proposal N(a,G), require G-V0 SPD for the
following sufficient bound (or the identical-base case, for which
B=1). Completing the quadratic as in section 2 gives

    log B = (log(det(G)/det(V0))
             + (a-m*)^T (G-V0)^(-1) (a-m*))/2.

Use stable full-matrix factorization/solve and proven SPD difference,
not diagonal variance checks or a guessed condition-number threshold.
The same normalized bound B/A_L and moment inequalities then apply.

A positive lower bound A_L follows on any finite rectangle
mu in [l,u], eta in [c,d]. The residual exact-density factor is at least

    exp[-exp(-2 c) sum(max((y_i-l)^2,(y_i-u)^2))/2].

For a censor event choose a finite subinterval [a,b] of positive width.
Let D be the largest absolute difference between a or b and l or u.
A conservative lower bound on its probability is

    (b-a)/(sqrt(2*pi)*exp(d)) * exp[-D²/(2 exp(2c))].

Multiply the bounds with the positive Gaussian mass of the rectangle.
A cell mass lower bound can also be obtained from the minimum Gaussian
density on that rectangle times its area; the maximum quadratic form
is attained at a vertex by convexity. Rectangular subdivision,
multivariate tail enclosures and numerical representation require
separate budgeted verification. For a diagonal synthetic base the
cell masses are products of one-dimensional Gaussian masses; that
shortcut must not be applied to correlated coordinates.

### Output-specific moment implications

The bounded-ratio argument establishes finite ratio moments if the
envelope is resolved. It does not establish all output expectations.

- For p<=1/2, q_p=r exp(mu+z_p exp(eta)) <= r exp(mu), so all
  positive powers have finite expectation under the Gaussian base
  and hence posterior. For p>1/2 the already reviewed divergence
  proof applies to the unrestricted Gaussian prior and these data.
  Its posterior mean design flood is undefined; do not attempt to
  certify its finite sampled surrogate.
- The log design quantile is affine in mu and exp(eta); all finite
  polynomial moments exist under the tilted joint Gaussian, including
  the fourth moment needed by the centered second-moment audit.
- Conditional CDF/SF and indicators remain bounded. Predictive and
  posterior quantile inverses may be defined despite the divergent
  mean; their finite representation and local numerical/tail resolution
  still require explicit checks. Reciprocal expected AEP requires a
  positive resolved probability interval.
- Parameter moments exist. For a quantile at mean parameters, a
  simultaneous rectangle for the two coordinate means could be
  propagated using monotonicity in mu and the sign of z_p for eta.
  This is a separate two-coordinate transform and must not reuse the
  one-coordinate fixed-scale interval or spend uncounted error budget.

This is a mathematical route to broader model admission, not admission
of a current free-scale run. The two-dimensional independent reference
and correlated rectangle/tail calculations, numerical envelope
evaluation, parameter-rectangle transform and output-specific
representability tests must be reviewed and implemented separately.
In particular, do not declare estimated-scale precision valid just
because the theoretical bound exists; it may be computationally too
loose under the explicit policy.

## 5. Original numerical evidence (author-side, not interval certification)

Executed independently of Toolkit's sampler and runtime precision code
using Python 3.13.5, NumPy 2.3.5 and SciPy 1.17.0 on Linux.
All examples are original synthetic log-discharge observations,
reference r=1. These are exploratory binary64 comparisons of the
exact-real bounds with independent numerical integration, not a
claim of certified interval arithmetic or production audit acceptance.
No random seed selection or existing test tolerance was changed.

### Fixed-scale mixed exact and censor events

Use exact y=(-0.4,0.5), left event Y<=-0.2, right event Y>0.1,
interval event -0.5<Y<=0.6, prior N(0,1), known sigma=0.8,
and explicit illustrative proposal N(0,1). The base Normal has
m=0.0378787878788, v=0.242424242424. Divide m +/- 8 sqrt(v)
into 4096 equal cells. Use section 2 extrema and Gaussian cell
probabilities; include the omitted normal mass in upper bounds.
Independently integrate the density and numerator by adaptive
quadrature on the real line (absolute and relative requested
integration tolerances 1e-13). For quantile means use the
exponential-tilt identity, avoiding overflow in the integrand.
Invert the numerical posterior/predictive CDF on [-3,3] in log
space for the displayed reference quantiles.

| Quantity | Independent quadrature | Exploratory finite-cell enclosure |
| --- | ---: | ---: |
| A | 0.0676240123107145 | [0.0674456465907688, 0.0678026053057583] |
| Posterior mean mu | 0.0158223513737 | [0.0139807751405, 0.0176737116019] |
| E[mu²] | 0.1329595261158 | [0.1316421299113, 0.1342895617661] |
| Mean q at F=0.9 | 3.026592795635 | [3.007689324523, 3.045615347351] |
| Expected AEP at exp(0.35) | 0.3519021069439 | [0.3496425142923, 0.3541759713951] |

The E[mu²] quadrature value is the sum of the independently integrated
variance 0.1327091793128 and squared mean. The exploratory ratio
ceiling log B - log A_L is 3.405913220763. Using the independently
integrated A in the envelope formula gives 3.403272126899 (still
an envelope, not an observed or necessarily attained ratio maximum).

Other independent reference quantities for F=0.9 are predictive
quantile 3.134133021193, posterior median design quantile 2.831929920446,
quantile at expected parameters 2.832227781193 and log10 posterior
SD 0.158210314200. The distinction between these and the mean
design quantile remains material.

### Free-scale tilted Gaussian

Use exact y=(-0.4,0.5), left Y<=-0.2, right Y>0.1, independent
mean-zero Gaussian prior coordinates with SDs 0.4 and 0.12.
This is the existing synthetic binding example, with n=2.
The tilted base has means (0,-0.0288) and unchanged variances.
Independent tensor standard-normal Gauss-Hermite integrations with
64 and 128 nodes per dimension give A=0.123168872497708 and
agree to displayed precision for posterior coordinate means
0.00359830488687 and -0.0133856466775, expected AEP at exp(0.35)
0.3686203803796, and mean log10 design quantile at F=0.9
0.5546099794288. The latter is not the logarithm of a finite
posterior mean design flood.

Divide each tilted-base standardized coordinate interval [-8,8]
into 256 cells. For exact residual use the extrema of the convex
sum-of-squares at the two mu endpoints and the clamped exact
sample mean, and exp(-2 eta) extrema. For the one-sided censor
terms use the min/max standardized argument over rectangle corners.
Multiply by cell masses (valid here because the synthetic base
coordinates are independent); add the marginal union bound for
the omitted mass to the upper normalizer. The exploratory result is

    A_L = 0.118926957878729
    A_U = 0.127455600017058.

For an illustrative explicit proposal identical to this tilted base,
log B=0, and -log(A_L)=2.129245773661. This is evidence of the
usefulness of the factorization in this case, not a proposal
selection rule or a universal acceptance ceiling.

### Reproduction of key normalizer checks

The following original snippet reproduces the independent fixed-scale
integral and finite-cell bounds, and the free-scale quadrature
normalizer. It is research verification only, not runtime bound
arithmetic or a replacement for the independent sampler.

```python
import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr, roots_hermitenorm

sigma = 0.8
v = 1 / (1 + 2 / sigma**2)
m = v * (-0.4 + 0.5) / sigma**2
s = np.sqrt(v)


def interval(mu):
    return ndtr((0.6 - mu) / sigma) - ndtr((-0.5 - mu) / sigma)


def censor(mu):
    return ndtr((-0.2 - mu) / sigma) * ndtr((mu - 0.1) / sigma) * interval(mu)


def base(mu):
    return np.exp(-0.5 * ((mu - m) / s) ** 2) / (np.sqrt(2 * np.pi) * s)


A = quad(lambda mu: base(mu) * censor(mu), -np.inf, np.inf, epsabs=1e-13, epsrel=1e-13)[0]
edges = np.linspace(m - 8 * s, m + 8 * s, 4097)
a, b = edges[:-1], edges[1:]
mass = np.diff(ndtr((edges - m) / s))
lo = ndtr((-0.2 - b) / sigma) * ndtr((a - 0.1) / sigma) * np.minimum(interval(a), interval(b))
hi = ndtr((-0.2 - a) / sigma) * ndtr((b - 0.1) / sigma) * interval(np.clip(0.05, a, b))
print(np.dot(mass, lo), A, np.dot(mass, hi) + 2 * ndtr(-8))

for n in (64, 128):
    x, w = roots_hermitenorm(n)
    mu = 0.4 * x[:, None]
    eta = -0.0288 + 0.12 * x[None, :]
    sd = np.exp(eta)
    residual = np.exp(-0.5 * ((mu + 0.4) ** 2 + (mu - 0.5) ** 2) / sd**2)
    residual *= ndtr((-0.2 - mu) / sd) * ndtr((mu - 0.1) / sd)
    print(n, np.sum(w[:, None] * w[None, :] * residual) / (2 * np.pi))
```

## 6. Bounded implementation and review sequence

Review sections 2–3 and the numerical enclosure policy independently.
First implement the fixed-log-SD censored reference/admission component
and its meaningful analytic/numerical verification. Include exact-only
and all-censored limiting cases (including the exact Beta and one-event
controls in section 3.1); left/right/interval/mixed records;
finite-cell lower/upper containment and monotonic refinement for a
fixed extent; nonconjugate reference outputs and transformed moments;
positive very small normalizer and exhausted explicit budgets;
unrepresentable or extremely narrow censor intervals; equal-base versus
displaced equal-covariance and unsupported narrower proposals; direct
rare SF and insufficient indicator/contribution resolution. Compare
against independent integration, using the synthetic values above and
additional preregistered cases. Verify common source-coordinate
rescaling, unchanged exact-only behavior, retained failed attempts
and no reference midpoint shortcut. No new native fixture or external
campaign is required.

Do not route the censored model through the exact-only NormalAdmission
reference formulas. Keep #31 behavior and limits intact until the new
admission is implemented and independently accepted. After review of
the fixed-scale extension, implement the separate full-Gaussian
estimated-log-SD envelope and independently justified reference
integration/transform routes. Section 4 is substantive preliminary
mathematics, not an implemented production admission.

This checkpoint introduces no runtime, shared schema, registry status,
CLI, dependency, fixture, CI or publishing change. Issue #5 remains
open. The proposal, numerical checks and eventual implementation
need their own appropriate independent review; native Flike
qualification remains external.


## 7. Fixed-scale numerical reference implementation checkpoint

The mathematical specification at commit `6653fd488d896c0bfbd9ed4908c4c0101c857ec1`
received an independent review with no findings. The following new
functional contribution requires its own exact-head review.

### Implemented component

The internal `censored_reference.fixed_scale_reference` computes
deterministic reference enclosures for the exact-real fixed-scale
model (stored binary64 settings interpreted exactly). The adapter
`importance_censor_reference.bound_fixed_scale_reference` extracts
the included records and the actual prior/proposal settings from
`ClassifiedLognormalPosterior` and preserves their complete identity.
It does not evaluate a different prior or treat missing/excluded
rows as observations. The existing stochastic precision audit and
its accepted exact-only tests remain unchanged.

Arithmetic uses independent 64-digit Decimal contexts with directed
floor/ceiling rounding for arithmetic operations. The monotonic
elementary exp, ln and sqrt operations enclose their correctly
rounded results by adjacent decimal numbers; inputs are exact
binary64-to-decimal conversions, not short decimal renderings.
This relies on the documented decimal arithmetic contract
([Python decimal documentation](https://docs.python.org/3/library/decimal.html)).
No unchecked float transcendental, CDF evaluation, tolerance-based
quadrature error estimate or midpoint substitution enters the
enclosure. The pi constant is bounded and independently checked
by the alternating rational Machin identity.

For Normal survival at a nonnegative argument x<=8, integrate the
Taylor expansion of exp(-t²/2) from 0 to x. With an even number
of terms, the partial sum is a lower bound and adding the next
positive term gives an upper bound, by the signed Taylor remainder.
Every term and partial sum is enclosed; at most 192 terms are used.
The fixed 1e-55 term cutoff only selects between two mathematically
valid Taylor bounds, not an output acceptance tolerance. For
8<x<=64, integration by parts supplies

    phi(x)*x/(1+x²) <= S(x) <= phi(x)/x.

For x>64, use zero lower bound and the upper bound at 64 by
monotonicity. Reflection handles negative arguments. A resulting
wide interval is not a precision pass. Direct survival evaluation
avoids subtracting rounded CDFs from one in the upper tail.

Partition the standardized base coordinate z=(mu-m)/sqrt(v) into
exact dyadic cells on [-extent,extent]. On each cell the product of
its width and the enclosed standard Normal density range encloses
the base mass. Interval arithmetic maps the cell to mu, evaluates
censor-factor ranges, and encloses the product with each integrand.
For a two-sided event the CDF/SF difference is evaluated on the
whole mu cell (a possibly wider alternative to the separate extrema
of section 2.2). The lower bound is intersected with zero and the
upper with one by proven probability support. Cells crossing an
indicator threshold use [0,1], not a midpoint indicator.

The omitted mass is bounded by 2 phi(extent)/extent. Mean and
centered-second-moment tails use the absolute truncated moment
bounds in section 3; exp(mu) uses the shifted Normal identity.
All signed normalization divisions use the interval quotient.
Supported reference requests are E[mu], E[(mu-c)^2], E[exp(mu)],
expected F/direct AEP at positive flow, and posterior mean-coordinate
CDF/direct strict SF at a specified threshold. Return full Decimal
intervals, not an estimated reference point; the actual flood output
transforms, certified inverse-Normal thresholds, and their connection
to the stochastic audit remain separate implementation work.

`ReferenceControls` requires an explicit standardized extent,
dyadic level, cell budget, cell-term budget, and normalized log-ratio
ceiling. The current extent range 1..16 and dyadic levels 0..10
are implementation resource limits, not engineering tolerances.
Preflight reserves cell terms for all requests and censor events;
a cell term needs at most four point survival evaluations, each with
at most 192 series terms. No adaptive grid refinement or sampling is
performed by this component. Ratio eligibility is withheld for
unresolved normalizer, displaced equal base variance, narrower
proposal, unresolved positive variance difference, numerical failure,
or ceiling exceedance. It is only mathematical ratio-envelope
eligibility, not a conditional precision pass or qualification of
the binary64 likelihood/sampler implementation.

### Executed checks and current limitations

Seven focused tests of the standalone numerical core passed on
Linux CPython 3.13.5 (7.03 seconds). The local source/test bytes
were verified against the Git blobs added to the PR. They cover
mixed exact/left/right/interval reference containment and dyadic
refinement; independent exact single-event and all-censored Beta
identities; signed, centered, exponential and probability
integrands; direct rare survival; narrow unresolved interval;
identical versus displaced equal covariance; smaller proposal
variance; ratio ceiling; unit rescaling; ambient Decimal context;
predeclared resource budgets; and exact rational pi verification.

For the mixed case in section 5, extent eight and dyadic level five
(512 cells) the implementation encloses A in
[0.0649765706311381, 0.0703427704429115], with normalized log-ratio
upper bound 3.44320850548382 (rounded here for display).
Level seven (2048 cells) gives A in
[0.0669554962743191, 0.0682969857185926] and log-ratio upper
3.41320709459474. Both intervals contain the independent quadrature
value 0.0676240123107145. These deliberately loose bounds use density
rectangles and dependency-preserving interval arithmetic, not
convergence extrapolation. They are not necessarily narrow enough
for a particular stochastic audit's specified tolerance.

Repository endpoint validation and independent functional review
are still required, including the two additional typed-binding
tests. The full checkout is not available in the local Linux
environment used for the standalone tests. The concurrent PR #34
owns broader scalar moment eligibility and sampler regression tests;
coordinate registry and method-status updates with that PR rather
than duplicate them or imply a new precision-pass route. Keep #5
open. This contribution introduces no installed CLI/schema,
sampling/proposal change, dependency, native claim or publication.
