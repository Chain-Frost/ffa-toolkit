# WP06 — Mode and curvature construction of a Gaussian importance proposal

**Method ID:** `importance-sampling-reference`. **Scope:** independent local curvature
proposal construction, explicit covariance scaling, generated importance sampling and a bounded
classified lognormal source/prior binding. Broader model/observation binding, output-specific
precision and installed Bayesian fitting remain separate development requirements. These
components do not establish native Flike compatibility.

## Mathematical basis and coordinates

ARR Book 3 v4.2 (2019), Chapter 2, §2.6.3.6, printed pp. 41–42 (PDF pp. 58–59
counting the cover as page 1), gives the modal search in Eq. 3.2.52, the
multivariate Normal approximation in Eq. 3.2.53 and its inverse negative
log-posterior curvature in Eq. 3.2.54. The guidance recommends SCE as a search
option and adaptive differences for the Hessian, with particular care near
GEV/LP3 support limits. The implementation below is a narrower *local* Newton
method, not SCE or a claim about the vendor algorithm. Section 2.6.3.7,
printed p. 43 (PDF p. 60), describes weighted probability limits and expected
probability (Eq. 3.2.56). Preserve the distinction between parameter and
quantile means (the printed Eq. 3.2.55 mixes their notation).

Primary reference: [ARR3](https://www.arr-software.org/pdfs/ARR_190514_Book3_V4.2.pdf),
© Commonwealth of Australia (Geoscience Australia), 2019, CC BY 4.0 unless
otherwise marked. Kuczera (1999), Water Resources Research 35(5), pp. 1551–1557,
[DOI 10.1029/1999WR900012](https://doi.org/10.1029/1999WR900012), establishes
the FFA application. Equations (16)–(17), p. 1554, define the inverse negative
log-posterior Hessian and a multiplier on proposal standard deviation.
These equations support the explicit scaling implemented here, not a native
default or a general convergence guarantee. See `ffa references`.

Let theta be the **ordered, named source coordinates** declared by the caller.
The likelihood and prior callbacks define log t(theta) = log L(D|theta) +
log prior(theta), up to a finite theta-independent constant. The prior density
and any necessary Jacobians are defined with respect to those same coordinates.
MAP generally changes under nonlinear reparameterization. A mere change of
numerical step size does not change the density or create a new prior.

At a smooth interior local maximum m, the gradient vanishes and a second-order
expansion gives

    log t(m + d) ~= log t(m) - d^T H d / 2,
    H = - Hessian_theta log t(m),
    unscaled proposal = N(m, H^-1),
    scaled proposal = N(m, gamma^2 H^-1).

Equation (17) of Kuczera (1999) scales standard deviations by gamma, not
variances by gamma. The unscaled approximation is the exact posterior only
for a quadratic log target with full real support. Widening or narrowing
the proposal does not modify the target, prior or Jacobian. All correlations
are retained (including signed off-diagonal covariance entries). The paper
discusses widening to improve coverage and gives a typical range 1.5–2.5;
neither this range nor its suggested value establishes a mandatory domain,
native default or guarantee of coverage. Choose a positive finite gamma
explicitly for new configured workflows; the historical unscaled builder
uses gamma=1 when omitted.

## Explicit inputs and numerical method

Internal callable `build_curvature_proposal` in `ffa_methods.importance_proposal`
consumes the accepted `ExplicitPosteriorTarget` (separate prior and likelihood,
coordinate order, source provenance) and returns the accepted
`ExplicitNormalProposal` inside a diagnostic result. It does not generate
particles or add a CLI fit mode. Supply a single finite named starting point
in exactly the declared order, and all `CurvatureControls` fields.

The admitted case has one to eight coordinates and a deterministic, smooth
finite log target with positive-definite curvature throughout the visited
local Newton basin and finite central probes. In internal scaled increments
z, theta = point + S z, S = diag(scales), positive scales in [1e-12, 1e12].
This is an affine numerical preconditioner, not a nonlinear change of
posterior coordinates. The local objective is f = -log t.

- `initial_step`: scaled central difference step in [1e-6, 0.1].
- `max_refinements`: 1–8 successive step halvings.
- `max_iterations`: 1–100 Newton updates.
- `gradient_tolerance`, `mode_tolerance`, `curvature_tolerance`: positive,
  at most 0.01. The first two are absolute scaled-coordinate stationarity
  and displacement limits; the third is relative matrix-norm agreement.
- `max_condition`: upper limit on scaled Hessian infinity-norm condition
  number, in [1, 1e8]. Choose scales and tolerances for the numerical problem;
  no universal engineering or native defaults are claimed.

At each visited point, evaluate the objective and central differences in z.
For step h, the gradient component is (f(+h e_i)-f(-h e_i))/(2h).
The diagonal curvature is (f(+h e_i)-2f(0)+f(-h e_i))/h^2.
For i != j, use all four corners (+,+), (+,-), (-,+), (-,-),
with signed sum (+,-,-,+)/(4h^2). Symmetric entries share the same value.
Repeat at h/2, accepting the finer estimate only when adjacent estimates
agree within the declared checks. Use max absolute gradient change plus
16 times the largest binary64 spacing of evaluated log terms/objective
divided by h, and Hessian maximum absolute row-sum change plus
32 n times that spacing divided by h^2. These are conservative numerical
resolution *indicators*, not rigorous bounds on arbitrary callback error or
Taylor remainders. Deterministic accurately evaluated smooth callbacks are
required. For derivative probes, reject rounded perturbations differing from
the requested step by more than 1e-6 relative. Reject if agreement cannot be obtained before
the refinement limit.

Factor the fine Hessian by Cholesky and solve for its inverse; this is
actual finite-difference curvature, not a quasi-Newton optimizer's inverse
approximation. Require finite condition number below the configured limit,
inverse times curvature-error indicator less than 0.05 in infinity norm,
and identity-product inverse residual at most 1e-7. Newton steps use
the full inverse curvature; backtrack with up to 24 fractions 2^-k and
Armijo sufficient decrease (coefficient 1e-4), using the actual representable
scaled trial displacement for the directional slope. Trial components may
round to zero if their absolute error is at most mode_tolerance/16; otherwise
the same relative resolution limit applies. Require actual descent and strict
objective reduction. A nonfinite evaluation at
any visited trial or probe terminates the builder; a boundary or
nonconcave basin is not a supported search problem. A finite descent
failure or iteration exhaustion is also an error.

Success requires both scaled maximum gradient plus its resolution
indicator <= gradient_tolerance, and scaled maximum Newton displacement
plus inverse norm times the gradient indicator <= mode_tolerance.
Report the local numerical mode, not a global optimum or unique mode.
Convert the verified scaled Hessian K and inverse back using
H = S^-1 K S^-1 and covariance = S K^-1 S. Finite source covariance
must also pass the sampler's full SPD and marginal resolution checks.
After the unscaled proposal is admitted, the keyword-only `gamma` setting
multiplies every entry of its covariance by gamma squared. The raw Hessian,
mode, convergence diagnostics and numerical preconditioners are unchanged.
The implementation combines binary mantissas/exponents instead of first
forming gamma squared, which can overflow or underflow even when the final
matrix is representable. Reject nonfinite or vanished nonzero entries and
relative mantissa rounding errors exceeding 1e-12 (a binary64 representation
check, not a paper constraint). Re-admit the resulting matrix through the
existing Normal proposal constructor, including its SPD and marginal
coordinate-resolution checks. Draw generation and log-density evaluation use
the same factorization. Gamma=1 preserves the unscaled numerical result,
including same-seed draws; the derivation text now records the applied gamma.
There is no jitter, eigenvalue clipping, hidden rescaling of the posterior,
correlation removal or fallback proposal.

The result retains actual start, settings, mode, source-coordinate Hessian,
proposal covariance, applied `gamma` and derivation (also propagated to
sampler particle provenance), log target at mode, scaled gradient norm,
gradient and curvature resolution indicators, condition number, last
difference step, iteration/evaluation counts and termination description.
The caller owns the source data, model and prior provenance. Callback or
arithmetic failure aborts construction; do not use partial results.

## Original reproducible worked example

Consider independent Gaussian observations of a+b=0 (variance 1) and
a=3.5 (variance 1/2); independent unit Gaussian priors on a and b.
Dropping constants, log likelihood = -((a+b)^2 + 2(a-3.5)^2)/2 and
log prior = -(a^2+b^2)/2. Expanding gives precision [[4,1],[1,2]]
and linear coefficient (7,0). Solving the linear equations gives (2,-1);
the inverse by the 2x2 determinant formula is [[2,-1],[-1,4]]/7.
Thus the exact posterior correlation is negative; it must not be discarded.

```python
from ffa_methods.importance_proposal import CurvatureControls, build_curvature_proposal
from ffa_methods.importance_sampling import ExplicitPosteriorTarget


def likelihood(theta: tuple[float, ...]) -> float:
    a, b = theta
    return -0.5 * ((a + b) ** 2 + 2 * (a - 3.5) ** 2)


def prior(theta: tuple[float, ...]) -> float:
    a, b = theta
    return -0.5 * (a * a + b * b)


target = ExplicitPosteriorTarget(
    coordinates=("a", "b"),
    log_likelihood=likelihood,
    log_prior=prior,
    source="synthetic Gaussian observations and priors",
)
controls = CurvatureControls(
    scales=(1.0, 1.0),
    initial_step=0.01,
    max_refinements=8,
    max_iterations=30,
    gradient_tolerance=1e-5,
    mode_tolerance=1e-5,
    curvature_tolerance=1e-4,
    max_condition=1e5,
)
result = build_curvature_proposal(
    target=target,
    starting_point=(("a", 0.0), ("b", 0.0)),
    controls=controls,
    gamma=2.0,
)
print(result.proposal.means, result.negative_log_hessian, result.proposal.covariance)
```

Here gamma=2 is an illustrative explicit setting, so the proposal covariance
is [[8,-4],[-4,16]]/7 while the posterior covariance remains [[2,-1],[-1,4]]/7.
For a point (a,b), set da=a-2 and db=b+1. The independent proposal log density
is -log(2*pi) + log(7)/2 - 2*log(gamma)
- (4*da^2 + 2*da*db + 2*db^2)/(2*gamma^2).

The focused tests use independent exact precision/covariance formulas, a
reordered affine coordinate case (including the density Jacobian), a
nonquadratic two-mode target, failures and actual generated draws.
The sampler test passes both gamma=1 and gamma=2 constructed proposals to
the existing `independent_importance_sample` with seed 47 and 16,000 draws
each, checks independent proposal log densities, and uses its existing
weighted moments and weighted empirical quantile calculation,
and checks them against the analytic posterior, including the 95th
percentile of a+b, whose distribution is N(1,4/7). The numerical
tolerances are for these smooth test cases, not universal engineering
acceptance thresholds or native report tolerances.

## Sampling and limitations

The accepted sampler generates actual correlated Normal draws, evaluates
the likelihood, prior and proposal in the same coordinates, and normalizes
weights proportional to exp(log likelihood + log prior - log proposal).
It uses separate caller-supplied seed, count, ESS and maximum-weight gates.
A Normal proposal has full real support; out-of-support sampled particles
may have zero weight, even though the builder does not admit nonfinite
optimization evaluations.

The local Newton search does not prove unimodality, global mode,
integrability or global/tail coverage. Heavy tails, other modes, narrow
support, discontinuities, ill-conditioned or numerically unresolved
curvature can invalidate this approach even when a finite proposal is
constructed. For these cases use an independently specified and qualified
alternative, not a silent fallback. ESS and approximate posterior-mean
MCSE remain diagnostics of the sampled calculation, not guarantees of
extreme-quantile precision. Posterior probability limits describe flood
uncertainty conditional on the data/model, not numerical integration
precision. See `ffa explain quantile-estimands --format markdown`.

## Explicit lognormal observation binding

The independent lognormal binding accepts a caller-classified systematic
annual record, physical reference discharge r, and a supplied joint
Gaussian prior. The coordinates are the mean of ln(Q/r) and the natural
log of its SD, in that order; the caller may explicitly fix the latter,
in which case only the mean is estimated. For exact Q the log likelihood
includes the discharge Jacobian -ln Q. Positive left, right and interval
censor bounds use Normal tail probabilities; explicit missing and excluded
rows contribute no likelihood. Native flags, historical/grouped exposure
and zero-threshold/rating laws are not interpreted. The exact Year,AMS
CSV path rejects missing/zero values and hashes the source bytes. Direct
typed records retain a caller-asserted source digest, not proof of custody.

A reproducible synthetic verification uses Q=exp(1), exp(2), exp(3), r=1,
known log-flow SD=1 and mean prior N(0,1). The exact posterior mean and
variance are 1.5 and 0.25; the predictive ln(Q) distribution is N(1.5,1.25).
This bounded binding explicitly retains the unscaled proposal (`gamma=1.0`); the separate
builder supports covariance scaling, while installed configuration remains future work.
The binding constructs the local proposal, generates weighted particles
and evaluates expected AEP at exp(2) against that predictive distribution.
A separate estimated-scale test verifies actual generated particles for
a censored dataset against independent two-dimensional quadrature,
including both posterior coordinate means and expected AEP. The tests
also check same-runtime seeded replay, unit changes, independent censor
and correlated prior formulas, and invalid source admission. See the
[lognormal binding specification](https://github.com/Chain-Frost/ffa-toolkit/blob/main/docs/wp06-lognormal-binding.md)
for the independent worked examples. This is not an installed Bayesian
fitting command or legacy settings mapper.

Broader family and observation binding, output-specific numerical
precision and installed Bayesian fitting remain outstanding (Toolkit
issues #5, #6, #8 and #9). Native settings-matched comparisons remain
separate external qualification. No native runs, new dependencies or
third-party code/data are introduced. Sustainable Use License 1.0 is
unchanged.
