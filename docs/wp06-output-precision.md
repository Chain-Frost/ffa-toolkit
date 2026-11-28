# WP06 — Output-specific importance-sampling precision (proposed contract)

**Mathematical contract with a bounded fixed-scale implementation; broader model admission and integration remain pending.**
Tracking: [#5](https://github.com/Chain-Frost/ffa-toolkit/issues/5),
initial delivery [#1](https://github.com/Chain-Frost/ffa-toolkit/issues/1).
Model binding #3, proposal scaling #4, CLI integration #6 and presentation #7
are separate scopes.
This specification is for the existing independent self-normalized importance
sampler (SNIS), not a different posterior, an adaptive sampler or native
Flike qualification. Native qualification remains separate.

The internal fixed-budget audit also admits parameter mean and expected
CDF/direct AEP for classified fixed-scale censored records when explicitly
configured with [deterministic reference controls](work/wp06-censored-precision-admission.md).
Each successful check requires both endpoints of the Decimal reference
enclosure inside the relevant interval. The exact-record route retains
its conjugate reference and existing criteria.

## 1. Source and scope

Read the merged [execution contract](wp06-importance-execution.md),
[weighted-output evidence](wp06-weighted-output-evidence.md),
[curvature proposal](methods/importance-sampling-reference.md),
[shared contracts](shared-contracts.md) and actual
`importance_sampling.py`, `bayesian_outputs.py`, `contracts.py` at
the current implementation. These components supply the estimands, actual
iid proposal draws and local proposal construction. Their numerical tests
and ESS/max-weight gates do not establish output accuracy.

- [ARR Book 3 v4.2](https://www.arr-software.org/pdfs/ARR_190514_Book3_V4.2.pdf),
  §2.6.3.6–7, equations 3.2.52–54: source basis for proposal/target
  integration and weighted flood outputs, as documented in the execution contract.
- [Kuczera (1999)](https://doi.org/10.1029/1999WR900012), bibliographic
  context for flood-frequency Bayesian inference, as listed in the
  [reference register](references.md). The precision equations below
  are independently derived from ratio estimation and do not rely on
  an unverified source-specific error formula or numerical default.
- [Owen, Monte Carlo theory, methods and examples, ch. 9](https://artowen.su.domains/mc/Ch-var-is.pdf),
  §9.2, printed p. 9, equations (9.8)–(9.9), inspected in the PDF:
  the ratio-estimator delta-method variance and its sample estimate.
  §9.3 distinguishes weight and variance-estimate diagnostics.
- [Glynn (1996), Importance Sampling for Monte Carlo Estimation of Quantiles](https://web.stanford.edu/~glynn/papers/1996/G96.pdf),
  §2, Theorem 1, PDF p. 3, inspected in the PDF: SNIS empirical-quantile
  CLT (third moment condition and positive derivative at the true quantile).
  The density in its denominator is not the annual-flood density when
  estimating a posterior credible limit of a design quantile.

The derived equations and proposed audit policy below are original.
No source PDFs, source tables or third-party implementation are copied.
Sustainable Use License 1.0 and existing dependencies are unchanged.
The proposed contract is independent of the covariance-scaling child:
use a fully admitted, unchanged proposal for all audit streams.

## 2. Estimands and units

Write t(theta) = likelihood times prior (including source-coordinate
Jacobians), Z = integral t, posterior pi = t/Z, proposal g, conditional
annual-flood CDF F(x|theta), survival S(x|theta) = P(Q>x|theta), and
q_p(theta) = inf{x: F(x|theta) >= p}, where 0 < p < 1.
Annual AEP a = 1-p; a posterior probability level b is a **different**
axis from p. Q uses the declared discharge unit. Define inverse CDFs
as generalized inverses; retain the existing weighted left inverse ECDF.

| Existing ResultMeaning / quantity | Target functional | Precision route |
| --- | --- | --- |
| POSTERIOR_MEAN_QUANTILE | E_pi[q_p(theta)] when absolutely integrable | Scalar expectation, discharge units |
| EXPECTED_ANNUAL_EXCEEDANCE_PROBABILITY | E_pi[S(x\|theta)] at fixed x | Scalar expectation, fraction, direct SF |
| EXPECTED_NON_EXCEEDANCE_PROBABILITY | E_pi[F(x\|theta)] at fixed x | Scalar expectation, fraction, direct CDF |
| POSTERIOR_PREDICTIVE_QUANTILE | inf{x: E_pi[F(x\|theta)] >= p} | Conditional-CDF inversion bracket, discharge units |
| POSTERIOR_MEDIAN_QUANTILE | inf{x: P_pi(q_p(theta) <= x) >= 0.5} | Posterior-indicator inversion bracket |
| QUANTILE_MONTE_CARLO_PROBABILITY_LIMIT | same inverse at b=(1-m)/2 or 1-(1-m)/2 for credible mass m | Separate posterior-indicator bracket for each endpoint |
| RETURN_PERIOD_FROM_EXPECTED_AEP | 1/E_pi[S(x\|theta)], if positive and finite | Transform simultaneous AEP bounds, years |
| RETURN_PERIOD_MONTE_CARLO_PROBABILITY_LIMIT | inverse posterior CDF of 1/S(x\|theta) at b, with +infinity at true S=0 | Separate posterior-indicator bracket, years, finite endpoints only |
| Source-coordinate parameter posterior mean | E_pi[theta_j] | Scalar expectation, coordinate unit |
| LOG10_QUANTILE_POSTERIOR_MOMENT | E_pi[log10 q_p] and SD_pi(log10 q_p), if q_p>0 a.s. and moments exist | Mean; separately bounded second moment / SD (below) |
| QUANTILE_AT_EXPECTED_PARAMETERS | q_p(E_pi[theta]) | Not the mean of quantiles; no first-stage numerical-precision claim |

The credible mass m, posterior limits and log-moment SD describe
conditional flood-estimate uncertainty, **not integration error**.
The predictive quantile concerns a future annual flood averaged over
the posterior. Return period from mean AEP is not the mean of the
per-draw return periods. Keep identities, units, probability type,
input discharge and its unit, interval mass, side and construction
from `ResultPoint`; neither a probability limit nor a precision bound
should be relabelled as the other.

## 3. Scalar precision: derivation and admission

For independent theta_i ~ g, r_i = t(theta_i)/g(theta_i),
w_i = r_i/sum r_j. For a scalar integrand h and mu = E_pi[h],
mu_hat = sum w_i h_i. In exact arithmetic,

    mu_hat - mu = mean_g[r_i (h_i - mu)] / mean_g[r_i].

Require 0 < Z < infinity, t absolutely continuous with respect to g,
E_pi|h| < infinity, iid draws from the same fixed proposal, exact
same-coordinate densities and deterministic valid h. The law of large
numbers yields consistency. A CLT for the centered numerator with
finite variance, followed by Slutsky's theorem, gives asymptotic variance
V/n with V = E_g[r^2(h-mu)^2]/Z^2. A conservative admission condition
for a plug-in estimate is E_g[r^2] and E_g[(rh)^2] both finite;
this also gives convergence of the empirical centered second moment.
The approximation is

    se_hat = sqrt(sum_i (w_i (h_i - mu_hat))^2).

This is Owen (9.9), without a second division by n. It is not the
posterior SD divided by sqrt(global ESS), an unbiased variance estimator
at finite n, a bias bound, or a universal confidence guarantee.
For bounded F/S/indicator h, finite E_g[r^2] is sufficient for
the scalar second-moment condition. The upper-tail AEP must use SF,
not 1 minus rounded CDF. For unbounded h, check the actual product
moment as well as the posterior moment, not just the weights.

Use robust scaling (as the existing sampler does for its parameter
mean diagnostic), stable weighted sums and no arbitrary weight clipping,
discarding, winsorization or PSIS. Finite representability is necessary,
not evidence of finite mathematical moments. A finite positive-weight
integrand with zero observed residual variance (including every observed
indicator equal) is **indeterminate** for a stochastic functional:
it may reflect missing support or rounding, not exact knowledge.
An analytically established constant on the whole target support can
instead be marked exact, with its derivation and no stochastic MCSE.
A positive mathematical contribution rounded to zero or a finite log
weight normalized to zero must be detected: use a verified log-domain
method or withhold precision rather than silently treat it as
out-of-support. Preserve genuine zero target mass and all raw density
terms; do not alter existing kernel failure policy.

### Why global ESS is insufficient: an admitted-model warning

For a proper Gaussian posterior N(m,v) and proposal N(m,s^2), direct
Gaussian integration gives E_g[r^2] < infinity only if s^2 > v/2
(and E_g[r^3] < infinity only if s^2 > 2v/3), for normalized r.
Full support is not enough, even for bounded h. In a finite sample a
narrow proposal may fail to encounter the problematic tail at all.

More importantly, the proposed free log-SD lognormal binding in #3
specifies a full untruncated SPD Gaussian prior on (mu,eta), eta=ln(sigma),
positive reference r0, and positive exact / fixed finite censor bounds.
Its fixed-p design flood is

    q_p = r0 exp(mu + z_p exp(eta)),  z_p = Phi^-1(p).

For p>1/2, an untruncated estimated eta with the stated Gaussian prior
and finite systematic exact/positive censor observations has an
**infinite posterior mean of q_p**. To see this, restrict mu to a fixed
bounded interval of positive length and take eta -> +infinity. Each
exact flow density is bounded below by a positive constant times
exp(-eta); a nonzero-width finite log-censor interval probability is
also bounded below by a constant times exp(-eta); a one-sided positive
finite censor probability tends uniformly to 1/2. A nondegenerate
Gaussian prior on this strip has a lower bound c exp(-C eta^2 - D eta)
for sufficiently large eta (finite positive c,C and finite D).
Thus t(theta) q_p(theta) on the strip is bounded below by a positive
multiple of exp(z_p exp(eta) - C eta^2 - D' eta); its integral diverges.
The posterior itself is proper: the likelihood is bounded above by
a constant times exp(-n_exact eta) and the Gaussian prior integrates
this exponential. The result does not depend on the local proposal,
random seed, number of draws or ESS. It does not apply to the fixed-SD
case or a different justified bounded/tail-restricting prior.

Do not report the finite sample mean as a defined population expectation
or precision-pass for that free-scale, upper-design-quantile case. Keep
the predictive quantile and posterior quantile limits separately assessed;
they may be well defined even when the mean is not. This is a derived
finding to be independently checked against the admitted model and CLI
contracts in #3 and #6 before runtime admission,
not permission to silently truncate the prior. For known fixed sigma and
a conjugate Normal posterior for mu, the design quantile is lognormal
under the posterior and has finite moments. Proposal/product moment
evidence is still required separately. Log10(q_p) is affine in mu and
exp(eta), not exp(exp(eta)); the exact/censor likelihood upper bound above
and Gaussian prior imply finite posterior log moments in the free-scale
case. Their SNIS product moments are a different condition.

## 4. Inverse outputs: audit in probability space, not heuristic density

For a posterior endpoint let H(x)=E_pi[1{q_p(theta)<=x}] (or the
corresponding return-period indicator) and b the desired posterior
probability. For a predictive quantile let H(x)=E_pi[F(x|theta)]
and b=p. H is a right-continuous CDF; v=inf{x:H(x)>=b}.
A continuous strictly increasing local H with H'(v)>0 gives an
inverse-CDF delta-method variance V_H/(n H'(v)^2) for a regular inverse,
where V_H is the scalar asymptotic variance for the CDF integrand at v.
Glynn's empirical-quantile result assumes a finite third moment of the
ratio and a positive derivative. For the predictive inverse the
integrand is conditional CDF, not a posterior indicator: analogous
linearization needs local empirical-process regularity and a justified
derivative of the integrated CDF. Do not plug in an arbitrary density,
spacing bandwidth or annual-flood PDF for the posterior indicator law.

**Proposed first bounded implementation:** avoid estimating H'(v).
Compute a finite primary estimate v_hat with the existing inverse.
With a caller's absolute tolerance d>0 form a representable window
[L,U]=[v_hat-d,v_hat+d]. Round inward when necessary (never widen
the declared tolerance) or mark indeterminate if distinct finite
endpoints cannot be represented. Predeclare d; the primary run sets
the bounds, and **independent** audit runs then evaluate fixed-point
scalar expectations at L and U. Conditional on the primary draw set
these points are fixed. In exact arithmetic the sufficient bracket is

    H(L) < b and H(U) >= b  =>  L < v <= U.

Use strict separation from b on both sides for numerical and statistical
acceptance (conservative at ties). For b<=1/2 evaluate the CDF directly:
upper one-sided bound for H(L) < b, lower one-sided bound for H(U) > b.
For b>1/2 use the direct complementary probability K(x)=1-H(x):
lower bound for K(L) > 1-b and upper bound for K(U) < 1-b.
K is direct conditional survival for predictive outputs and strict
greater-than indicators for posterior outputs. Never subtract a
rounded CDF to estimate a rare survival. The chosen representation is
fixed for all audit streams. Use the configured probability level
actually admitted by the existing output API; reject indistinguishable
0/1 or nonrepresentable complements instead of silently changing levels.

This approach checks location in a **user-specified discharge/period
window** without converting a probability MCSE to a flow MCSE. It does
not claim a quantile standard error. It is conservative for atoms,
ties, flat CDFs, nearly zero slopes and collapsed tails. If the
two strict inequalities cannot be established the audit cannot pass.
Use directly evaluated conditional CDF/SF and the same weighted
indicator/inverse definition as the primary estimator. Record
probability bounds, per-side MCSEs, event/non-event positive-weight
counts and masses, plus existing tail ESS as diagnostic context only.
Do not interpret zero sampled tail probability as zero true probability.
For return-period endpoints, indicators can be evaluated through
the equivalent survival threshold if needed to avoid reciprocal
overflow, respecting strict/non-strict ties and the exact S=0 case.
Infinity or a zero-survival underflow without support evidence is not
a finite endpoint precision pass.

## 5. Fixed-budget, independently replicated audit

This is a proposed **conditional large-sample audit policy**, not a
sequential stopping procedure or a distribution-free certificate.
A caller must predeclare:

- exact model, data/settings identity, proposal and coordinates,
  outputs/axes/units, posterior interval mass, and a positive finite
  absolute tolerance in each output's unit (no blanket relative
  percentage; a user-chosen scale can be converted to an explicit
  absolute tolerance before execution);
- one primary seed, at least two audit seeds, distinct explicit
  uint64 seeds fixed without inspecting outcomes, equal declared
  draw count per stream (existing 2..1,000,000 range), and maximum
  total draws and number of scalar evaluations. Validate feasibility
  and integer products before running. No fallback seeds or partial
  success selection. Distinct local Random instances and seeds
  implement separate reproducible streams under the PRNG independence
  approximation; distinct seeds alone are not a mathematical proof of
  disjoint or statistically independent sequences;
- sampler ESS/max-weight gates as currently required; scalar
  one-sided significance budget alpha with 0<alpha<1, explicit
  per-output minimum sampled event and non-event counts/weight-ESS
  where applicable, and evidence for target propriety, support,
  required moments and practical tail coverage. A caller's numerical
  thresholds are policy, not evidence that the assumptions hold.
  Use identified model/proposal-specific analytic or independently
  verified numerical arguments; unknown evidence cannot be promoted
  by a boolean checkbox, successful seeds or normality tests.
  Predeclare the criterion source/rationale. Existing proposal/source
  provenance and package/runtime/seed records must remain available.

For each output use the **primary estimate unchanged**. Each audit
stream estimates the required scalar expectation(s), independently
of the primary stream and other streams. Let M be the total number
of one-sided bounds across all requested outputs and audit streams
(count both sides of scalar intervals; count two for an inverse
bracket; include transformed-moment checks). Fix M before sampling;
do not reduce M after a failed/undefined output. Put
c = -Phi^-1(alpha/M), requiring a finite positive c and a representable
tail probability. For each scalar audit estimate m_j, form required
bounds m_j +/- c se_j, outward-rounded conservatively. These are
asymptotic marginal bounds under the admitted assumptions. Bonferroni
gives a simultaneous *asymptotic*, conditional-on-primary error
budget for the fixed checks; no independence between functionals is
needed for the union bound. Normal approximation, SNIS bias, finite
sample tail discovery and numerical errors are not eliminated by
Bonferroni. This is not an exact finite-sample coverage statement.

For direct expectations require every audit two-sided interval to be
contained in the declared primary window. For inverse outputs require
both strict probability-bracket inequalities in every audit stream.
Retain each audit's own point estimate when computed, and max/min
replicate spread as empirical diagnostic, not an SE or an alternative
pass rule. For the same primary output require the computed point
estimates of all audit streams to lie in its window as an additional
predeclared stability check. Never report the best seed or a
successful subset. A zero-width stochastic interval is indeterminate.
Audit replication compares the same target and proposal, not different
priors, transformations, proposals or data. Verify identities rather
than trusting matching coordinate names alone.

A reciprocal expected AEP is admitted only if primary and audited
AEP bounds are strictly positive and the finite transformed interval
[1/U_a,1/L_a] is contained in the return-period window. Bounds may
be intersected with proved [0,1] support, not arbitrarily floored.
No infinite/near-zero-AEP substitution. For log10 moments the mean
uses the scalar route when strictly positive support and finite
moments are established. For its SD, let m0 be the finite primary
log mean, h=log10(q_p), and audit the two scalar means
a=E[h-m0], b2=E[(h-m0)^2]. Obtain simultaneous bounds for both
(4 one-sided bounds per stream) and propagate via SD=sqrt(b2-a^2),
using interval extrema of a^2 and mathematically nonnegative
variance. A negative upper variance bound, nonrepresentable
subtraction or unsupported required product moments is indeterminate.
This avoids an uncentered moment subtraction; it is not a claim
that SD itself is asymptotically Normal. Treat numerical error and
nonpositivity separately. Defer SD runtime admission if not reviewed;
do not substitute the posterior SD as a Monte Carlo error.

Quantile-at-expected-parameters needs a separate verified parameter
confidence region and nonlinear propagation (including kernel support
and parameter transformation); no generic delta method or scalar
mean-of-quantiles diagnostic certifies it. Retain the computed value
and explicitly track unassessed precision for it under #5.

Run the full scheduled plan even when another attempt has failed,
unless a genuine resource/interrupt condition prevents execution.
Retain attempted, successful, failed, indeterminate, and not-attempted
counts, seed, configured/actual draw count when available, reason
and available diagnostics for each. The existing sampler raises on
weight/density failures and may not expose partial particles; record
the failed attempt and unavailable counts rather than fabricate
data. CLI orchestration (#6) owns the attempt ledger; precision
mathematics (#5) owns per-output diagnostics and aggregation. A separate statistical specification
is required before any adaptive stopping, choosing proposal or seed
after inspecting an audit, or repeated budget escalation with a
nominal confidence claim. A new predeclared audit can be reported
separately, never erase the old one.

## 6. Fail-closed result contract and integration

Proposed internal precision outcomes (coordinate exact naming with
#6 and #7 before adding a runtime contract):

| State | Meaning |
| --- | --- |
| conditional_pass | Every predeclared check passed, all attempts retained, mathematical assumptions/evidence and numerical resolution admitted. Conditional approximate precision audit, not convergence/native qualification. |
| criteria_not_met | Valid diagnostics did not establish requested tolerance; not proof that actual error exceeds it. |
| indeterminate | Missing/unverified moment, support or practical tail evidence; insufficient tail resolution; degenerate sample residuals; numerical saturation/underflow; unsupported precision route; incomplete plan. No valid MCSE claim. |
| undefined | Functional is mathematically undefined or divergent under the model; keep reason and do not publish a finite sample surrogate as that population quantity. |
| execution_failed | Density/kernel/sampler/representation/weight-gate failure; retain seed, attempt and available reason; no selected successful subset. |

Nonfinite or invalid configuration is rejected before execution.
Every requested output gets its own outcome. Any failed, indeterminate,
undefined or unmet required output prevents an overall conditional
pass; a failed attempt cannot disappear. If multiple reasons apply,
retain them all. A report may show a computed *unqualified numerical
estimate* for an indeterminate functional known to exist, but not
imply a validated MCSE or a defined mean when divergence is proved.

Do not change `EvidenceStatus` (legacy artifact admission), or use
`ResultPoint.confidence_level` / probability-limit construction
for integration uncertainty. Match precision records to the complete
result identity (meaning, p/AEP or x with unit, mass, side, log moment,
model/settings and run). A `RunDiagnostic` can carry finite numerical
diagnostics with explicit unit and output identity; `AnalysisResult.warnings`
can explain unavailable estimates. Nonfinite placeholders, fake zero
MCSE or implicit success are prohibited. Preserve typed internal
outcomes/attempt evidence separately; coordinate CLI and reporting before
changing shared schema or installed reporting. If current containers
cannot carry the full identity/status without ambiguity, withhold the
CLI precision claim until that explicit bridge is reviewed. #6 owns
file/settings/seed orchestration, #7 owns report/plot presentation.
This specification owns no CLI dispatch, plotting, proposal builder
or workflow changes.

## 7. Verification plan and independent review gates

Use actual sampler executions, not deterministic quadrature passed
off as draws. Test independent equations and exact invariants first.
All synthetic test tolerances and seed schedules are declared for
their own mathematical cases, not imported as engineering defaults.

1. For an independently constructed known-SD lognormal model with r0=1, observations
   exp(1), exp(2), exp(3), sigma=1 and prior mu~N(0,1), the posterior
   is N(1.5,0.25). At p with z=Phi^-1(p):
   - mean design flood = exp(1.5+z+0.125);
   - median and b-posterior endpoint = exp(1.5+z+0.5 Phi^-1(b));
   - predictive quantile = exp(1.5+sqrt(1.25) z);
   - expected AEP at x = 1-Phi((ln(x)-1.5)/sqrt(1.25));
   - log10 mean = (1.5+z)/ln(10); SD=0.5/ln(10).
   Independently verify all identities, including distinction of
   q_p(E[theta]) and mean q_p, both tails, return periods and units.
   For g=pi, compare sample SNIS errors to known integrand variance
   (e.g. Var(q_p)=(exp(0.25)-1) exp(2(1.5+z)+0.25)).
   Use a broader nonidentical Gaussian proposal and independent
   numerical quadrature for bounded AEP and inverse probability
   integrands. Inspect coverage/empirical spread across predeclared
   repeated runs including failures; do not pick favorable seeds.
2. Exact synthetic weight arrays check the centered ratio formula
   with independently calculated sums, zero genuine weights,
   different units, cancellation, common log-ratio scale, subnormal
   residuals, invalid shapes and no false zero from underflow.
3. Analytic narrower Gaussian counterexample s^2<=v/2: finite observed
   weights and bounded h do not establish SNIS variance; require
   indeterminate despite a possible high ESS. A target with a small
   remote component gives the same lesson for practical tail coverage.
   Exercise unsupported/absent evidence and finite but highly
   concentrated product residuals. Include mathematically divergent
   free-log-SD design-quantile mean from §3 (analytic proof, not an
   attempted empirical convergence demonstration).
4. Weighted posterior indicator and predictive bracket controls:
   known continuous inverse, jumps/atoms and ties, flat/local shallow
   CDF, almost all mass on one side, rare SF with CDF rounded to 1,
   genuinely zero and numerically underflowed survival, unrepresentable
   tolerance, nonfinite/undefined outputs and finite/infinite period
   endpoints. An atom need not fail if a strict bracket is established.
   Independently verify bracket orientation and every one-sided bound.
5. Fixed-plan tests: distinct seeds, invalid counts/budgets, identical
   model/settings identity and changed-model rejection, independent
   primary/audit streams, per-output acceptance/aggregation, stable
   replay, failure retention and no seed filtering, predeclared M
   with failed outputs still counted, missing and not-attempted runs.
   A successful sample mean gate must not certify tail bounds.
6. Final implementation: focused tests then existing repository quality
   gates at coherent head, Python 3.11 ordinary PR / 3.11 and 3.14
   main policy. Add method-register implementation/evidence updates
   only once the diagnostics exist, with #3/#6/#7 owners to avoid
   conflicting edits. Independent scientific review of this proposal
   first; independent Codex review of final functional changes and
   tests afterward. This docs-only checkpoint does not fulfill either
   runtime or final review acceptance.

Outstanding until separately reviewed/implemented: explicit admitted
model/proposal-specific moment and practical tail-evidence policy
(especially free log-SD), inverse-bracket implementation and
regularity/numerical resolution checks, log SD and return-period
transforms, quantile-at-mean-parameter error propagation, full attempt
ledger and CLI/reporting integration. Any density-based quantile MCSE,
adaptive stopping, or new proposal/smoothing method requires a separate
justification. Do not move these unsolved scientific matters to
native qualification. Keep #5 open until its mathematical and software
acceptance is met. No native-equivalence claim.


## Bounded implementation checkpoint

The internal `importance_precision` module implements this policy for the
explicit exact-record lognormal binding with fixed log-SD and a supplied
one-coordinate Gaussian mean prior. It accepts explicit missing/excluded
rows but no included censor rows. Its conservative admission condition is
proposal variance at least the prior variance. For posterior N(m,v) and
proposal N(g,s²), where s²>v, the supremum of the normalized density
ratio has logarithm (log(s²/v)+(g-m)²/(s²-v))/2. The caller supplies a
maximum admitted log ratio; passing this condition establishes a bounded
ratio under the stated exact Normal mathematics, not a universal proposal
selection rule. All positive or polynomial/lognormal integrands admitted
here have finite required moments in that model. Each audit additionally
checks the analytic scalar reference or inverse probability against the
corresponding integration bounds. No analytic value replaces the primary
weighted sample estimate.

The primary and independent audit streams use one model and one proposal.
The bound budget is reserved before execution (including four one-sided
bounds for log10 SD). Seed schedules, resource budgets and concentration
limits are explicit. Positive and negative centered-contribution count,
mass and effective size supplement the global sampler weight gates; for
indicator integrands these are event and non-event resolution measures.
Continuous conditional-probability integrands instead use the two sides
of their centered contributions. Failure to resolve either side with the
predeclared criteria is indeterminate. The component rejects saturated
component CDF/SF values in the admitted continuous positive-support model,
positive weights rounded to zero and unresolved stochastic intervals.

Only the known-scale one-coordinate expected-parameter quantile admits
a monotone parameter-mean interval transformation. The centered two-moment
SD route admits only the same fixed-scale model. Both retain the calculated
primary output and distinguish the conditional audit from posterior
uncertainty. The tests use original synthetic exact records and the
independent conjugate formulas above, not restricted datasets. No new
native or general-model acceptance is implied. In particular, estimated-
scale upper quantile means are marked undefined under the admitted full
Gaussian prior; the other estimated-scale and known-scale censored
precision routes remain unadmitted. The CLI/report bridge and broader
model-specific scientific validation remain outstanding under #5/#6/#7.

### Numerical representation and incomplete diagnostics

For known log-SD, log10 design quantiles are evaluated directly as the
affine function `(ln(reference) + mu + sigma*z_p)/ln(10)`. Their moment
checks do not require an exponentiated discharge to be representable.
Tests cover both overflowing and underflowing discharge-space quantiles
while independently verifying the finite logarithmic means and SDs.
A later inverse-bound or second-moment resolution failure retains any
scalar diagnostics already computed for that audit. It remains
indeterminate and never promotes an unsuccessful attempt to a pass.
