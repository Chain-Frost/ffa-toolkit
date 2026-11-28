# Estimated scale with fixed GEV shape or LP3 skew

## Model and evidence

Use this internal binding to integrate uncertainty in location and scale for a stated shape,
an explicit reference and a supplied joint Gaussian density. The input is a classified systematic
annual series or exact Year,AMS bytes. Every included year has one likelihood contribution.
Missing/excluded years retain their classification. The worked example below fits two observations
of the same magnitude in distinct years and retains both density factors. Source, unit, year
definition, LP3 reference, prior and numerical controls travel with the executed fit.

For GEV use x=(mu, eta), sigma=exp(eta), EVT xi fixed above -1. For LP3 use
x=(mu, eta) for Y=ln(Q/r), mu=E[Y], sigma=SD[Y]=exp(eta), fixed |g|<2.
The inherited binary64 shape exclusions also apply. Use the exact zero limits.
The normalized posterior density with respect to dmu deta is L(x) phi_P(x)/Z,
where P is the supplied positive definite covariance, including the signed cross term,
and Z is its integral. The parameter Jacobian is one in these named prior coordinates.
A density on raw scale would require a factor exp(eta) and defines a different input contract.
For each LP3 exact observation the data Jacobian is 1/Q. Censor probabilities already
integrate the density. The moving GEV boundary is mu-exp(eta)/xi. The moving LP3 boundary
in log-discharge coordinates is mu-2 exp(eta)/g. Use the lower boundary for positive shape,
the upper for negative shape, and the entire real statistic axis at zero shape.
See the [fixed-shape equations and source pages](wp06-fixed-family-posterior.md) and the
[ARR Book 3](https://www.arr-software.org/pdfs/ARR_190514_Book3_V4.2.pdf) treatment of flood
quantile estimands in Chapter 2, section 2.5, and Bayesian estimation in section 2.6.3.

In new units Q'=cQ, GEV coordinates become (c mu, eta+ln c) with covariance D P D'
for D=diag(c,1). LP3 coordinates are unchanged if r'=cr. A further change of physical
reference r to b r shifts mu by -ln b. The raw data likelihood gains c^(-n) for n exact
years. Those constants cancel in normalized posterior expectations.

## Finite support in numerical evaluation

Likelihood, CDF and survival queries standardize each observed flow before applying the WP03
unit-scale law. Work from the exact binary64 inputs, enclose exp(eta) and the two LP3 logarithms
with adjacent 80-digit decimals, and propagate the enclosures with rational arithmetic. Compare
the resulting interval with the exact standardized endpoint. This preserves distinct neighboring
flows when ordinary subtraction of logarithms rounds to zero. It also distinguishes a proven
exterior point from an interior point whose log probability underflows. Reject an unresolved
standardized coordinate or an interior distance within 32 spacings of the rounded statistic:
in that range even a one-spacing error consumes at least about 3% of the distance at the cutoff.
This guard applies to bound evaluations, independently of sample weight. Inverse kernel evaluations
use the existing WP03 conditional quantiles. Finite-sample raw-quantile aggregation and output
precision admission remain outside the estimated-scale binding.

## Propriety, moments and precision

All admitted standardized densities have a finite maximum M at fixed shape. Let n be the
number of exact years. L <= C exp(-n eta), where C includes M^n and the LP3 data Jacobians.
Censor probabilities are at most one. This exponentially tilted Gaussian bound is integrable
for every supplied full-rank covariance. On any bounded mu strip with nonempty interior,
taking eta sufficiently large puts all finite observations and censor bounds inside support.
The standardized density is positive near zero. Each exact year and each finite interval then
costs a factor proportional to exp(-eta), while half-line censor probabilities approach positive
constants. Thus Z>0, including tied records. This lower bound also supplies the large-scale
direction needed to prove divergence below. Both arguments concern the untruncated prior.

| Requested quantity | Existence for the specified posterior |
| --- | --- |
| Coordinate moments, GEV design quantile at interior non-exceedance p | All positive absolute orders. GEV q_p=mu+exp(eta) k_p where k_p is the standardized quantile. Gaussian tilting absorbs every resulting exponential linear in the coordinates. |
| LP3 design quantile in log space | All positive absolute orders by the same bound for ln(q_p/r). |
| LP3 raw design quantile to any positive power h | Finite if k_p<=0, since q_p^h <= r^h exp(h mu). Diverges if k_p>0, equivalently p>F_standard(0). On the bounded mu strip, exp(h k_p exp(eta)) dominates both Gaussian decay and the likelihood lower bound. At g=0 the cutoff is exactly p=1/2. |
| Predictive absolute moment of order h>0, GEV | Finite precisely for h xi<1. Conditional tails are polynomial of index 1/xi at positive xi and have all absolute moments for xi<=0. The finite case integrates using the same location/scale bound. Equality diverges. |
| Predictive raw discharge moment of order h>0, LP3 | Diverges for every admitted skew, including zero and negative. A future standardized variate exceeds some c>0 with fixed positive probability. On the same large-scale strip its raw h-th moment is bounded below by a constant times exp(h c exp(eta)). Negative skew bounds the conditional upper endpoint only at each fixed scale. |
| Expected AEP at a fixed flow | Exists by boundedness between zero and one. The mixture CDF and its quantiles can exist even when raw predictive moments diverge. |

At sampling order t>1 with full Normal proposal covariance G, finiteness of
the weight moment follows if t P^-1 -(t-1) G^-1 is positive definite: the integrated
envelope is a Gaussian times exp(-t n eta) and a linear exponential for shifted means.
It also bounds squared residual integrands for bounded AEP, coordinate polynomials and GEV
fixed-probability quantiles at t=2. This is a sufficient test. For example, with G=P/4
and equal prior/proposal centers the second weight moment diverges along the large-eta strip
by the lower bound above. A covariance on the boundary needs a separate likelihood-tail analysis.

The binding returns particles and coordinate summaries. Global weight checks and approximate
coordinate MCSE are diagnostics. Issue #5 owns output-specific precision admission, including
unresolved tails, saturation, inadequate proposal coverage and precision for the requested output.
A finite particle mean cannot estimate a divergent population moment. Use expected AEP or an
explicitly named log-scale quantity if that is the requested estimand. The joint aggregate
quantile-output helper includes a raw mean and is unsuitable for divergent LP3 requests.
Issue #6 owns schema, installed CLI and output routing.

## Independent comparison protocol

Use the same six synthetic cases on both Python endpoints. Shapes are +0.2, 0 and -0.2 (GEV) and
+1, 0 and -1 (LP3). The first-coordinate prior centers are 3 and 0 respectively, the log-scale
center is zero, marginal SDs are 0.25 and 0.16, and correlation is 0.45.
Use two tied exact years and one year of each censor type at the offsets in the fixed-family example.
Use 1200 draws, seed 87, gamma 1.5. Compare the evidence, two means, three distinct population
covariance entries and expected AEP at the specified flow against independent two-dimensional
quadrature. Fix each stochastic absolute tolerance at six reference asymptotic standard errors
for that specific statistic. Include the centered covariance influence function and use
unnormalized weights for the evidence. Compare quadrature on whitened prior axes at 10 and 12 SDs
with increased integration order. Require absolute agreement within 2e-8 on the seven reference
values and relative agreement within 1% on their variance coefficients. These are regression
criteria for the stated draws, integrands and proposal.

The two exact observations have offset 0.2. Left and right bounds are 0.8 and -0.2.
The interval is [-0.3,1.1], and the evaluation threshold is 1. Add 3 to offsets for
GEV, or apply Q=2 exp(offset) for LP3. In the independent oracle the variables are
mu=center+0.25u and eta=0.16(0.45u+sqrt(1-0.45^2)v), for independent standard Normals
u,v. Apply tensor midpoint quadrature to direct GEV, Normal or Erlang-4 densities and
tails. Each variance coefficient integrates target^2/proposal against the squared
influence of its output. For evidence subtract Z^2 from the second raw weight moment.

The domain is a quadrature choice only. For these six cases the largest standardized
density is below one and each LP3 exact Q exceeds one. The evidence envelope is thus
the whitened prior tilted by exp(-2 eta). The tilt shifts the Normal center by at most
0.32 SD and multiplies its mass by exp(2*0.16^2). Outside 10 SDs the union of the four
tails has envelope mass below 1e-21. For the variance integrals, G-P is positive definite
in these runs: the precision of the whitened squared-weight envelope is at least I.
Its mean norm is at most the whitened proposal-center displacement plus 4*0.16
(the exp(-4 eta) tilt); this is below 1.3. Thus a 12 SD box leaves at least 10.7 SDs
in every marginal tail before multiplication by the polynomial residuals (degree at most
four). Widening the box from 10 to 12 and refining from 81 to 145 points per axis checks
the effect on each reported integral.

| Family, fixed shape | Z | E[mu] | E[eta] | Var(mu) | Cov(mu,eta) | Var(eta) | E[AEP] |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GEV +0.2 | .02266062 | 3.02318019 | -.05327638 | .04985362 | .01258216 | .02340153 | .32654726 |
| GEV 0 | .02627421 | 2.99587927 | -.06206699 | .04914636 | .01252914 | .02339088 | .29331643 |
| GEV -0.2 | .03170489 | 2.96716412 | -.07190806 | .04837257 | .01240370 | .02333189 | .25132134 |
| LP3 +1 | .00376041 | .09356525 | -.03289074 | .04916460 | .01156414 | .02299172 | .16727865 |
| LP3 0 | .00534194 | .02208882 | -.05221361 | .04662915 | .01134596 | .02288832 | .15778560 |
| LP3 -1 | .00844557 | -.05389124 | -.07665486 | .04409322 | .01056575 | .02239257 | .11248936 |

For example, in the positive-skew LP3 case the probability that a new annual maximum
exceeds 2e in source units, averaged over parameter uncertainty, is about 0.1673.
The raw design-quantile mean at p=0.99 diverges in this same example: for Erlang-4,
F_standard(0)=1-exp(-4)*(1+4+8+64/6), below 0.99. Run
`python -m pytest tests/test_wp06_estimated_scale.py` to reproduce the comparisons.

## Subsequent estimated shape

Specify a normalized joint prior and its coordinate support before fitting a third coordinate.
Compact subsets of xi>-1 or |g|<2 avoid the density singularities. Prove a uniform likelihood
bound and choose a proposal whose density covers the entire admitted support. Include the
Jacobian of any unconstraining transform and the cross-covariances with location and scale.
With tied exact years an upper-endpoint GEV or either nonzero LP3 skew contributes the local
power m(a-1) in distance to the endpoint, with a=-1/xi or 4/g^2 respectively. For regular
positive remaining factors the local posterior needs m(a-1)>-1 and the second weight moment
needs 2m(a-1)>-1. A prior/proposal vanishing at that boundary changes the weight integral.
A bounded prior alone cannot fix the LP3 near-zero kernel approximation, or admit a predictive
moment when shape approaches its critical value. Verify joint endpoint, scale and shape tails,
normalization, weight variance, shape-zero continuity, and independent three-dimensional
references before extending the current shape restrictions.
