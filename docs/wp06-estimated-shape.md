# GEV shape and LP3 skew estimated jointly with location and log-scale

Use the existing `ffa fit` command to integrate shape uncertainty when a bounded shape law,
its correlations and the observation classifications can be specified. The numerical evidence
here concerns the stated independent models. Shape bounds and the proposal scale are modelling
and numerical choices supplied by the caller. They have no inferred Flike defaults.

```console
ffa fit --ams docs/examples/bayesian-shape-ams.csv --settings docs/examples/bayesian-shape-settings.json --output shape-fit
```

The [settings](examples/bayesian-shape-settings.json) and [classified annual series](examples/bayesian-shape-ams.csv)
run the positive-skew LP3 example below. Its expected annual exceedance probability at
Q=2e is about 0.1664, averaged over all three uncertain parameters. The CLI preserves the
source/settings bytes and hashes. JSON, CSV and text reports use the installed Bayesian reporting
path. Source classification, units, year definition, LP3 reference, RNG, seed, sample size, weight
gates and the unassessed precision mode are explicit. The schema is `ffa-bounded-shape-bayesian/1`.
Use `gev_bayesian` and a null reference for GEV, or `lp3_bayesian` and a positive physical reference
for LP3. Exact Year,AMS and classified systematic CSV are supported. Missing/excluded years,
tied exact values and the three censor types follow the [estimated-scale contract](wp06-estimated-scale.md).

## Normalized prior, support and simulation

Write x=(mu,eta,s). In GEV mu is location, eta=ln(scale), and s is EVT xi. In LP3 mu and exp(eta)
are the mean and SD of ln(Q/r), and s is its skew. Given finite bounds a<b, define
z=ln((s-a)/(b-s)). The prior is specified directly by three means m and a lower triangular
factor B with positive diagonal in (mu,eta,z), with covariance BB'. The settings supply all nine
entries in order, including upper zeros. Correlation with the physical skew or xi after the
nonlinear transformation generally differs from correlation with z. For u standard trivariate
Normal, set (mu,eta,z)=m+Bu, and s=a+(b-a)expit(z). The prior's support is the open interval;
this transformation constructs a normalized distribution from the outset.

The physical-coordinate density is phi(u)/J, where
J=det(B)(s-a)(b-s)/(b-a). The log-scale coordinate is already the logarithm of scale.
Each LP3 exact year contributes the separate raw-data Jacobian 1/Q. Censored years use
one CDF, one survival or an interval probability, and require no additional data Jacobian.
Changing a flow unit or LP3 reference transforms mu, eta, their factor rows and observed
data as in the [two-coordinate derivation](wp06-estimated-scale.md). A shape prior on ARR
kappa would need sign reversal of EVT shape and its dependence structure.

The prior and proposal share the bijection. The caller supplies gamma for independent Normal
innovations with covariance gamma²I under the proposal. The output particles carry physical
coordinates and both physical log densities, including their common Jacobian. Drawing and
weighting reuse the existing independent sampler. At gamma=1 the weight is just the likelihood.
The reported log evidence is the logarithm of the arithmetic mean of the unnormalized weights.
The log itself need not be unbiased. Evidence units depend on the units of the exact observations.
Posterior means, SDs, covariance and quantiles use physical shape, including for the quantile at
expected parameters. See [ARR Book 3, chapter 2, sections 2.5 and 2.6.3](https://www.arr-software.org/pdfs/ARR_190514_Book3_V4.2.pdf)
for the likelihood, expected probability and importance integration. The bounded shape prior and
prior-scaled proposal above are explicit independent choices.

## Propriety, support and weight moments

The closed bounding interval stays within xi>-1 for GEV; for LP3 it stays strictly inside
(-2,2), on one side of the nonzero near-Normal band |g|<1.6e-5. The fixed-shape zero limits
remain available. For either family the standardized density has a uniformly bounded maximum M
on each admitted compact shape interval. For instance the GEV maximum is
(1+xi)^(1+xi) exp(-(1+xi)), continuous also at zero. For LP3 the Gamma shape 4/g² is greater
than one and the standardized mode height varies continuously on the admitted interval.

For n exact years, the raw likelihood obeys L<=C exp(-n eta), including the LP3 data Jacobians.
The tilted Gaussian upper bound gives finite evidence. On a bounded location strip, a sufficiently
large scale puts all finite data bounds inside the support uniformly on a compact interior shape
interval. Standardized densities there are positive near zero. Exact years and finite censor
intervals cost at most exponential decay in eta, and half-line censor probabilities tend to
positive limits. This region has positive prior mass, proving positive evidence.

An upper-endpoint GEV or either nonzero LP3 skew has density proportional locally to distance
raised to A-1, with A=-1/xi or 4/g². Tied exact years multiply this exponent; a vanishing censor
probability can change it again. Our compact bounds keep A>1 in these endpoint cases. Singularity
outside those bounds cannot be admitted by checking just a single exact year. Source-coordinate
endpoint enclosures from #38 distinguish exterior zero mass from an unresolved interior density.
GEV converges to Gumbel from both signs by ln(1+xi t)/xi -> t. A nonzero subnormal xi that the
binary64 kernel cannot resolve aborts the evaluation. For LP3 the standardized Gamma has third
cumulant g and tends to Normal as g tends to zero. The first CDF correction at its mean is
g phi(0)/6. A test on both admitted edges of the approximation band checks that correction.
Replacing a continuous interval of nonzero skew by a Normal kernel would change the model.

For order t>1, the weight integral in innovations is proportional to the integral of
L^t exp(-(t-(t-1)/gamma²)||u||²/2). A positive quadratic coefficient absorbs the linear
log-scale tilt, coordinate polynomials and all fixed single-exponential scale factors.
Conversely take a tube with bounded mu and large but bounded eta so the data are inside support
at a limiting shape, and let z go to infinity. The likelihood stays bounded below on a sub-tube.
The integral diverges for a zero or negative quadratic coefficient. Thus the second weight
moment requires gamma²>1/2 for this proposal; the implementation rejects smaller or equal
values. This establishes a moment condition. Proposal concentration, rare-event resolution and
the error of any finite reported estimate remain output-specific. [Issue #5](https://github.com/Chain-Frost/ffa-toolkit/issues/5)
owns precision admission.

## What exists

- GEV design quantiles mu+exp(eta) k_p(s) have all absolute polynomial moments at fixed interior
  p because k_p is bounded on the closed shape interval. The raw predictive absolute moment of
  order h>0 diverges if h b>1, since some shapes have an infinite conditional moment. It exists
  for h b<1. At h b=1 the interior conditional moment grows at most as a constant times
  1/(1-hs) near b by the GEV tail integral. The scaled logit makes this at most a constant
  times (1+exp(z)), integrable jointly with exp(h eta) under the Gaussian envelope.
- GEV raw design quantiles can be nonpositive on a region of positive posterior mass even when
  all sampled quantiles are positive. Choose a large scale and place the threshold for the
  p-quantile below zero while retaining interior support for finite observations. For upper-bounded
  GEV the upper endpoint can remain above all observations. Real log10 design-quantile moments
  are undefined in this setting and the report omits them explicitly.
- LP3 log-design quantiles ln(q_p/r)=mu+exp(eta) k_p(g) have all polynomial moments. Positive
  raw design moments diverge when k_p(g)>0 on an open subinterval: keeping mu and z bounded,
  exp(h k_p(g) exp(eta)) dominates the Gaussian and the likelihood lower bound. If k_p<=0
  everywhere, q_p^h<=r^h exp(h mu). The dividing probability is F_standard(0;a), at the lower
  skew bound. F_standard(0;g) increases with signed skew by the monotonicity of Gamma survival
  at its own mean (positive g uses the lower Gamma CDF; negative g the upper).
  At exact equality the moment is finite. Binary64 requests within 1e-12 of the computed cutoff
  are rejected as unresolved. [Pinelis, Theorem 1, c=0](https://arxiv.org/abs/2012.13578).
- Every positive raw LP3 predictive moment diverges, including negative-skew intervals. A future
  standardized log flow has a uniformly positive probability of exceeding some positive c on
  a compact interior skew subinterval. Integrating exp(h c exp(eta)) gives the same divergence.
  Expected AEP at fixed discharge and an interior predictive quantile are different quantities.
  The report omits divergent raw design means while retaining defined estimands when representable.

Collapsed transformed shape, unresolvable Gaussian draws, zeroed positive weights and failed
interior likelihoods stop the fit. The innovation round trip uses a 1e-10 absolute/relative
resolution check. Conditional inverse queries check the smaller forward log tail within 1e-8;
a finite raw flow that fails this check cannot resolve the requested probability. These
floating-point checks do not measure integration precision.

## Fixed comparison protocol and independent results

The generated cases use the two tied exact years plus one left, one right and one interval-censored
year from the [estimated-scale example](wp06-estimated-scale.md). Shapes span (-.35,.35) for GEV
and (.55,1.25) or (-1.25,-.55) for LP3. Means are (3,0,0) or (0,0,0), and the factor rows are
(.25,0,0), (.08,.14,0), (.20,-.15,.60). All three coordinates are uncertain and correlated.
The acceptance protocol fixed 1,600 draws, seed 87, gamma=1 and a six-reference-standard-error
bound per output before the first comparison. The reference uses direct GEV formulas or
transformed Gamma density/tails, independently of the fitted kernels. Direct adaptive integration
of the Gamma density checks the reference special functions and its .99 inverse at six skews.

Full-domain Gaussian quadrature uses 32 then 56 nodes per axis (maximum 175,616 nodes per case).
The criteria are 3e-7 absolute on each reference and 1% relative on each asymptotic variance
coefficient. The coefficients integrate squared likelihood times squared centered influence,
including the influence of a population covariance. At gamma=1 the evidence uses Var(L).
The widest nodes reach 13.83 innovation SDs. For these examples M<1 and the two exact LP3 flows
exceed one, so L²<=exp(-4 eta). Completing the square shifts the Normal center by at most .56 per
axis. The envelope mass outside this cube is below 1.3e-39 before polynomial/design factors.
Those factors are bounded by polynomials times linear exponentials, absorbed by the Gaussian.
Grid refinement provides a separate check per integrand.

| Model | Evidence | E[mu] | E[eta] | E[shape] | E[AEP] | E[.99 design] |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GEV | .026623015 | 2.98624956 | -.05980953 | -.00027163 | .28947712 | 7.53357956 |
| LP3 positive | .003840262 | .08106847 | -.02988614 | .91674794 | .16638536 | 2.99379820 |
| LP3 negative | .008066159 | -.05259762 | -.07301742 | -.90215444 | .11747796 | 1.50603901 |

The final column is raw discharge for GEV and ln(q/r) for LP3. Covariance references in upper
triangle order (00,01,02,11,12,22) are

| Model | 00 | 01 | 02 | 11 | 12 | 22 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GEV | .05085661 | .01414917 | .00818767 | .02325203 | -.00092615 | .01104904 |
| LP3 positive | .04956673 | .01309221 | .00744723 | .02290325 | -.00133934 | .01083882 |
| LP3 negative | .04490682 | .01199323 | .00688114 | .02226765 | -.00135574 | .01079801 |

The largest reference change was 4.38e-10; the largest relative coefficient change was 9.76e-8.
The largest observed differences for the three cases were respectively 1.761, 1.826 and 1.638
reference standard errors. Run `python -m pytest tests/test_wp06_shape_reference.py tests/test_wp06_shape_contract.py`
to reproduce the comparisons, independent density checks and rejection cases. These results apply to
the stated cases. No controlled native comparison has been made.
