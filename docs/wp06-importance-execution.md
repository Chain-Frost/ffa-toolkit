# WP06 — Explicit multivariate Normal importance sampling (initial execution component)

**Status:** Mathematical particle-generation and self-normalized weighting implementation,
not native Flike compatibility, not a user-facing fit CLI, and not an automatic
posterior optimiser. See [development requirements](development.md) for the
remaining mathematical and software scope; native qualification is separate.

## Primary mathematical basis and admission limits

[Australian Rainfall and Runoff (2019), Book 3 v4.2, §2.6.3.6, equations
3.2.52–3.2.54](https://www.arr-software.org/pdfs/ARR_190514_Book3_V4.2.pdf)
describes locating the posterior mode, forming an approximate multivariate-Normal
density from the negative log-posterior Hessian, then generating proposal
particles and weighting them by the target/proposal density ratio. The
[Kuczera (1999)](https://doi.org/10.1029/1999WR900012) treatment addresses
the flood-frequency application, rating error and censored likelihood.
These sources motivate the mathematical approach but **do not** establish
the pinned Flike executable's exact optimizer, covariance regularization,
proposal, stopping, RNG or default controls.

This component implements **only the final draw-and-weight stage**, given
a **caller-supplied** Normal proposal in fully named source coordinates.
It does **not** fit the mode, numerically calculate a Hessian or select a
proposal from observations. The separate [local curvature builder](methods/importance-sampling-reference.md)
can construct the explicit proposal from a supported smooth target without
changing this sampler. Its SD multiplier gamma scales the full inverse-curvature
covariance by gamma squared (Kuczera 1999, p. 1554, equation 17); gamma=1
retains the unscaled result. The sampler receives the admitted final matrix
and derivation, never applies a second scale, and uses the same matrix factor
for draws and log densities. It does not infer a prior or a statistical
likelihood from a project file. This distinction is deliberate: a generally
correct importance sampler is useful without misrepresenting native defaults.

## Explicit execution contract

The internal `ffa_methods.importance_sampling` module requires:

1. `ExplicitNormalProposal`: unique source-coordinate names and order,
   finite mean vector, complete **symmetric positive-definite** covariance
   and the proposal's source/derivation description. The Cholesky
   decomposition retains correlations; a diagonal approximation is never
   silently substituted.
2. `ExplicitPosteriorTarget`: density functions and the exact same ordered
   coordinate names. The caller owns observational evidence, chosen
   parameters, priors, support and **all coordinate Jacobians**.
   A density outside model support may equal minus infinity; other invalid
   or nonfinite target evaluations fail closed.
3. `ExplicitSamplingControls`: **required** draw count, integer seed,
   minimum effective sample size and maximum single normalized weight.
   No statistically justified universal thresholds or native defaults
   are claimed by the implementation.
4. `RunProvenance`: true source/method/package/evidence identity. The module
   refuses to manufacture a `QUALIFIED_REFERENCE` provenance classification
   because it has not performed any native execution or qualification.

The binary64 storage constraint is explicit: any coordinate whose
**marginal proposal SD does not exceed one spacing at its supplied mean**
is rejected, rather than emitting a falsely degenerate posterior with zero
variance. This is a numerical representability criterion, **not** a vendor
proposal/default or a guarantee that marginal precision is otherwise adequate.

The sampler creates its own **local** `random.Random(seed)` instance and
generates independent `gauss(0,1)` components. A lower-triangular Cholesky
factor produces correlated multivariate Normal draws in the named coordinate
space. Actual *stored* vectors are used to evaluate their Normal proposal
log density; named likelihood and prior values are evaluated separately at
the same vectors. The method does not mutate the global random stream.
The seed, Python runtime and generator method are recorded as computational
provenance. Same-seed replay within the pinned environment is tested; RNG
identity is **not** asserted to match Flike or to be bitwise invariant over
all future Python implementations/versions.

For independent draws `theta_i ~ g`, define the unnormalised density
`t(theta)=L(D|theta) * prior(theta)` in exactly the same coordinates as `g`.
Weights are

`a_i = log L(D|theta_i) + log prior(theta_i) - log g(theta_i)`

and

`w_i = exp(a_i - max_j a_j) / sum_k exp(a_k - max_j a_j)`.

The existing reviewed `ImportanceParticles` object computes log-domain
normalization, the true weighted source-coordinate moments, posterior-weight
ESS, weight concentration and other distinct numerical estimands. A malformed
evaluation or failed explicit ESS/maximum-weight requirement aborts the run;
there is no hidden clipping, replacement or particle deletion. These are
*independent self-normalized* importance weights, not MCMC draws, equally
weighted posterior draws or deterministic quadrature nodes.

## Independent checks and numerical uncertainty

The independent synthetic [conjugate Normal test](../tests/test_wp06_importance_sampling.py)
uses observations 1, 2, 3 with known observation SD 2 and source mean prior
Normal(0,1). Exact posterior mean = 6/7 and variance = 4/7. A second
integration test evaluates the **real** WP03 positive-lognormal kernel,
WP05 exact-record likelihood, and WP05 supplied Gaussian source-coordinate
prior, with known posterior mean 3/2 and variance 1/4. Neither test
references candidate Flike run reports or substitutes deterministic
quadrature nodes for actual random draws. Separate independent numerical
checks cover the full Gaussian proposal log density/correlations, local-seed
replay and explicit failure modes.

For a scalar function `h(theta)` (here each source-coordinate mean),
the reported **approximate** self-normalized Monte Carlo standard error is

`sqrt(sum_i w_i^2 * (h_i - sum_j w_j h_j)^2)`.

The implementation uses a scaled Euclidean norm of weighted centered
residuals (and an overflow/underflow-safe hypotenuse for independent-replicate
standard errors), because directly squaring each component can falsely
underflow to zero even when the MCSE is representable.

This is an asymptotic plug-in diagnostic for independent proposal draws,
**not** a validated confidence bound when the proposal misses posterior
tails or the variance of the importance ratios is infinite. Replications
from distinct explicit seeds can be compared using the absolute
mean difference divided by the quadrature sum of their approximate
standard errors, but the code does not invent an acceptance critical
value. ESS or this global mean MCSE **cannot certify extreme flood
quantiles, posterior tail probability limits or stochastic convergence**.
Quantity-specific precision and workflow integration remain separate
development requirements.

The proposal is a Normal law with full real support. A posterior supported
on a strict subset may legitimately produce sampled minus-infinity log
density, with zero weight retained; all-zero weights are a hard failure.
If a Normal proposal fails to cover the target appropriately or cannot
represent its sampled values, the user must modify the **explicit
configuration** and independently test it, never silently replace a failed
run with a fabricated result.

## Scope ownership and next work

- **Posterior integration:** bind source-coordinate likelihoods and priors,
  including user-specified transformations and Jacobians, and assess
  output-specific numerical precision.
- **Installed workflows:** integrate admitted data and explicit algorithm
  settings into the CLI. The report bridge formats already calculated
  quantities; this sampler does not by itself enable Bayesian fitting.
- **Native qualification:** establish controlled executable, settings and
  matched-run evidence separately; do not claim native equivalence from
  independent calculations.

The repository retains **Sustainable Use License 1.0**. The implementation
copies no third-party proprietary/GPL/AGPL code, tests or numeric data.
