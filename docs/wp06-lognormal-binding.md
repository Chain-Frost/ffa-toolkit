# WP06 — Explicit lognormal posterior binding

Bounded [lognormal binding task](https://github.com/Chain-Frost/ffa-toolkit/issues/3)
within the [independent mathematical delivery](https://github.com/Chain-Frost/ffa-toolkit/issues/1). This is an internal
independent mathematical component, not an installed Bayesian fitting
command or a native Flike model/default mapping.

## Model and evidence contract

For a positive discharge Q and explicitly chosen reference discharge r
in the declared source units, Y = ln(Q/r) has Normal mean mu and standard
deviation sigma = exp(eta). The free coordinates are
`lognormal.mean_loge_flow` and `lognormal.loge_sd_loge_flow` in that order.
Either fit both with a supplied full Gaussian density in (mu,eta), or
explicitly fix eta and supply a one-coordinate Gaussian density for mu.
A fixed scale is a modelling assumption, not a prior on that scale or
an approximation to the unknown-scale posterior.

With f_Y, F_Y and S_Y denoting the Normal density, CDF and survival:
an exact observation contributes log f_Y(ln(Q/r)) - ln Q; the
left/right censor contribution is log F_Y(ln(upper/r)) or
log S_Y(ln(lower/r)); the interval contribution is the logarithm of
the corresponding CDF difference. The existing WP03 and WP05 components
perform these calculations, including stable probability differences.
The prior is already specified in the exact free parameter coordinates,
so there is no extra parameter Jacobian. A raw-scale prior is not
silently converted. Scaling Q and r by the same positive factor leaves
Y unchanged and adds a constant to each exact-flow log density; the
posterior shape and dimensionless probabilities are unchanged.

Basis: ARR Book 3 v4.2, Chapter 2 §2.4.2 (distribution choice),
§2.6.3.1–2.6.3.5 (observation likelihood and explicit prior),
§2.6.3.6 equations 3.2.52–3.2.54 (proposal and importance weights),
§2.6.3.7 (weighted outputs); see [reference register](references.md),
[likelihood explanation](methods/censored-historical-likelihood.md),
and [importance method](methods/importance-sampling-reference.md).
The normal-density example below is an original independent derivation.
Published mathematical formulas support this explicit model; unknown native details remain unsupported.

`ClassifiedLognormalPosterior` consumes an explicitly classified
`ObservationSeries`, source label and SHA-256, year definition, discharge
unit, reference discharge, fixed-log-SD or estimation choice, and
`JointGaussianPrior` with supplied mode and exact coordinate order.
The explicit classification string is
`caller_asserted_classified_systematic_annual_maxima`. A direct typed
series has caller-asserted source identity; a supplied digest is not
independent custody proof. `bind_exact_lognormal_csv` reads actual
Year,AMS bytes through WP02 and computes their SHA-256. It refuses
zero, missing or nonrepresentable flows instead of discarding rows.
Year definitions and observation classification remain caller assertions.

The typed boundary admits systematic unique-year positive exact and
positive one-sided/interval-censored records. Explicit missing/excluded
records are retained and have no likelihood contribution. At least one
included record is required. Native flags, historical/grouped exposure,
zero-threshold law, rating-error law, unidentified priors and unresolved
records are rejected. No inference from native file flags, fabricated
historical periods or automatic MGBT censoring occurs here.

## Execution and original worked example

`execute_lognormal_posterior` uses the target's separate likelihood and
prior callbacks and the existing reviewed local proposal builder and independent
sampler. The caller must supply starting coordinates, curvature settings,
seed, particle count and weight-adequacy thresholds. The same bound
`kernel` factory is supplied to the existing weighted-output functions.
The result retains the model, source, prior, full proposal diagnostics,
actual particles, numerical uncertainty and provenance. No new sampler,
optimizer or fit CLI is introduced.

Take three synthetic positive annual discharges exp(1), exp(2), exp(3)
in the declared unit, with reference discharge 1, known log-SD eta=0
and supplied prior mu ~ N(0,1). Completing the square gives posterior
precision 1+3=4 and linear coefficient 1+2+3=6; hence posterior
mu ~ N(1.5,0.25). The predictive log discharge is N(1.5,1.25).
For fixed Q=exp(2), the exact expected AEP is
1-Phi((2-1.5)/sqrt(1.25)), where Phi is the standard Normal CDF.
The independent test uses actual Year,AMS CSV bytes, constructs the
proposal, generates 800 particles with seed 51, and checks posterior
moments and expected AEP against these analytic results. The mean
and variance tolerances are five Normal sampling standard errors;
the probability tolerance is five worst-case bounded-integrand
standard errors in this exact-Normal proposal test. These tolerances
are not generic flood-quantile or native acceptance rules.

For the free two-parameter case, prior and log-density/censor terms
are checked against independently derived Normal and joint Gaussian
formulas. The local Newton solver may reject nonconcave basins,
unresolved curvature, unrepresentable densities or failed stationarity.
Passing local optimization and global weight/mean diagnostics cannot
certify global coverage, numerical tail precision or the existence of
every posterior flood moment. The conditional posterior/flood
uncertainty is distinct from Monte Carlo error.

Remaining: other families, source-preserving historical/grouped and
broader WP07 integration, explicit tail precision and installed
Bayesian settings/CLI/reporting remain tracked in Toolkit issues [#5](https://github.com/Chain-Frost/ffa-toolkit/issues/5),
[#6](https://github.com/Chain-Frost/ffa-toolkit/issues/6), [#8](https://github.com/Chain-Frost/ffa-toolkit/issues/8),
[#9](https://github.com/Chain-Frost/ffa-toolkit/issues/9) and [#10](https://github.com/Chain-Frost/ffa-toolkit/issues/10).
Native defaults and comparison acceptance remain separate external qualification work. Sustainable Use License 1.0
and the existing mathematical kernels and dependencies are unchanged.
