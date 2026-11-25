# WP03 â€” Five continuous distribution kernels

**Status: WP03 continuous mathematical kernels and report-internal checks implemented, independently reviewed and merged; native Flike equivalence not qualified.** Mathematical implementation: [development requirements](https://github.com/Chain-Frost/ffa-toolkit/blob/main/docs/development.md). Deferred vendor rules, controlled runs and comparison: [development requirements](https://github.com/Chain-Frost/ffa-toolkit/blob/main/docs/development.md). Final compatibility/release gate: [development requirements](https://github.com/Chain-Frost/ffa-toolkit/blob/main/docs/development.md). This is an internal, scalar probability module for the **CLI-only** MVP, not a public API, a fitting procedure or an observation/likelihood model. Original base: `e48c58154a47a60692389ca24cf7a0506e3598f9`.

## Common contract

`src/ffa_methods/distributions.py` implements immutable `Lognormal`, `LogPearsonIII`, `Gumbel`, `GEV` and `GeneralizedPareto` objects with `support()`, `logpdf(q)`, `logcdf(q)`, `logsf(q)`, `cdf(q)`, `sf(q)`, `ppf(F)` and `isf(S)`. Inputs q are in declared discharge units. Probabilities are fractions; `ppf` takes non-exceedance F, `isf` takes exceedance S. Annual-maximum AEP p maps to `isf(p)` only for an applicable annual-maximum model. The GP kernel does **not** model POT event rates, annual exposure or event extraction.

Invalid parameters and NaN inputs raise `ValueError`. Impossible exact observations have `logpdf=-inf`. Predictions below or above a valid component's support remain mathematically valid with F=0/1 and S=1/0, even when averaging probabilities over posterior components. No GEV/Gumbel distribution is silently truncated at zero.

SciPy primitives implement Gumbel, GEV, GP and normal. LP3 uses a gamma-transform Pearson III representation with log-domain incomplete-gamma series/continued-fraction fallback when SciPy's log-tail underflows, and with overflow-safe gamma coordinate arithmetic: the default SciPy Pearson III inverse survival lost precision in rare tails for some skew values during local tests. The *lower* series now has a decreasing-term remainder bound and an adaptive, finite iteration budget; near its large-shape mode the logarithmic prefactor uses the Stirling/deviance form to avoid subtracting enormous nearly equal terms. The exact zero-skew limit is normal; for |skew| below 1.6e-5 the numerical implementation uses a normal approximation to avoid catastrophic cancellation in the gamma density (a numerical rule, **not** a claimed legacy inference rule). This approximation is accurate at regular quantiles but **not** uniformly accurate in extremely deep log tails: immediately below the switch, the normal and exact-gamma log CDF at ln(q)=-40 can differ by about 0.17. Rare upper tails must use inverse survival directly; `ppf(1-p)` is not reliable when 1-p rounds to one. Finite endpoints and logarithmic tails beyond floating-point representability remain limited by binary64.

## Coordinates, support and sign conventions

| Family | Coordinates | Support and SciPy mapping |
| --- | --- | --- |
| LN | `LogCoordinates(mean, log_sd, base, reference_discharge)` | q>0, normal law of ln(q/ref) with discharge Jacobian |
| LP3 | Same log coordinates and log-space `skew` | q>0 with skew-dependent bounded endpoint; gamma/normal primitives |
| Gumbel | `location`, `log_scale=ln(scale)` | Full real line; SciPy `gumbel_r` |
| GEV | `location`, `log_scale`, EVT `xi` | xi>0 finite **lower** endpoint; xi<0 finite **upper** endpoint; SciPy `genextreme(c=-xi)` |
| GP | `location`, `log_scale`, EVT `xi` | q>=location; xi<0 finite **upper** endpoint; SciPy `genpareto(c=xi)` |

LogCoordinates defines Y = log_base(Q/reference), and `log_sd=ln(sd(Y))`: the **outer** logarithm is natural for any inner base. For b>1, `mu_ln = mu_b * ln(b)`; `eta_ln = eta_b + ln(ln(b))`; skew is invariant. The joint parameter coordinate map has Jacobian determinant ln(b), so log absolute determinant `ln(ln(b))`; converting sd=exp(eta) has log Jacobian eta. These formulas protect source-coordinate prior densities but **do not define a new prior**. For an exact q>0, `f_Q(q)=f_Y(ln(q/reference))/q` with a natural-log internal Y; using base b directly gives denominator `q*ln(b)`.

`with_reference(new_ref)` adds log_base(old_ref/new_ref) to the source mean without changing skew/SD; `in_rescaled_discharge_units(factor)` changes both Q and reference by a positive factor and rescales density by 1/factor. The discharge reference must be tied to physical units; it is not a fitting offset.

## Analytic and primary references

- **LN:** `F(q)=Phi((ln(q/ref)-mu)/sd)`, `f(q)=phi(z)/(q*sd)`. [SciPy lognormal](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.lognorm.html).
- **LP3:** for nonzero skew g, `a=4/gÂ²`, `edge=mu-2*sd/g`, `t=sd*|g|/2`, and `Y=edge+sign(g)*t*Gamma(a,1)`. g=0 gives normal. [SciPy Pearson III](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearson3.html), [SciPy gamma](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gamma.html), [Kuczera 1999](https://doi.org/10.1029/1999WR900012).
- **Gumbel:** `F(q)=exp[-exp(-(q-mu)/sd)]` and `Q(F)=mu-sd*ln[-ln(F)]`. [SciPy Gumbel](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gumbel_r.html).
- **GEV:** `F(q)=exp[-(1+xi*(q-mu)/sd)^(-1/xi)]` on `1+xi*(q-mu)/sd>0`. Xi=0 is the Gumbel limit. [SciPy GEV](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.genextreme.html); [HEC-SSP shape convention discussion](https://www.hec.usace.army.mil/confluence/sspdocs/sspum/2.3.1/distribution-fitting-analysis/distribution-fitting-and-parameter-estimation).
- **GP:** `S(q)=[1+xi*(q-location)/sd]^(-1/xi)`, with exponential limit at xi=0. [SciPy GP](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.genpareto.html).

## LP3 rare-tail numerical justification and independent checks

For gamma shape `a=4/gÂ²` and argument `x>0`, the fallback computes `log P(a,x)` or `log Q(a,x)`, the regularized lower/upper incomplete gamma tails. The lower-series correction is

`S = 1 + Î£[nâ‰¥1] x^n / [(a+1)(a+2)â€¦(a+n)]`,

and `log P(a,x) = a log(x) - x - log Î“(a) + log(S) - log(a)`. For `x<a+1`, the successive positive term ratios decrease: `r_n=x/(a+n+1)<1`. After adding term `t_n`, the omitted sum is bounded by `t_n*r_n/(1-r_n)`. The initial ratio `r_0` supplies a conservative finite iteration limit from `r_0^(N+1)/(1-r_0) <= 1e-14`, rather than the former fixed 10,000 terms. In a tail underflow case, `x<a+1` is the applicable lower-gamma region; the opposite tail uses the existing upper continued fraction. If the bound cannot be satisfied, an explicit arithmetic error remains preferable to a silently inaccurate answer.

When `a>=10,000` and `x` is within 10% of `a`, the log prefactor is evaluated using `u=(x-a)/a` and

`a*[log1p(u)-u] + 0.5*log(a/(2Ï€)) - 1/(12a) + 1/(360aÂ³)`.

For `|u|<0.01`, the bracket is evaluated with its alternating Taylor expansion. This avoids cancellation in `a*log(x)-x-lgamma(a)` at gamma shapes around `4e8` to `1.5625e10`. The neglected Stirling term is of order `a^-5` and is negligible at the admitted large-shape threshold. None of these numerical rules selects an LP3 prior or establishes any Flike implementation detail.

**Finite but inaccurate SciPy lower tails.** Direct 80-digit checks also exposed finite erroneous SciPy `gamma.logcdf` values for large shapes before numerical underflow. At `g=0.0001`, `ln(Q)=-37`, SciPy 1.17.0 gives approximately -689.900681 while the independent value is -689.87597398. At `g=0.00002`, `ln(Q)=-5`, SciPy's log CDF differs by about **2.315**. For `a>=100,000` and `0<x<a-4.4*sqrt(a)`, the LP3 lower log tail therefore uses a **16-node Gaussâ€“Laguerre quadrature** of the same independently derived endpoint integral below, rather than trusting a finite but inaccurate SciPy result. This endpoint integrand is smooth and nearly constant under the exponential weight for the selected standardized distance; the 16 nodes/weights are the classical `L_16` rule (generated using SciPy 1.17.0, not copied implementation code). The shape/domain guard retains SciPy for the ordinary central range. The same corrected lower-tail log value is exponentiated for the probability-space `cdf` / `sf` methods when representable; their previous direct SciPy values had the same error. The large-shape log density likewise uses `(a log(x)-x-loggamma(a))-log(x)-log(gamma_scale)` with the stable prefactor, because direct SciPy `gamma.logpdf` loses up to about 1e-5 in the checked large-shape cases. Tests against 80-digit direct mpmath integration at z=-4.5, -5 and -10 for three small skews, plus very deep tails, pin the accuracy to **2e-9 absolute log probability**. The geometric-bound series remains available for gamma lower tails outside this accelerated regime. As with all finite-order quadrature, this is a verified numerical rule for the tested region, not a general exact symbolic integral.

**Independent oracle.** Regression values were calculated with Python 3.13.5, mpmath 1.3.0, **80 decimal digits**, without evaluating SciPy, NumPy, or the FFA kernel. For `g>0`, `mean(log Q)=0`, `log SD(log Q)=0`, and `Q=exp(-40)` or `exp(40)`, use `a=4/gÂ²` and `x=aÂ±80/g`. Set `d=|x-a|` and `B=a*log(x)-x-loggamma(a)`. The endpoint integral for the lower tail (`x<a`) is

`P(a,x) = exp(B)/d * integral[0,âˆž] exp(-v - x*(exp(-v/d)-1+v/d)) dv`,

and the upper tail (`x>a`) uses

`Q(a,x) = exp(B)/d * integral[0,âˆž] exp(-v - x*(exp(v/d)-1-v/d)) dv`.

Direct high-precision quadrature over `[0,1,5,20,100,250]` computes both without gamma special functions; the final omitted positive-integral contribution is below the retained decimal precision in the tested cases. The negative-skew counterparts follow by reflection of the log variable. Recorded log-tail targets:

| Absolute skew | Positive-skew lower log CDF (negative-skew upper log SF) | Positive-skew upper log SF (negative-skew lower log CDF) |
| ---: | ---: | ---: |
| 0.01 | -930.3503719533658756 | -711.7462970045258524 |
| 0.001 | -815.4376996432623760 | -794.0992697843885181 |
| 0.0003 | -807.8229077423288419 | -801.4227769801055585 |
| 0.0001 | -805.6767099980118645 | -803.5433740384410887 |
| 0.00002 | -804.8218391182044722 | -804.3951729093332381 |
| 0.000016 | -804.7791494514121107 | -804.4378164961116057 |

The absolute test tolerance **2e-9 in log probability** includes binary64 construction of `a`, `x` and `exp(Â±40)`; it is not a tolerance on ordinary probabilities or a flood peak. Additional `g=0.0001` checks bracket SciPy's underflow boundary: at ln(Q)=-37 the expected log CDF is -689.8759739832407111, and at -39 it is -766.0732534457897997.

**Independent published analytic law.** The 0BSD [USACE-RMC/Numerics v2.2.0 source](https://github.com/USACE-RMC/Numerics/blob/7e8e8d1c5f26e045a35ec9fc09367de95ed05b02/Test_Numerics/Distributions/Univariate/Test_NormalPearsonRobustness.cs) and the [development requirements](https://github.com/Chain-Frost/ffa-toolkit/blob/main/docs/development.md) give `mean(log Q)=0`, `SD(log Q)=1`, **`log SD=0`**, skew -2. If `E~Exponential(1)`, `log Q=1-E` and `Q~Uniform(0,e)`. Interior `logpdf(1)=-1` (upstream absolute tolerance 1e-14), and `ppf(1e-20)=e*1e-20` (original absolute tolerance **2e-31**). This adopts the analytic identities directly in the WP03 tests, without duplicating or moving the PR #22 fixture while that evidence PR remains independent. It does not copy endpoint-density conventions or add unsupported moment APIs.

## LP3 report-output mapping: mathematical interpretation only

For a report labelled "expected parameter quantile", the mathematical LP3
interpretation of mean natural-log flow, natural log of the log-flow standard
deviation and log-flow skew is
`LogPearsonIII(LogCoordinates(mu, ln_sd), skew).isf(1/T)` at return period
`T > 1`, **provided** the reported curve is an unconditional positive-flow
continuous law.

Where the legacy report *also* identifies a fourth coordinate as the logit of
the probability at or below a declared threshold, the **above-threshold**
conditional identity is
`LP3.isf((1/T)/(1-logistic(mean_logit)))`, for a positive quantile and a
valid conditional exceedance probability. A reported most-probable threshold
probability is a different statistic from the posterior mean *logit*.
This algebra alone does **not** establish a new zero-flow atom, likelihood,
prior, posterior estimator, or legacy default; those are WP07/WP05/WP06
responsibilities.

A separate [development requirements](https://github.com/Chain-Frost/ffa-toolkit/blob/main/docs/development.md)
records candidate identifiers, numerical observations, source hashes and
rounding tests. The source candidates remain **unverified project/run/report
pairings**, so that evidence neither qualifies Flike 5.0.306.0 compatibility
nor resolves GEV `k` sign and LN/Gumbel/GP coordinate conventions.

## Legacy compatibility admission: incomplete

| Legacy source | Mapping status | Remaining evidence |
| --- | --- | --- |
| LP3: `Mean(loge flow)`, `loge[Std dev(loge flow)]`, `Skew(loge flow)` | Non-distributed report-only evidence supports the expected-parameter quantile interpretation, including an above-threshold logit conversion where explicitly reported | Verified project/run/report custody, exact source controls and independent native rerun; zero-threshold likelihood belongs to WP07 |
| GEV: location `u` and `loge(scale a)` | Labels **observed**; match location and log scale if discharge units agree | Shape `k` sign **not proven** to be xi or -xi; must not guess |
| LN, Gumbel and GP full legacy parameter order, optional location/threshold and estimator combinations | **Unverified** in current archive | Target-version reference cases and settings for each family |
| Priors, zero threshold, L/LH fitting, POT exposure | Not provided by a mathematical kernel | WP04â€“WP07 evidence and integration |

See [compatibility contract](compatibility-contract.md), [sample-data audit](sample-data-audit.md), [method specification](statistical-methods.md) and [ARR Book 3 Â§2.4.2](https://www.arr-software.org/pdfs/ARR_190514_Book3_V4.2.pdf). This work makes no unverified legacy numeric claim.

## Checks

From the configured repository development environment:

```bash
PYTHONPATH=src python -m pytest -q tests/test_distributions.py
python -m ruff check src/ffa_methods/distributions.py tests/test_distributions.py
python -m ruff format --check src/ffa_methods/distributions.py tests/test_distributions.py
python -m pyright
```

Independent tests cover analytic normal/gamma/Gumbel/GEV/GP formulas; both shape/skew signs; zero limits; endpoints/outside-support predictions; rare 1e-25 survival inversion; reversible base/reference/unit conversions; discharge and parameter Jacobians; independent 80-digit LP3 log-tail quadrature, the finite-but-inaccurate SciPy-to-quadrature, underflow-to-log-fallback and normal-to-gamma transitions; the pinned analytic LP3 Uniform(0,e) law; and error paths. The mathematical kernels were independently reviewed and merged in [development requirements](https://github.com/Chain-Frost/ffa-toolkit/blob/main/docs/development.md). The **report-internal-only** LP3 checks, source-custody verification and exclusion of private values from distributable guidance were independently reviewed and merged in [development requirements](https://github.com/Chain-Frost/ffa-toolkit/blob/main/docs/development.md). The [development requirements](https://github.com/Chain-Frost/ffa-toolkit/blob/main/docs/development.md) passed Ruff lint/format, strict Pyright, **400 tests on both Ubuntu and Windows**, build, distribution/resource checks, installed-wheel CLI smoke and whitespace. These results support **the implemented mathematical/software scope of issue #6 only**. Issue #6 may close once this implementation completion record is accepted; it does **not** establish Flike native qualification. Vendor-specific LN/Gumbel/GEV/GP conventions (including GEV `k` sign), native executable/settings custody, matched reference cases and numerical comparisons are **deferred to [development requirements](https://github.com/Chain-Frost/ffa-toolkit/blob/main/docs/development.md)**. The [development requirements](https://github.com/Chain-Frost/ffa-toolkit/blob/main/docs/development.md), [development requirements](https://github.com/Chain-Frost/ffa-toolkit/blob/main/docs/development.md) and overall [development requirements](https://github.com/Chain-Frost/ffa-toolkit/blob/main/docs/development.md) have separate remaining criteria. Do not infer vendor defaults from these mathematical or report-internal checks.
