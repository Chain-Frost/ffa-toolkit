# WP07 / issue #32 — USGS MGBT 1.0.7 conditional rank-p-value specification

**Status: independently reviewed mathematical specification, followed by a bounded Python mathematical comparator. Not a Flike-native fitting or inference mode.**

This specification is only for a **source-identified external comparator**, the 21 July 2021
CRAN MGBT **1.0.7** `MGBT17c` function. It is **not** an identification of
TUFLOW/BMT Flike 5.0.306.0's algorithm, defaults, native reference outputs,
likelihood or fit mode. See [development requirements](https://github.com/Chain-Frost/ffa-toolkit/blob/main/docs/development.md)
and the [variant comparison](wp07-mgb-variant-comparison.md).

## Exact source and licences

- [CRAN mirror commit `aa4356208f05d6036f3f425ec37c656da345bba0`](https://github.com/cran/MGBT/tree/aa4356208f05d6036f3f425ec37c656da345bba0):
  `DESCRIPTION` declares version **1.0.7**, copyright and **CC0**.
  `R/MGBT.R` Git blob SHA-1 `0885ad5dce4a12802652f0274526c40336c79753`;
  `R/RthOrderPValueOrthoT.R` `a739ec5962d7dc059d32dc316dfc3725db364c62`;
  `R/peta.R` `d881373d3b7d9c48c510ea08efc99bf399d3934b`;
  `R/EMS.R` `5476b7d71af10994afa36aafc259517d6f3b56de`;
  `R/CondMoms.R` `95a34b1a7b7ecdb606f75cd1d6b382f9007c5c7f`;
  `R/gtmoms.R` `e8b42f2b4ca166265c9a590e6cc16c0af2082def`;
  `man/MGBT.Rd` `b12b8ceafd967f71f90ed07b570e1b2418f0d6c1`.
  These are **Git blob SHA-1**, not archive SHA-256; no source copy is imported.
- Mathematical basis: [Cohn et al. (2013)](https://doi.org/10.1002/wrcr.20392),
  with source-specific approximations as implemented by the pinned R functions.
  [Bulletin 17C](https://doi.org/10.3133/tm4B5) provides an independent US
  frequency-analysis context, **not** the legacy Flike Bayesian likelihood.
- Comparison-only source alternatives:
  [USACE Numerics `7e8e8d1`](https://github.com/USACE-RMC/Numerics/blob/7e8e8d1c5f26e045a35ec9fc09367de95ed05b02/Numerics/Data/Statistics/MultipleGrubbsBeckTest.cs)
  (0BSD), [FlowFreq `7fe7cac`](https://github.com/pinhead001/hydrolib/blob/7fe7cac1f88998253c81a7793a04c677c34b5b88/flowfreq/bulletin17c.py)
  (MIT and separately noticed PeakFQ). Preserve the repository **SUL-1.0** licence,
  without GPL/AGPL code or tests.

## Observed statistic and p-value integral

For recorded **nonnegative finite** flows `Q` in original units, sort ascending
without losing source-position mapping. For the source variant compute
`z_i=log10(max(Q_(i),sqrt(eps_binary64)))`, where binary64 machine epsilon
is `2**-52` and the temporary internal floor `sqrt(eps)` is `2**-26`.
The original zero and any other original flow remain unchanged and distinguishable.
The default candidate count is `n2=floor(n/2)` (other explicit candidate counts
require a separate source-backed contract).

For `1 <= r <= n2`, let `k=n-r`. Compute the **sample** SD of the upper
`z_(r+1),...,z_n`, with divisor `k-1`:

```text
eta_r = (z_r - mean(z_(r+1)..z_n)) / SD_sample(z_(r+1)..z_n)
p_r   = integral from a to (1-a) of h(u; n,r,eta_r) du
a     = sqrt(eps_binary64)
```

This `p_r` is the **one-sided significance probability for a low-outlier test**.
It is *not* flood non-exceedance `F(Q)`, an annual exceedance probability or
the LP3/LN `P(Q <= q0)` threshold parameter. The implementation must use the
existing WP07 `W(r)` calculation and source-identified three-sweep decision
rather than reinterpreting raw flags or recomputing a different metric.

For a quadrature node `u in (a,1-a)`, define

```text
b = Beta(r, n+1-r).PPF(u)             z = Normal.PPF(b)
H = Normal.PDF(z) / Normal.SF(z)     (stable log-PDF/log-SF evaluation)
m1 = H                               m2 = 1+z*H
m3 = 2*m1 + z*z*H                    m4 = 3*m2 + z*z*z*H
v  = m2 - m1*m1
c3 = m3 - 3*m2*m1 + 2*m1**3
c4 = m4 - 4*m3*m1 + 6*m2*m1**2 - 3*m1**4
VM = v/k
VS2 = (c4-v*v)/k + 2*v*v/(k*(k-1))
alpha = v*v/VS2                        beta_scale = VS2/v
ES = sqrt(beta_scale) * Gamma(alpha+0.5)/Gamma(alpha)
CMS2 = c3/sqrt(k*(k-1))              CMS = CMS2/(2*ES)
VS = v - ES**2                        lam = CMS/VS
muM = m1 - lam*ES                     VMP = VM - CMS**2/VS
df = 2*alpha                          ncp = (muM-z)/sqrt(VMP)
q  = -sqrt(v/VMP)*(eta_r+lam)
h(u;n,r,eta_r) = NoncentralT.SF(q; df,ncp)
```

The `ES` ratio must be evaluated with `gammaln`. The source uses
`1-pt(q,...)`; stable noncentral-t survival probability avoids catastrophic
cancellation when `pt` rounds to one. **Numerical differences from the
original R integration are expected** and must be measured, not concealed.
The source R code also returns `1` for an invalid `VMP` in one path and
optionally substitutes zero for an NA p-value; **this proposed Python comparator
will fail closed instead of silently classifying such values**.

## Limits, integration and independent verification gate

Proposed initial numerical contract, subject to independent review:

- Explicitly label the source version. Only integer `n >= 5` and ranks
  `1 <= r <= floor(n/2)` are admitted by the *existing* WP07 comparator
  interface; **this is an adapter boundary**, not an assertion that the source
  published a five-observation mathematical minimum. At least two upper sample
  observations are required. Do not carry over Numerics/BestFit's `n >= 10`
  setting as a universal rule.
- Reject non-finite or undefined `eta`, nonpositive `v`, `VS2`, `VS`,
  `VMP`, `df`, or invalid `b, z, q, ncp`. An undefined upper-sample SD
  is an error, not a significant low outlier or a zero-probability event.
- Use deterministic SciPy adaptive quadrature with initial
  `epsabs=1e-8`, `epsrel=1e-8`, bounded subdivisions and the **same
  R 1.0.7 integration interval** `[sqrt(eps),1-sqrt(eps)]`. Treat
  integration warnings, non-finite values, estimated error above the agreed
  budget, or out-of-[0,1] results as **explicit failures**. Recheck a
  representative subset with tighter quadrature settings or an independent
  quadrature rule; no stochastic fallback is part of this comparator.
- Preserve the pinned USGS **08066300** sample of 51 raw flows and its
  **25 printed p-values**, eight reported decimals, in the existing test.
  A read-only local mathematical prototype of the conditional integral
  differed from those source literals by up to **1.57e-7 absolute** using
  SciPy quadrature; the R default integration tolerance and R noncentral-t
  implementation mean eight *printed* decimal places are **not proof of
  5e-9 computational accuracy**. Before accepting, repeat with an
  independently integrated subset, document exact run/library versions, and
  justify an absolute comparison limit on the order of `2e-6` for **these
  printed source outputs only**. Tightened independent quadrature precision
  must be assessed separately.
- Explicitly check monotonicity in `eta` at fixed `n,r`, deterministic
  reproducibility, original-flow permutation invariance, edge values and
  failure paths. The USGS 08066300 selection (one flow **55 cfs**, next
  retained **284 cfs**) must remain a **reference comparator result**, not
  a native Flike classification.
- No zero-threshold LN/LP3 posterior, empirical `p0`, Beta prior, positive
  offset, changed source flags, guessed dummy-year chronology or new release
  command may be introduced. A finished numerical **component** does not
  close native WP07 #10.

## Implementation ownership and review sequence

1. Obtain **independent Codex/statistical review of this specification**.
   Record the exact reviewed revision and actionable findings.
2. Only afterward implement the bounded conditional p-value evaluator and
   source-labelled composition in WP07-owned files, retaining all existing
   screening APIs and source records. Avoid touching WP02/WP05 contracts.
3. Test against pinned source p-values and independent numerical checks.
   Pass full Python 3.14 Ubuntu/Windows CI and exact-final-head Codex review;
   preserve appropriate method-register qualification boundaries. Link
   `Refs #32`, `Refs #10` and `Refs #2` until this child actually meets
   its own acceptance criteria; keep #10 open in all cases.

**Review receipt and implementation boundary:** this specification at commit
`ef129ff5ddf009623e8f0ba6301ea9bf53a9318b` received an
[development requirements](https://github.com/Chain-Frost/ffa-toolkit/blob/main/docs/development.md)
**before implementation**. The subsequent separate `mgbt_pvalues.py`
component implements the identified USGS 1.0.7 conditional integral using
the existing SciPy 1.17 dependency, stable noncentral-t survival probabilities
and deterministic quadrature. It fails closed on under-resolved or saturated
probabilities; it never silently accepts the pinned R implementation's
stochastic integration fallback or p=0/1 coercion.

Its regression includes all 25 published rounded rank p-values for USGS
08066300, an independent 1024-node Gauss–Legendre check of an
endpoint-sensitive example (whose fixed-node convergence is distinct from
QUADPACK's adaptive error estimate), monotonicity, original-record order,
zeros, source-specific significance sweeps and failure boundaries.
**Only the final functional commit's Python 3.14 CI and independent
code/statistical review can complete issue #32.** These tests do not verify
legacy Flike 5.0.306.0 defaults or native classification.
The source's printed p-values and the Python quadrature's error estimates
have different uncertainties and must not be confused.

The historical design requirements above remain the reference for reviewing
this isolated implementation; this document does not activate a CLI fit.
