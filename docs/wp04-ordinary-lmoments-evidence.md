# WP04: independent H=0 ordinary L-moment family inversions

**Status: mathematical foundations only. These functions are not qualified against TUFLOW/BMT Flike 5.0.306.0.** The existing [fixed-H GEV evidence](wp04-fixed-h-gev-evidence.md) remains authoritative for GEV. [development requirements](https://github.com/Chain-Frost/ffa-toolkit/blob/main/docs/development.md) remains open.

## Inputs, parameter conventions and formulae

The new internal estimator module, src/ffa_methods/estimators/ordinary_lmoments.py, reuses the unbiased H=0 sorted-order-statistic moment estimator from WP04. Inputs are complete, finite, equally weighted exact annual maxima. No missing, excluded, threshold, censor, historical, prior, rating-error or POT records are admitted or automatically dropped. The caller selects an explicit distribution and records whether the first moments are of discharge or log discharge. The quantile argument is non-exceedance probability F, with AEP=1-F; requested outputs must be representably finite.

| Family | Moment source | Inversion | Numerical boundary |
| --- | --- | --- | --- |
| Gumbel (two-parameter) | Raw Q: L1,L2 | scale=L2/ln(2); location=L1-EulerGamma*scale | Positive L2; real support |
| Lognormal LN2 (zero location) | Raw Q: L1,L2 | sd_ln=2 erfinv(L2/L1); mean_ln=ln(L1/reference)-sd_ln²/2 | Q>0, reference>0, 0<L2/L1<1 |
| Log Pearson III LP3 | ln(Q/reference): L1,L2,L3 | tau3=sign(g)*(6 I_(1/3)(a,2a)-3), a=4/g²; solve exact beta-function root. Gamma scale beta=L2 sqrt(pi) Gamma(a)/Gamma(a+1/2); sd_log=beta sqrt(a); mean_log=L1 | Q>0; gamma a in [0.01,1e8]; supported endpoint must contain all observations; tau3=0 has exact Normal limit |
| Generalized Pareto GP, known location | Raw Q: L1,L2 | kappa=(L1-location)/L2-2; scale=(1+kappa)(L1-location) | Explicit fixed location; support and positive scale |
| GP, free location | Raw Q: L1,L2,L3 | kappa=(1-3tau3)/(1+tau3); scale=L2(1+kappa)(2+kappa); location=L1-(2+kappa)L2 | -1<tau3<1 and sample inside fitted support |

For GP, Hosking/ARR kappa = -EVT xi, while the WP03 GeneralizedPareto kernel uses SciPy genpareto shape xi directly. For GEV, ARR kappa also equals -EVT xi, but the *Flike report label k* is not yet verified to use either sign. The LN and LP3 source kernels use WP03 LogCoordinates with the mean of natural-log flow and the *natural logarithm of its log-space standard deviation*. A reference discharge rescales the log-coordinate mean without changing the underlying physical law. LN fitting uses L-moments of raw discharge; LP3 fitting uses L-moments of ln(Q/reference). Interchanging these sampling coordinates changes the method.

The LP3 exact gamma-shape inversion uses a bounded numeric root, NOT the Hosking rational approximation. The shape bracket is an explicit **numerical admission limitation, not a Flike bound**. Nonzero L-skew outside the bracket fails rather than silently fitting a normal distribution or clamping the parameter. This branch leaves high-precision verification of the extreme boundaries and the legacy algorithm for a separate evidence task. Nonpositive L2, nonrepresentable output, invalid support, and insufficient samples are rejected explicitly.

## Independent tests and provenance

Population mid-quantile grids independently reproduce Gumbel, raw-L-moment LN2, reflected Pearson III and GP families, including zero and opposite skew/shape signs. A deterministic grid is quadrature, not a random Monte Carlo result.

- Exactly spaced [1,2,3,4,5] produces Gumbel L1=3 and L2=1.
- The independent normal-integral identity for LN2 is L1=reference exp(mu+sd²/2), L2=L1 erf(sd/2).
- Independent gamma-quantile samples test LP3 against the regularized-incomplete-beta law, both reflection signs, and the Normal skew-zero limit.
- For GP, [1,2,3] has L1=2, L2=2/3, tau3=0; the fitted law is Uniform(0,4) with GP kappa=1 and EVT xi=-1, for either fixed or free location.
- Exact probabilities, affine/unit transformations, representable outputs and explicit unsupported input/shape/fitting cases are regression-tested.

These are independent mathematics, not native Flike benchmarks. Tolerances reflect deterministic population-grid discretisation and binary64 arithmetic, never a generic flood-discharge percentage.

## Sources, reuse and outstanding legacy work

- [ARR 2019 Book 3 v4.2, section 2.6.4 and Table 3.2.3](https://www.arr-software.org/pdfs/ARR_190514_Book3_V4.2.pdf).
- [Hosking 1990, L-moments theory](https://doi.org/10.1111/j.2517-6161.1990.tb01775.x).
- [Hosking and Wallis 1987 GP identities](https://search.r-project.org/CRAN/refmans/nsRFA/html/GENPAR.html).
- [Hosking and Wallis 1997, Pearson III L-moment formulas](https://search.r-project.org/CRAN/refmans/nsRFA/html/P3.html).
- [NIST lognormal density/parameter convention](https://www.itl.nist.gov/div898/handbook/eda/section3/eda3669.htm).

No R/GPL/AGPL implementation or tests have been copied or translated. The code is original and the project retains Sustainable Use License 1.0 / LicenseRef-SUL-1.0. Unqualified public mathematical references do not add a code or data dependency.

ARR section 2.8.10.3 states that H=0 is available for all five families, with positive H reserved for GEV. It does **not** prove whether Flike 5.0.306.0 uses LN2, which log/flow coordinate LP3 matches, whether GP location is fixed, default shape bounds, threshold handling, fitted-support policy or bootstrap interpolation. These require a pinned source/settings-matched native test matrix. The ARR screenshot itself is from an older Flike 5.0.220.0 release.

**Active initial development (#7/#40):** complete independent estimator review, explicit supported input adaptation, standardized optimized-H mathematics and CLI integration. Missing primary mathematical definitions remain active research/specification work. See [optimized-H specification](wp04-optimized-h-specification.md).

**Deferred native evidence (#39/#12):** actual Flike family variants/raw-versus-log choices, native fixed-location/default bounds, target bootstrap controls and settings-matched parameter/failure cases. These do not block the defined mathematical component, and their absence must not be presented as a native pass. Full WP04 mathematical completion still requires its active scope; native replication remains separate.

## Independent review corrections: reference invariance and numerical boundaries

An independent Codex review identified four P2 numerical defects in the
initial draft, all addressed with targeted analytical regressions.
**LP3:** sample L2/L3 are now computed from raw natural logarithms before
any change of reference discharge. Only L1 is shifted, preserving exact
zero L-skew for the [0.25, 1, 4] source sample under references 1, 2
and 10 without clipping genuine nonzero skew. For large gamma shape,
SD/L2 uses a stable expansion of sqrt(pi*a) Gamma(a)/Gamma(a+1/2)
rather than subtracting two ~1e9-magnitude log-gamma values. Tests
compare both skew signs at a=10,000, 100,000 and 1e8 with original
80-decimal mpmath reference values (4e-15 absolute SD tolerance).

**Generalized Pareto:** free- and fixed-location inversions now use
nonnegative integer-weighted adjacent sorted spacings for
L2-L3, L2+L3 and L1-L2, avoiding cancellation near finite-mean
xi=1. Independent exact rational three-point data [0, 2e-16, 1]
verify shape, scale and location without using the production formula.
Fitted support is finally checked in the **returned kernel's actual
log-scale parameterization**. The previous [0, 1-1e-8, 1]
near-endpoint case cannot silently return a kernel excluding its
maximum flow after that parameter conversion; unsupported fits
are explicitly rejected. Every accepted test fit is checked
against the returned kernel's support/standardized upper endpoint.

These are mathematical and binary64 guarantees, not Flike 5.0.306.0
settings, numeric acceptance thresholds or native qualification.

## Nonzero fixed-location GP rounding correction

The second independent Codex review found a residual cancellation when
the stable L1-L2 gap sum was **prematurely added to the minimum sample
value** before a supplied nonzero fixed location was subtracted.
With [-1, -1+2e-16, 0] and fixed location -1, adding the small gap
sum to -1 rounds away its contribution. The implementation now retains
the anchor and gap sum separately; it subtracts the fixed location from
the anchor *first* and only then adds the gap contribution with
math.fsum. A new exact-Fraction three-point U-statistic regression
verifies scale 7.401486830834379e-17, including both free and
fixed-location boundary coverage. No arbitrary scale floor or
reference-dependent tolerance was introduced.

## Discharge-unit invariance correction

The Codex follow-up review of PR #38 identified a distinct numerical failure
after the reference-only correction: scaling the exactly geometric annual
maxima [0.25, 1, 4] to [0.5, 2, 8] could generate a nonzero log-space
third L-moment when absolute logarithms were taken before centering.
The estimator now chooses the observed median discharge as its logarithmic
anchor and computes L1-L3 on log(Q/anchor). It shifts **only** the fitted L1
by log(anchor/reference) into the caller's requested source coordinate.
For an extreme finite source ratio outside binary64's division range,
the fallback is log(Q)-log(anchor) and remains explicitly subject to
numerical verification. LP3 support is assessed in the same centered
coordinates rather than subtracting two large absolute-log endpoints.

The regression explicitly rescales both annual observations and reference
discharge by factors 1, 2, 3 and 10, checks exact zero log L-skew,
invariant L1-L3 sample coordinates and proportional physical F=0.5
and F=0.9 quantiles. It does not round a genuine small nonzero skew to
zero or broaden the admitted gamma-shape interval; unsupported observations
continue to fail. This is a **numerical invariance** check for the
published mathematical estimator, not a native Flike comparison.

## Subnormal logarithmic ratio handling

An independent review identified quantization of positive **subnormal**
ratios before evaluation of their logarithms. The LP3 log-coordinate
helper now takes log(numerator/denominator) directly only for
representable normal ratios. For a subnormal, underflowed or overflowing
ratio of two finite positive quantities, it uses
log(numerator)-log(denominator) before any precision is discarded by
floating-point division. Both observation/median-anchor and
anchor/reference ratios share this logic.

A test uses 80-digit Decimal logarithms of the actual supplied
binary64 operands for (1e-20 / 1.7e303), (1e-320 / 1e-10)
and (1e300 / 1e-20), with a 3e-13 absolute log tolerance.
A full LP3 fit of [2.5e-21, 1e-20, 4e-20] checks that
changing only the reference discharge from 1 to 1.7e303
preserves zero LH skewness and the physical median/tail
quantiles, rather than incorrectly moving them by approximately
16 percent. No arbitrary small-probability or parameter floor
is added to the fitting method.

## Nonbinary reciprocal-log symmetry

The independent integration review identified one further arithmetic
issue: even for an exact geometric flow series [1, 10, 100],
binary64 log(0.1) can differ from -log(10), falsely producing positive
L-skew. The revised stable log-ratio operation now evaluates the
**larger numerator/denominator** and applies its appropriate sign;
log(1/k) and log(k) are exactly antisymmetric when the ratio k is
representable. For a larger/smaller quotient that overflows, it uses
the difference of their two finite logarithms rather than an
unrepresentable quotient. This preserves the prior subnormal and
large-range regression guarantees without a skew-clipping tolerance.

An independent mathematical regression verifies exact reciprocal
symmetry for ratios 3, 7 and 10; geometric three-point records
[1,7,49] and [1,10,100] must retain zero log L-skew under unit
factors 1,2,3,10, and their physical F=.5/.9 quantiles must scale
proportionally. The support and genuine nonzero-skew rejection
contracts remain unchanged. No Flike native variance or vendor
default is inferred.

## Overflow-safe exponents and unresolved machine-scale skew

The final integration review identified geometric values
[2**-1073, 2**-49, 2**975] whose reciprocal quotient overflows
binary64, causing separately rounded absolute logarithms to
produce a false nonzero log L-skew. The logarithmic ratio helper
now decomposes each positive finite operand into a mantissa
and binary exponent with frexp when direct division would
overflow. Equal exponent differences and mantissas produce
identical logarithms with opposite signs; no unverified
model-selection epsilon is introduced.

Near unity, taking a rounded quotient can discard information
in the difference of adjacent binary64 values. The helper
instead evaluates signed log1p((larger-smaller)/smaller) when
the positive difference is at most the smaller operand.
This maintains reciprocal symmetry and resolves the
source ratio before rounding it to unity.

Three-parameter LP3 shape cannot be inferred reliably when a
complete discharge sample spans four or fewer units in the
last place (ULPs) at its median. Those ill-conditioned inputs
now **fail explicitly**, rather than receive a false Normal
zero-skew limit. For the review sample [1.5,
1.5000000000000002, 1.5000000000000004], independent
100-digit Decimal log ratios give nonzero sample L-skew,
while a rounded quotient falsely implied exact zero.
The explicit four-ULP input-resolution guard is a
binary64 numerical admission criterion, **not** a vendor
Flike threshold or an arbitrary skewness clipping policy.
Existing geometric/log-normal and physical scaling laws
remain independently tested.

## Exact three-point skew-zero admission

An additional independent review found that a hard four-ULP series-span
criterion neither proved a zero third L-moment nor correctly identified
which samples are supportable: [1, 1.1, 1.2100000000000002] spans a
much wider discharge range yet its binary64 logarithms can cancel to
a false zero, while 100-digit logarithms of the *actual binary64
values* show genuinely nonzero skew. The four-ULP span gate has been
removed.

For a sample of **exactly three positive binary64 observations**,
the third sample L-moment of log(Q) vanishes if and only if
Q_low * Q_high = Q_middle squared. This is now verified by exact
rational arithmetic on the supplied binary64 observations, not
floating-point multiplication or a skew epsilon. A computed zero
L-skew that fails this mathematical identity raises an explicit
unresolved-precision ValueError; a proven three-point geometric sample
retains the Normal-limit fit. Tests independently verify the new
counterexample using 100-digit Decimal logarithms, and preserve
[0.25,1,4], [1,7,49], [1,10,100] and the huge exponent geometric
record. Larger samples still use the declared numerical estimator;
they do not inherit an unverified three-point identity or arbitrary
normal-limit rule.

## Exact logarithmic mirror symmetry for longer records

The independent Codex review of the three-point symmetry admission
found a six-point counterexample [1,2,4,8,16,32]. Its third log
L-moment should be exactly zero, but binary64 weighted spacing sums
introduced a residual of order 1e-17 and the fit falsely rejected a
Normal-limit distribution. An exact rational pair-product test now
recognizes log-symmetric **any-length** complete samples: after sorting,
every low/high pair must have the same product, and for an odd-length
sample the central value squared must equal that product. This is a
**sufficient**, source-exact identity for zero odd log L-moments; it is
necessary as well for the three-point L3. The computed third moment is
set to exactly zero only when that identity holds. No tolerance,
blanket near-zero truncation or guessed Flike skew policy is introduced.
Nonmirror longer series retain their independently calculated third
moment and may have legitimate zero L3 without mirror symmetry.
Tests use even four- and six-point geometric and nongometric symmetric
records, tied symmetric samples, physical-unit scaling, and a
nonmirror six-point negative control.

This continuation also uses Python 3.11-compatible TypeAlias/TypeVar
and Generic declarations instead of Python 3.12's PEP 695 syntax,
so the mathematical module follows the repository's newly expanded
Python 3.11–3.14 runtime support. Modern built-in collections and
union annotations are retained. No new dependency is introduced.

## General exact third log L-moment admission

A subsequent independent numerical review demonstrated that a
four-point non-geometrically-symmetric record
[1, 1.1, 1.1, 1.2100000000000002] can have its binary64 log
L3 round to zero even though the exact binary64 observations
have genuinely nonzero L3. A source-exact rational identity now
verifies any claimed zero L3 for **n=3..16**:
the integer weight on the i-th ascending log observation is
C(i,2)-2*i*(n-1-i)+C(n-1-i,2), so exactly zero L3 is equivalent
to equality of the product of positive- and negative-weighted
integer powers of the exact binary64 source discharges.
The implementation forms the rational products with
fractions.Fraction; it does not infer mathematical zero from a
rounded logarithm or an arbitrary epsilon.

This criterion also admits genuine nonmirror zero-L3 series
such as [1,2,4,32,32], whose signed third-moment weights
are [6,-3,-6,-3,6]. The original all-size exact mirror-pair
identity remains a separate sufficient proof for any sample
length. For larger nonmirror samples where the finite-precision
calculation returns exact zero, the result explicitly fails as
**unverified log L-skew** rather than silently fitting Normal.
The n<=16 computational bound limits potential rational power
growth and is not a native Flike data-size or skewness rule.
