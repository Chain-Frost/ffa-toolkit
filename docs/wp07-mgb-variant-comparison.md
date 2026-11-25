# WP07 — Source-pinned MGBT variant and sweep comparison

**Status:** source-identified external reference comparison, including
the supplied-p-value sweep decision and an independently reviewed USGS
MGBT 1.0.7 conditional p-value mathematical comparator under [development requirements](https://github.com/Chain-Frost/ffa-toolkit/blob/main/docs/development.md).
None of these sources identifies the algorithm, settings, test p-values or treatment in
TUFLOW/BMT **Flike 5.0.306.0**. The USGS comparator is not a legacy Flike
fitting mode, source-licence grant, or native qualification.

## Immutable source custody

The official `MGBT` entry point in the **CRAN MGBT 1.0.7** mirror forwards to
`MGBT17c`; it must not be compared interchangeably with `MGBTcohn2013`,
`MGBTcohn2016`, `MGBTnb`, `MGBT17c.verb` or later CRAN 1.1.8.
The pinned CRAN mirror commit is
[`aa4356208f05d6036f3f425ec37c656da345bba0`](https://github.com/cran/MGBT/commit/aa4356208f05d6036f3f425ec37c656da345bba0),
dated **21 July 2021**, with commit message **version 1.0.7** and Git tree
`82ebca79ba8d951dcc595e9a7d75dbad04922434`. The pinned
[`DESCRIPTION`](https://github.com/cran/MGBT/blob/aa4356208f05d6036f3f425ec37c656da345bba0/DESCRIPTION)
records `Version: 1.0.7`, USGS contributors and **CC0**.

| Audited release component | Immutable Git blob SHA-1 | Function |
| --- | --- | --- |
| [`R/MGBT.R`](https://github.com/cran/MGBT/blob/aa4356208f05d6036f3f425ec37c656da345bba0/R/MGBT.R) | `0885ad5dce4a12802652f0274526c40336c79753` | `MGBT17c` and `mgbt.sweeper` |
| [`man/MGBT.Rd`](https://github.com/cran/MGBT/blob/aa4356208f05d6036f3f425ec37c656da345bba0/man/MGBT.Rd) | `b12b8ceafd967f71f90ed07b570e1b2418f0d6c1` | Documented 08066300 flow, omega and p-value literals |
| [`R/RthOrderPValueOrthoT.R`](https://github.com/cran/MGBT/blob/aa4356208f05d6036f3f425ec37c656da345bba0/R/RthOrderPValueOrthoT.R) | `a739ec5962d7dc059d32dc316dfc3725db364c62` | Numerical integration and Monte Carlo fallback |
| [`DESCRIPTION`](https://github.com/cran/MGBT/blob/aa4356208f05d6036f3f425ec37c656da345bba0/DESCRIPTION) | `3585133d471e2ed572ce2c3b2ed35b3978ebd491` | Exact version, date, rights and authors |

The identifiers in this table are **Git blob SHA-1 hashes**, not tarball
SHA-256 hashes. Neither a CRAN binary nor an archive hash is claimed. No
third-party code was copied into the implementation. The published 51 observations
and 25 eight-decimal p-values were already present as literal test evidence;
the pinned `man/MGBT.Rd` identifies their exact source and CC0 basis.

## What the supplied-p-value decision actually tests

For sorted flows `Q_(1) <= ... <= Q_(n)`, the documented statistic uses
`Z_i = log10(max(Q_i, floor_variant))` and, for a candidate rank `r`,
`W_r = (Z_r - mean(Z_(r+1)..Z_n)) / sd_sample(Z_(r+1)..Z_n)`.
The rank p-value follows the source-specific Cohn/orthogonal-t conditional
integral. The WP07 adapter accepts these externally produced rank p-values:
**it does not calculate `Z`, `W`, or a p-value**. Caller provenance alone
does not authenticate supplied p-values as numerically correct.

For pinned USGS `MGBT17c`, the separate `mgbt.sweeper` starts all indices at
zero and executes three decisions over the `n2=floor(n/2)` ranked probabilities:

1. `J1`: largest rank satisfying `p_r < alphaout`, scanning from `n2` down.
2. `J2`: from rank `J1+1` upwards, take the rank **before** the first
   `p_r >= alphain`; if no such rank exists, leave `J2=0`.
3. `J3`: from rank 1 upwards, take the rank **before** the first
   `p_r >= alphazeroin`; if no such rank exists, leave `J3=0`.

The return is `max(J1,J2,J3)` with source defaults
`alphaout=0.005`, `alphain=0`, `alphazeroin=0.1`. In particular,
**the absence of a stopping rank does not mean that all `n2` values are
outliers in this version**. The preceding WP07 implementation incorrectly
initialized `J2` and `J3` to `n2`. This revision fixes the source-specific
decision and exercises the failure case; it does not declare other variants
incorrect. Equality at either inward cutoff retains the current rank; equality
at the outward cutoff does not classify it. The implementation still refuses
an outlier/retained boundary that splits identical measured discharges.

The published USGS station **08066300** has **51 cfs observations**, 25 printed
rank p-values, `(J1,J2,J3)=(0,0,1)`, one selected value **55 cfs** and
first-retained threshold **284 cfs**. Its source p-value for 55 is
`0.01192184`; the same manual quotes a separately computed PeakFQ 7.1
value **0.0123**. These are different numerical results, and the fixture
asserts only the printed USGS decision, not PeakFQ agreement.

## Comparison of four non-native implementations

| Source and exact revision | Log transform / rank population | Cutoffs and source handling | Integration, special cases and disposition |
| --- | --- | --- | --- |
| **CRAN MGBT 1.0.7**, Git `aa4356208`; CC0 | `MGBT17c` uses `log10(max(Q, sqrt(machine epsilon)))`, `n2=floor(n/2)`; other older R functions use `1e-8` | 0.005 / 0 / 0.1; `mgbt.sweeper` uses zero-initialized `J2,J3`, first retained flow plus optional explicit offset | `RthOrderPValueOrthoT` uses R integration on `[sqrt(eps),1-sqrt(eps)]`, with a stochastic 10,000-sample fallback; unqualified NA must **not** be rewritten to zero in this Python adapter. `min.obs=0` is the R default, **not** a verified five- or ten-value minimum. |
| **[USACE Numerics](https://github.com/USACE-RMC/Numerics/blob/7e8e8d1c5f26e045a35ec9fc09367de95ed05b02/Numerics/Data/Statistics/MultipleGrubbsBeckTest.cs)**, `7e8e8d1`; 0BSD | `log10(max(Q,1e-88))`, searches lower half, upper-sample mean and variance | Same nominal cutoffs; `J2` starts at `J1`, `J3` starts zero, first-retained threshold supplied by caller | For `n<10`, `GGBCRITP` returns **0.5**, not an integrated probability. Adaptive Gauss–Kronrod `1e-16 .. 1-1e-16`, max depth 25; integration failure returns NaN. |
| **[RMC-BestFit](https://github.com/USACE-RMC/RMC-BestFit/blob/bfce2ddbebb8d75e6cf6e21c4fef8ecfb5058269/src/RMC.BestFit/Models/DataFrame/DataFrame.cs)**, `bfce2dd`; 0BSD | Calls the pinned Numerics implementation on exact series | Requires **at least 10** exact observations; flags values **strictly below** the next retained flow | Recalculates plotting positions and derived state; does not create Flike's separate ARR synthetic-year group or prove a native Flike fit. |
| **[FlowFreq](https://github.com/pinhead001/hydrolib/blob/7fe7cac1f88998253c81a7793a04c677c34b5b88/flowfreq/bulletin17c.py)**, `7fe7cac`; MIT, bundled PeakFQ material separately CC0 | Zero-inclusive `log10(max(Q,1e-88))`, `n2=floor(n/2)` | Same nominal cutoffs; `J3` starts **at `n2`**, unlike USGS 1.0.7; rejects `n<5` with no selected low flows | SciPy quadrature over `[0,1]` with `epsabs=epsrel=1e-4`, `limit=50`; numerical failures can fall back to p=1.0. PeakFQ/Bulletin 17C **EMA is not Flike's Bayesian likelihood**. |

The **Australian repeated single-outlier/delete variant** is not a
Cohn/PeakFQ three-sweep oracle. Source-specific log floors are **temporary
calculational transforms**; neither zeros nor original discharge data may be
modified. Quadrature failure policies materially affect selections, so an
automatically substituted finite p-value is prohibited until individually
specified and reviewed.

## Current decisions, tests and remaining gate

- The supplied-p-value adapter deliberately refuses non-finite p-values, a
  missing provenance identifier, negative or non-finite flows, a split tie, or
  fewer than five records. The **five-record limit is a conservative adapter
  contract**, not a USGS mathematical minimum; the Numerics/BestFit ten-record
  restrictions do not automatically transfer to a different source variant.
- Source-anchored tests cover USGS 08066300, strict equality at each alpha,
  no stopping rank, reversed original input order with unchanged rank p-values,
  the five-record adapter boundary and retained original records.
- The MGBT **test significance probability** is not the event non-exceedance
  probability `F(Q)`, and neither is the fitted LN/LP3 probability
  `P(Q <= zero_threshold)`. The latter's reported logit coordinate remains
  a conversion only, with no inferred literal-zero atom or invented prior.
- [development requirements](https://github.com/Chain-Frost/ffa-toolkit/blob/main/docs/development.md) and
  [its reviewed mathematical specification](wp07-conditional-mgbt-spec.md)
  now provide a **USGS 1.0.7-only** conditional p-value evaluator, using
  source-identified log-floor/rank rules and strict deterministic numerical
  integration checks. This is separate from the source-agnostic
  `select_mgb_candidates` function, which still accepts externally supplied
  probabilities. These are **not** verified legacy Flike 5.0.306.0
  controls, flags, censor/exposure effects, prior or uncertainty. This
  document does not close [development requirements](https://github.com/Chain-Frost/ffa-toolkit/blob/main/docs/development.md).

Other reference evidence is staged in
[development requirements](https://github.com/Chain-Frost/ffa-toolkit/blob/main/docs/development.md);
the Wimmera comparison is a *candidate*, not a native MGBT calibration oracle.
