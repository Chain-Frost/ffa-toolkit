# WP07 — Independent MGBT rank-statistic verification

**Scope:** a mathematical component for [development requirements](https://github.com/Chain-Frost/ffa-toolkit/blob/main/docs/development.md), not a completed Flike 5.0.306.0 multiple Grubbs–Beck implementation. The [canonical source-pinned variant comparison](wp07-mgb-variant-comparison.md), merged by PR #27, is authoritative for the R, Numerics, BestFit and FlowFreq sweep and quadrature differences. This document records the *separate rank-statistic calculation*, its tests and its remaining numerical qualification. The repository retains Sustainable Use License 1.0.

## Statistic and explicit source transform

For measured annual peaks Q sorted ascending, define the source-specific transformed values zᵢ=log₁₀(max(Qᵢ,q_floor)). For candidate rank r=1,...,floor(n/2), the generalized Grubbs–Beck statistic is

\[
W_r=\frac{z_r-\overline z_{r+1:n}}{s_{r+1:n}},\qquad
s^2_{r+1:n}=\frac{\sum_{i=r+1}^{n}(z_i-\overline z_{r+1:n})^2}{n-r-1}.
\]

This calculation uses the **upper n-r observations**, excludes the candidate from the upper mean and uses the *sample* variance denominator n-r-1. The independent Python implementation is in \`src/ffa_methods/mgbt_statistic.py\`. It preserves the complete original discharge array, unit label, source identity and input positions. The log floor affects **only the statistic**, never the actual observations or historical exposure. It rejects invalid measurements, ill-defined candidate counts and upper samples with zero variance.

After independent Codex review identified a near-one-ULP rounding problem, all-equal transformed upper values are rejected *before* mean/variance evaluation. In the Python 3.14 counterexample \`(1,2,3,4,5,7,7,7,7,7)\`, a rounded upper mean had previously produced an erroneously nonzero variance and a huge finite W. A regression now rejects it, while accepting an upper set with a genuinely different final value. No arbitrary near-zero variance tolerance is imposed.

**The rank statistic is not a p-value.** \`verify_mgb_cohort\` detects a supplied-p-value decision using a different flow cohort, order or candidate count; it does not validate the numerical values or statistical variant of those supplied probabilities.

## Independently checked published numerical case

The fixed [CRAN MGBT 1.0.7 \`man/MGBT.Rd\`](https://github.com/cran/MGBT/blob/aa4356208f05d6036f3f425ec37c656da345bba0/man/MGBT.Rd) documents USGS station **08066300** with **51 flows**, **25 six-decimal rank statistics**, **25 eight-decimal p-values**, and one selected low flow of **55 cfs** below the first-retained **284 cfs**. All 25 computed W statistics agree with printed values within their stated six-decimal rounding interval; this is independent numerical verification, not a native Flike fit or an executed R test. The supplied-p-value sweep decision is tested separately in \`tests/test_wp07_low_flow.py\`.

The same R manual prints **p=0.01192184** for the first candidate, while separately citing PeakFQ 7.1 at approximately **0.0123**. Both select the same flow, demonstrating that **classification agreement does not imply numerical p-value agreement**. The manual also distinguishes an older 08165300 \`MGBTcohn2016\` selection of 24 low flows from 16 for a later variant and PeakFQ. Neither case identifies the precise Flike 5.0.306.0 method.

**Source custody:** CRAN \`MGBT\` **1.0.7**, source Git commit \`aa4356208f05d6036f3f425ec37c656da345bba0\` (21 July 2021), \`man/MGBT.Rd\` blob \`b12b8ceafd967f71f90ed07b570e1b2418f0d6c1\`, source \`R/MGBT.R\` blob \`0885ad5dce4a12802652f0274526c40336c79753\`, CC0. Git blob identifiers are **SHA-1**, not a source-tarball SHA-256; the latter remains unverified. [Full source custody, entry point and licence](wp07-mgb-variant-comparison.md). No CRAN implementation code is copied or R runtime installed.

## Source-specific zero and sample handling

- Pinned **CRAN MGBT 1.0.7 \`MGBT17c\`** uses \`sqrt(machine epsilon)\` ≈ 1.49×10⁻⁸ in its temporary log transform. Older differently named R routines used 10⁻⁸. The test fixture's default explicitly models the \`MGBT17c\` floor, although the published 08066300 flows are all positive and well above either floor, so those 25 statistics **cannot verify** the zero-handling choice.
- [USACE Numerics v2.2.0](https://github.com/USACE-RMC/Numerics/blob/7e8e8d1c5f26e045a35ec9fc09367de95ed05b02/Numerics/Data/Statistics/MultipleGrubbsBeckTest.cs) and [pinned FlowFreq](https://github.com/pinhead001/hydrolib/blob/7fe7cac1f88998253c81a7793a04c677c34b5b88/flowfreq/bulletin17c.py) use a 10⁻⁸⁸ calculation floor with different downstream integration and failure rules. The statistic tests deliberately allow this as an **explicit different convention**, without calling it Flike or the R1.0.7 default.
- **Flike 5.0.306.0** source/version, zero rules, minimum sample, alpha settings, test statistic and integration method remain unknown. The [ARR Book 3](https://www.arr-software.org/pdfs/ARR_190514_Book3_V4.2.pdf) example supports separate excluded annual observations and grouped censored counts, not a verified default MGBT implementation.

Positive-flow scale invariance (without floor activation), original-order permutation invariance, hand-calculated sample variance, ties and zero-floor sensitivity are separately tested. A degenerate upper sample **must fail closed**, not be assigned an arbitrary significance probability.

## Gates before advancing WP07

The one-sided conditional p-value requires the rth normal order-statistic distribution and source-specific integration, not merely W. The next mathematical package must pin a specific published version, implement and independently check its p-values (including failures, small samples, zeros, ties, finite precision and quadrature) and **not** silently fall back to p=0 or p=1 on numerical failure. USGS, Numerics and FlowFreq may agree on selected outliers while differing in p-values and no-stop sweep rules. Neither becomes the Flike algorithm without controlled native evidence.

The following full WP07 requirements remain open: matching **native Flike 5.0.306.0** MGBT settings/classifications, file3.10 flags, selection threshold precision, a verified observation/grouped-censor likelihood without double counting alongside WP02/WP05, and exact LN/LP3 zero-threshold likelihood, prior, parameter uncertainty and endpoint semantics. The reported \`logit P(Q <= threshold)\` is a coordinate identity, **not** permission to invent a literal-zero mixture or Beta prior.

No new CLI fitting option, GUI, external runtime dependency, GPL source, release or native qualification is part of this PR. Keep [development requirements](https://github.com/Chain-Frost/ffa-toolkit/blob/main/docs/development.md) open after any safe merge of this bounded component.
