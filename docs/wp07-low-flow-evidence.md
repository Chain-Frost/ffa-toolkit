# WP07 — Low-flow screening and zero-threshold evidence boundary

Status: **partial reference/record implementation; not a qualified Flike 5.0.306.0 reproduction**. Issue [development requirements](https://github.com/Chain-Frost/ffa-toolkit/blob/main/docs/development.md); overall tracker [development requirements](https://github.com/Chain-Frost/ffa-toolkit/blob/main/docs/development.md). This is CLI internal composition, not a new public API or an enabled fitting command.

## Separate operations

| Operation | Representation | Current support |
| --- | --- | --- |
| Low-outlier *screening* | Statistical test of selected exact systematic years | Explicitly sourced three-sweep **decision on supplied rank p-values**, preserving source indices and nonnegative flows; independently calculated USGS probabilities are available in the separate component described in the implementation acceptance section |
| Prior *exclusion* | Annual EXCLUDED row retaining original year, flow and source flag | Retained separately; not implicitly in the screening cohort |
| Individual *censoring* | Annual threshold or interval constraint | Retained; not transformed into an exact value or omitted year |
| Grouped threshold/count | CensorGroup with threshold, counts and exposure | Retained, checking disjointness from the screened cohort; not constructed from excluded rows |
| LN/LP3 *zero-threshold* parameter | Reported logit P(Q <= q0) | Stable source-coordinate transform only; no independently assumed literal zero atom, empirical p0 or Beta prior |

An exact zero is not an absent year, a flagged exclusion or an assumed threshold observation. A source AT_OR_BELOW_ZERO_THRESHOLD row conveys threshold information, not necessarily an exact zero. No positive offset is applied to source flows. A screening decision is an **annotation**, not a fitted Flike likelihood or approved censoring transformation.

## Documented USGS comparison, not a Flike default

The [source-pinned CRAN MGBT 1.0.7 manual](https://github.com/cran/MGBT/blob/aa4356208f05d6036f3f425ec37c656da345bba0/man/MGBT.Rd) documents the MGBT17c procedure and three ordered sweeps. See the [version, Git blob hashes and variant comparison](wp07-mgb-variant-comparison.md) before using the supplied-probability decision. The pinned **comparison** settings are alphaout=0.005, alphain=0, alphazeroin=0.10 and n2=floor(n/2). The component accepts independently calculated one-sided p-values for ranks 1..n2 and computes the outward, inward and zero-in indices; their maximum defines the candidate low-outlier count. It reports the next retained order statistic as a threshold, using a **strictly below** comparator. A cutoff splitting tied flows is rejected pending an explicitly qualified source rule.

The R manual's USGS 08066300 sample has 51 flows and 25 published rank p-values. The independent decision test selects only 55 cfs as a potential low outlier, with a threshold of 284 cfs. This checks the **sweep decision only**. It does not validate the p-value formula, native Flike settings, uncertainty or fitted discharge results. [Cohn et al. (2013)](https://doi.org/10.1002/wrcr.20392) provides the generalized Grubbs-Beck statistic and conditional probability foundation.

There are material variant differences. The source-pinned USGS **MGBT17c** applies a temporary log10 floor of `sqrt(machine epsilon)` (other older R variants use `1e-8`), whereas the pinned Numerics and FlowFreq/PeakFQ-derived implementations use `1e-88`. They also differ in sample restrictions, quadrature and fallback rules. The pinned USGS `mgbt.sweeper` leaves the second and third sweep indices **at zero** if no stopping probability exists; treating no stop as `n2` would overclassify the sample and has been corrected. FlowFreq intentionally uses a different no-stop third-sweep fallback. These are external algorithm choices, **not changes allowed to original measured discharges**. The Australian fork's repeated single-outlier test with deletion is not the three-sweep method. The real Flike build's algorithm/version, one-sided significance, numerical integration tolerance, tie resolution, zero behavior and output effects remain unverified. An integration failure must not silently become an outlier decision. See the [pinned comparison](wp07-mgb-variant-comparison.md) for exact source identities and distinctions.

**Independent rank-statistic checks:** [the version-pinned MGBT omega verification](wp07-mgbt-variant-audit.md) separately checks all 25 published USGS rank statistics, the source-specific logarithmic floor and degenerate upper samples. The corrected [source variant/sweep comparison](wp07-mgb-variant-comparison.md) remains authoritative for sweep semantics. None of these methods yet qualifies a Flike 5.0.306.0 p-value or fit.

## Documented Flike censoring operation (distinct from the test statistic)

ARR Book 3 §2.8.6.4 and Figures 3.2.37–3.2.38, printed pages 81–82 ([official volume](https://www.arr-software.org/pdfs/ARR_190514_Book3_V4.2.pdf)), explicitly record **two** changes after the user accepts the multiple Grubbs-Beck selection:

1. The selected 27 lowest Wimmera gauged values are marked **excluded** from calibration, retaining the underlying recorded flows.
2. A separate grouped censored record is added, with **27 years at/below a threshold and zero years above**, using synthetic years **-1000 through -974** to satisfy the legacy editor's non-overlap check. Those years are not actual historical observation years.

The printed prose reports a threshold of **54.396 m³/s**, while the editor screenshot displays **54.399 m³/s** and its first retained gauged value is shown as **54.400 m³/s**. These numbers are not identical at reported precision. The first retained flow (the selection boundary) must **not** be silently used as the group threshold. Until the actual legacy source selection/rounding rule is qualified, `propose_arr_mgb_censoring` requires an explicitly sourced threshold **strictly between the highest selected and lowest retained observations**, plus an explicit synthetic start year. It checks for collisions with existing groups.

The proposal preserves the original immutable source series. `apply_approved_arr_mgb_censoring(..., approve=True)` separately materializes excluded annual **kinds** while retaining original years, magnitudes, units and **raw source flags**, and appends one grouped count. The source flag remains raw input metadata rather than a fabricated file3.10 exclusion code. Synthetic group years have `ExposureKind.UNRESOLVED` deliberately; they do **not** assert an observed historical period or an independently qualified WP05 likelihood. The proposal is not automatically accepted, invoked by the CLI or applied to an unreviewed Flike project. This operation follows documented **UI transformation semantics only**; classification p-values, group endpoint precision, native output and inference remain unqualified.

## Annual records and exposure

The audit requires named **exact systematic years**, in exactly the order of supplied flows. Original annual values, units, flags, excluded/missing years, grouped historical exposure, zero threshold and source metadata remain unchanged. It rejects duplicate years, nonexact/non-systematic or absent records, flow/order mismatches and overlapping or unbounded censor-group exposure. Unbounded historical groups are deliberately refused: the historical period cannot yet be proven disjoint and double counting would be possible. No candidate classification changes an observation kind or constructs a count-only group.


## Zero-threshold logit is not a new probability model

For an **already reported** parameter eta=logit(P(Q <= q0)), its identity is p=1/(1+exp(-eta)) and eta=log(p)-log1p(-p) for 0<p<1. The code preserves original units, declared family and source identifier and evaluates extreme logits without overflow. At floating-point extremes the returned probability may round to zero or one; this is not evidence of a certain literal atom. LP3 reports expose this coordinate. The supplied evidence does **not** identify a corresponding LN fitted mode, prior, inference/uncertainty rule, conditional positive-flow distribution or threshold endpoint convention. Those remain blocked. No zero mixture, replacement prior or automatic zero-count likelihood is enabled.

## Initial mathematical/software acceptance — issues #10 and #40

The original module implements supplied-p-value sweep decisions, auditable observation/exposure handling, separately approved exclusion/count proposals and threshold-logit identities. The package now also includes source-pinned USGS conditional MGBT p-values and numerical quadrature checks in merged PR #33, completing #32; see [conditional MGBT evidence](wp07-conditional-mgbt-spec.md). The source-preserving paired-censor likelihood integration is merged in PR #34, completing #31; see [paired-censor integration](wp07-paired-censor-integration.md). Do not list those delivered components as missing or repeat the old #17/#18 integration task.

Retain independent source/version, probability/sweep, tie/zero, numerical-failure and exposure tests. The 24 November closeout audit accepted the completed screening/audit/censor-interface/threshold-coordinate scope using merged final-head CI and independent review receipts recorded in [development requirements](https://github.com/Chain-Frost/ffa-toolkit/blob/main/docs/development.md). Its native-only remainder is transferred to #39. Supplied threshold-logit conversion is not a fitted zero-probability inference model, and downstream sampler/CLI orchestration remains #9/#11 work.

## Deferred native evidence — issue #39

Exact Flike MGBT variant/settings, source flags, threshold rounding and native classification/output effects remain unknown. So do the native LN/LP3 zero-threshold endpoint/likelihood/prior/posterior rules. Those requirements are transferred to #39 with final native conformance in #12. Keep unsupported zero inference explicit and do not invent a literal-zero atom or prior. Whole legacy-profile registry status is separate from the documented implemented independent components; this closeout does not qualify a native analysis or authorise publication. See [development stages](development-stages.md).
