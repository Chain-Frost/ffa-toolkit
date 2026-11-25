# Multiple Grubbs–Beck screening of potentially influential low flows

**Method ID:** `mgb-low-flow-screen`. **Legacy target:** TUFLOW/BMT Flike 5.0.306.0, file 3.10. **Qualification:** not established. WP07 has a bounded, comparison-side screening decision and an explicitly approved ARR-style censor proposal, **not** a complete native Flike MGBT implementation.

## Practical purpose and required evidence

An unusually small annual peak can influence an estimated rare-flood tail. Screening flags *potentially influential low flows* (PILFs) for engineering appraisal; it does not prove a flow was mismeasured or authorise removing an observation. The analyst must review gauge/rating history, annual-maximum extraction, stationarity, unusual catchment processes, year labels, original flags, ties and genuine zero discharges. Missing years are not small floods.

The required record is an explicitly selected set of **exact, systematic annual peaks**, with source indices, year definition, units and immutable original magnitudes. Excluded records and grouped historical threshold counts are separate evidence; do not silently include them in screening or discard them from exposure.

## Technical specification and supported combinations

For ordered selected peaks `q(1) <= ... <= q(n)`, a generalized Grubbs–Beck test evaluates candidate low ranks. WP07 has **two distinct comparison components**: `low_flow.select_mgb_candidates` performs the outward, inward and zero-in sweeps on explicitly *supplied*, source-identified rank p-values; `mgbt_pvalues.calculate_usgs_mgbt_1_0_7` independently calculates those conditional probabilities **only for the pinned external USGS MGBT 1.0.7 procedure** and passes them to that same decision. The decision uses `k = max(k_out, k_in, k_zero)`. If `0 < k < n`, the first retained flow `q(k+1)` is a selection boundary. A split through tied observations is rejected. Neither component identifies the native Flike algorithm, defaults, threshold rounding or fitting rules.

For its separately documented USGS MGBT17c comparison only, settings are `alphaout=0.005`, `alphain=0`, `alphazeroin=0.10`, `n2=floor(n/2)`. These are **not verified legacy defaults**. WP07 may propose an explicitly sourced threshold and separate user-approved exclusion/group transformation; it must not automatically mutate the record or likelihood.

### Source-pinned numerical comparator (not a Flike fitting mode)

The comparator is based on **CRAN USGS MGBT 1.0.7**, [commit `aa4356208f05d6036f3f425ec37c656da345bba0`](https://github.com/cran/MGBT/commit/aa4356208f05d6036f3f425ec37c656da345bba0), whose DESCRIPTION declares **CC0**. The 51 published annual peaks for USGS gauge **08066300** have 25 published rank p-values. Their comparison selects the lowest **55 cfs** as a potentially influential low flow and reports first retained **284 cfs**. This comparison is **not** an original Flike run or proof of a matching Flike default.

For rank `r`, the reference statistic is `W(r) = [z(r) - mean(z(r+1)..z(n))] / sd_sample(z(r+1)..z(n))`, with `z=log10(max(flow, sqrt(binary64 epsilon)))` for the **source-specific test calculation only**. Original zero flows, magnitudes, year labels, flags and discharge units remain unchanged. The conditional one-sided significance p-value integrates the reference's truncated-Normal/Gamma approximation and noncentral-t **survival probability** over `[sqrt(epsilon), 1-sqrt(epsilon)]`. This p-value is **not** the flood non-exceedance probability `F(Q)`, an AEP, or the reported LN/LP3 threshold probability.

The implementation uses the existing SciPy dependency and **deterministic adaptive quadrature** (not a random Monte Carlo replacement). For each candidate rank, it returns the evaluated probability together with the quadrature's **estimated absolute integration error**. Undefined upper-sample variance, non-finite calculations, integration warnings, unresolved tiny probabilities and saturated endpoints **fail explicitly** instead of being silently classified as outliers or accepted as an arbitrary zero/one. The numerical error estimate is not flood-estimate uncertainty, posterior uncertainty or a confidence interval. The 25 printed USGS values are compared within a justified **2e-6 absolute printed-source tolerance**, separately from stricter numerical-integration checks. This tolerance is **not** a native Flike acceptance criterion.

Only Python-internal comparison is implemented; the installed CLI does **not** offer this comparator as an enabled Flike analysis. See `docs/wp07-conditional-mgbt-spec.md` for the full source pin, integral, independent quadrature test and source-specific failure boundaries. The repository remains SUL-1.0 with applicable source notices.

ARR Book 3 §2.8.6.4 depicts 27 low Wimmera records excluded while adding a separate **27 at/below, 0 above** group with non-observed negative synthetic editor identifiers. Printed example threshold values **54.396** and **54.399 m³/s** differ, while the first retained row is **54.400 m³/s**. No number is silently replaced by the selection boundary. The transformed group is *not* an independent 27-year historical observation.

## Reporting, uncertainty and limitations

A later qualified analysis must report the selected years, original flows and flags, supplied p-values and provenance, cutoffs, ties/zeros, proposed and adopted group threshold, retained exposure, sensitivity of actual fitted outputs and solver diagnostics. Screening does not provide an interval estimate of a design flood; neither comparison-side p-values nor a clean decision qualify Flike's likelihood or tail uncertainty. The current installed CLI does **not** run MGBT fitting.

**Scientific foundations:** ARR 2019 Book 3 Chapter 2 §§2.6.3.9, 2.8.6.4 (`ARR3`); Cohn, Lane and Baier (2013), DOI 10.1002/wrcr.20392 (`COHN2013`).

**Observed evidence:** ARR's printed exclusion and grouped-censor example, plus unresolved target settings and sample correspondence (`LEGACY_SAMPLES`).

**Current implementation decision:** WP07 supplied-p-value sweeps, the independently tested **external USGS 1.0.7 conditional p-value comparator**, and an opt-in evidence-preserving ARR proposal are available as mathematical/internal components. **Outstanding for the legacy Flike MVP:** actual Flike MGBT variant and default p-value integration, source flags, native threshold rounding, controlled run correspondence and matched likelihood/quantile outputs. This research comparator does not enable a native Flike fit or change the method's unqualified status. Detailed records: `docs/wp07-low-flow-evidence.md`, `docs/wp07-mgb-variant-comparison.md`, `docs/wp07-conditional-mgbt-spec.md`. Citations are available with `ffa references ARR3` and `ffa references COHN2013`.
