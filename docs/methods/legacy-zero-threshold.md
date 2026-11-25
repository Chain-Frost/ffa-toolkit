# LN/LP3 zero-threshold treatment

**Method ID:** `legacy-zero-threshold`. **Legacy target:** Flike 5.0.306.0, file 3.10. **Qualification:** not established. Only a source-coordinate logit conversion exists in WP07; the actual legacy likelihood, prior, posterior and native fit are unavailable.

## Practical purpose and required evidence

Annual-maximum floods may be recorded as literal zeros or only known to be at/below a declared discharge threshold. These states are not identical, and neither is a missing year. Logarithmic LN/LP3 positive-flow models need an evidenced treatment rather than a fabricated small positive offset. Preserve original observations, thresholds, units, source flags and years. Low-flow screening and zero-threshold probability are different operations.

Required inputs for a future fit: supported LN or LP3 combination, exact or at/below-threshold records and exposure, `q0` and its units, any fixed/fitted status, prior/no-prior option, bound comparisons, source parameter ordering and the target legacy configuration.

## Mathematical coordinate (known) versus probability model (unknown)

The inspected LP3 report includes a parameter labelled as logit of the probability that flow is at or below the zero threshold. For `p0 = P(Q <= q0)`, `eta = log[p0/(1-p0)]` and

```text
p0 = 1/(1+exp(-eta))
eta = log(p0)-log(1-p0), 0 < p0 < 1
```

This is an identity relating the source logit coordinate and threshold probability, **not** evidence of a literal-zero point mass, positive-component CDF, fitted Beta prior, default threshold or inequality endpoint. WP07 computes this identity stably without imposing the missing model.

The observed LP3 source coordinates are mean of natural-log flow, natural logarithm of the standard deviation of natural-log flow, skew of natural-log flow and, where enabled, this separate logit. Averaging source log-standard-deviation is not equivalent to averaging standard deviation after exponentiation.

## Outputs, uncertainty and support

Future reports must label `q0`, units, `p0`, `eta`, fitted/fixed status, posterior summaries and threshold counts. Any probability limit must identify the underlying random quantity, level, side and construction. Numerical integration precision is not parameter uncertainty. Cases just below, equal to and above `q0`, zeros, missing years and overlap with grouped exposure need independent tests. At present **no LN/LP3 zero-threshold fitting combination is qualified or exposed by the CLI**.

**Scientific foundation:** ARR 2019 Book 3 Chapter 2 §§2.4.2.4, 2.6.3.1–2.6.3.4 and 2.6.3.8–2.6.3.9 (`ARR3`).

**Observed legacy fact:** an LP3 report logit label (`LEGACY_SAMPLES`). **Current project rule:** retain its natural-log source coordinates; no invented literal-zero mixture or default prior. **Unresolved:** exact target likelihood, bound convention, prior, inference and native comparison. See repository `docs/wp07-low-flow-evidence.md` and offline `ffa references ARR3`.
