# Held-out direct LH calibration

This synthetic audit evaluates the Toolkit's direct calibrator at every admitted fitted shape. It retains the provisional shape envelope, calibration precision gates and existing seeds. An independent study used an interpolation grid and a larger held-out sample. [Independent PR review](https://github.com/Chain-Frost/ffa-toolkit/pull/32#pullrequestreview-6095232487)

## Design

Before generating or scoring records, the audit fixed six cells with n in {20, 60} and generating ARR k in {-0.3, 0, +0.3}. Each cell used the first 100 complete records from NumPy PCG64 seeded with `2026113000 + 10*n + i`, where i is the zero-based shape index. Draw a `(100, n)` array of uniforms in one call to `Generator.random`; apply the ARR GEV quantile with location 20 and scale 1. The positive shift is affine and leaves the LH ratios invariant. The audit used Windows x64 Python 3.14.6, NumPy 2.4.6 and SciPy 1.17.0 with one OpenBLAS thread per worker.

For each record, call `fit_gev_lh(values, shift=h)` for every H from 0 to 4. If any fit fails or any fitted shape lies outside [-0.8, 0.6], retain the record and refusal without calibration. Otherwise, call `calibrate_fitted_lh(n, fit.shape, h, seed=base_seed + h)` for every H at base seeds 20261128 and 20261129. Each call uses the production 20 × 10,000 simulation budget. Each candidate's absolute z divides the absolute fourth-ratio residual by the square root of the pooled variance. Calculate its z bracket by dividing that residual by the square roots of the upper and lower variance bounds, respectively.

Apply the predeclared cutoffs 1.96 and 2.5758293035489004 to each seed's candidates. Count a calibration refusal if any variance call fails. Otherwise count a cutoff refusal if any bracket satisfies `lo <= cutoff < hi`. On admitted records count the reject indicator for each H, the lowest passing H and no-pass. Compare selections between seeds only on records admitted at both seeds, counting no-pass as a distinct outcome. A fixed-order replay of representative records through `evaluate_candidate_h` checks the independent scoring logic against the runtime entry point.

This audit measures operational behaviour at the sampled cells. With 100 records per cell, a rejection rate near 5% has about 2 percentage points of binomial standard error even before domain and precision refusals. The independent large-sample study conditions its rejection rates on fitted-domain eligibility. The rates below condition on both fitted-domain eligibility and numerical admission. Both studies report approximate conditional rejection at individual H values, before H selection.

## Boundary and numerical cases

- The boundary grid used n in {20, 60}, fitted k in {-0.8, 0.6}, H from 0 to 4 and both base seeds, giving 40 calls. The one refused calibration was n=20, k=-0.8, H=4, seed 20261132. Its relative variance half-width was 2.040%, above the unchanged 2% gate. Shapes one binary64 step outside each endpoint were refused at each n before simulating.
- With 20 identical discharges of 20, selection reports nonconstant-observation failure. With 19 such discharges and one at the next larger binary64 value, it reports numerical fit failure because the sample LH skew is 1. Both fail at H=0.
- The tied record `20 + floor(j/2)` for zero-based j from 0 to 19 refuses the entire domain. Fitted shapes for H=2, 3 and 4 are 0.623105, 0.715630 and 0.795166, respectively. Rounding the regular record below to the nearest quarter yields 15 distinct discharges, fits all five candidates and selects H=0 at 1.96. Ties alone do not determine eligibility.
- For the regular record `20 - log(-log((j - 0.35)/20.3))` for j from 1 to 20 and seed 7, the five signed z values are approximately -0.018068, -0.018483, -0.012861, -0.008357 and -0.004913. Cutoff 1.96 selects H=0, and 0.000001 yields a complete no-pass outcome. Setting the cutoff to the computed H=0 point magnitude deliberately straddles its numerical bracket and refuses selection. The point rule's inclusive equality is therefore distinct from admission of a decision whose z is uncertain.

## Held-out results

All 600 records were retained. All five fits completed on every record. There were 55 domain refusals and 545 eligible records. The latter exercised 5,450 direct candidate calibrations across the two seeds. Four calls exceeded the 2% variance half-width gate. All were n=20, generating k=-0.3, H=4 at base seed 20261128. Zero-based record indices 41, 42, 46 and 91 had half-widths 2.018%, 2.026%, 2.024% and 2.010%, respectively. Among the 2,721 candidates with calibration results at both seeds, the largest absolute point-variance difference divided by their mean was 0.788%. Every pair of reported variance brackets overlapped.

In the table, A is the admitted record count. V is variance-precision refusal. C is cutoff-bracket refusal. N is the no-pass count among admitted records. The five counts in parentheses give rejections at H=0, 1, 2, 3 and 4, each with denominator A. Paired is the count admitted at both seeds. Domain-ineligible records are excluded from both seeds' calibration attempts.

| n | Generating k | Domain outside | Cutoff | Seed 20261128 A/V/C/N (rejections) | Seed 20261129 A/V/C/N (rejections) | Paired | Selection changed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 20 | -0.3 | 6 | 1.96 | 87/4/3/0 (4,2,1,2,3) | 91/0/3/0 (4,3,1,2,3) | 87 | 0 |
| 20 | -0.3 | 6 | 2.5758293 | 89/4/1/0 (0,1,0,0,0) | 94/0/0/0 (0,1,0,1,1) | 89 | 0 |
| 20 | 0 | 8 | 1.96 | 88/0/4/0 (2,5,3,2,6) | 88/0/4/0 (2,5,3,2,6) | 88 | 0 |
| 20 | 0 | 8 | 2.5758293 | 92/0/0/0 (1,1,2,2,1) | 92/0/0/0 (1,1,2,2,1) | 92 | 0 |
| 20 | +0.3 | 34 | 1.96 | 66/0/0/0 (3,1,1,1,3) | 66/0/0/0 (3,1,1,1,3) | 66 | 0 |
| 20 | +0.3 | 34 | 2.5758293 | 66/0/0/0 (0,0,0,0,0) | 66/0/0/0 (0,0,0,0,0) | 66 | 0 |
| 60 | -0.3 | 1 | 1.96 | 98/0/1/1 (4,5,3,4,4) | 98/0/1/1 (4,5,3,4,4) | 98 | 0 |
| 60 | -0.3 | 1 | 2.5758293 | 99/0/0/0 (0,0,1,1,0) | 99/0/0/0 (0,0,1,1,0) | 99 | 0 |
| 60 | 0 | 1 | 1.96 | 99/0/0/0 (2,6,6,7,4) | 99/0/0/0 (2,6,6,7,4) | 99 | 0 |
| 60 | 0 | 1 | 2.5758293 | 99/0/0/0 (0,1,2,1,1) | 99/0/0/0 (0,1,2,1,1) | 99 | 0 |
| 60 | +0.3 | 5 | 1.96 | 94/0/1/0 (6,4,5,2,1) | 94/0/1/0 (6,4,5,2,1) | 94 | 0 |
| 60 | +0.3 | 5 | 2.5758293 | 93/0/2/0 (1,0,0,0,0) | 93/0/2/0 (1,0,0,0,0) | 93 | 0 |

Across both seeds, cells and H values, admitted-record rejection proportions ranged from 1.06% to 7.07% at 1.96 and 0% to 2.17% at 2.5758293. There were zero changed selections among 532 and 538 paired-admitted records, respectively. At 1.96 the first seed selected H=0, 1, 2, 3 and 4 in 511, 15, 4, 0 and 1 cases respectively, with one no-pass. The second seed's additional four admitted records all selected H=0. At 2.5758293, the first seed selected H=0 in 536 cases and H=1 in two cases. The second seed's additional five admitted records all selected H=0. The counts reflect the two particular deterministic calibration streams.

### Runtime replay

Replaying at base seed 20261128 and cutoff 1.96 through `evaluate_candidate_h` agreed with the independent scoring. The examples below use generating k=-0.3 and zero-based record indices.

- With n=20, index 0 reported domain refusal at H=3 and H=4. Index 2 reported cutoff-precision refusal at H=2 with an absolute-z bracket of about [1.95434, 1.97610]. Index 4 selected H=0.
- With n=60, index 92 reported domain refusal at H=4. Index 60 reported cutoff-precision refusal at H=4 with a bracket of about [1.95674, 1.98123]. Index 38 gave a stable no-pass with all five point values above 1.96. Index 0 selected H=0.
