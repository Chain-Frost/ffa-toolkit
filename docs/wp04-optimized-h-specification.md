# WP04: optimized-H GEV selection — evidence and acceptance specification

**Status: specification only; NO automatic H selector is implemented.** The reviewed fixed-H GEV core remains explicit H=0..4. Its fourth-moment residual is an *unstandardized* ratio difference and cannot be treated as Wang's z statistic. [development requirements](https://github.com/Chain-Frost/ffa-toolkit/blob/main/docs/development.md) stays open.

## Verified primary mathematical requirement

ARR 2019 Book 3 v4.2 sections 2.6.4.5 and 2.8.10.3–2.8.10.4 and Wang (1997, 1998) establish:

1. H=0 is ordinary L-moments for every family; H=1,2,3,4 can be selected only with GEV.
2. For each candidate H, compute sample LH1–LH4. Fit GEV location/scale/kappa by matching LH1–LH3. The fourth LH-moment is a separate goodness-of-fit diagnostic.
3. Define sample t4hat(H)=LH4hat/LH2hat and fitted t4(H)=LH4_GEV/LH2_GEV, and let t3hat(H)=LH3hat/LH2hat. Equation 3.2.78 defines the approximate statistic:

   z(H) = [t4hat(H)-t4(H)] / sigma[t4hat(H) | t3hat(H)=t3(H)].

4. The denominator is the **conditional standard deviation**, incorporating H, sample size and GEV skew/shape dependence. It is NOT raw LH scale, the sample SD, a bootstrap confidence limit or an arbitrary common variance. Wang (1998) describes a simple variance approximation, but its exact formula and conditions have **not yet been independently verified** in this implementation.
5. Under a fitted-GEV null, z is approximately standard Normal. That alone does not specify the target executable's critical level, search direction, ordered selection/tie policy or behavior when all candidates fail.
6. ARR example 10 (Albert River at Broomfleet) reports an optimized choice **H=4**. This is a worked illustration, NOT a qualified numeric fixture for Flike 5.0.306.0; ARR screenshots show Flike 5.0.220.0.

## Active mathematical prerequisites — issues #7 and #40

| Mathematical requirement | Current position |
| --- | --- |
| Candidate H and family | H=0–4 for GEV in this selection scope; fixed-H code is already tested |
| Fourth sample moment | Requires n>=H+4, positive LH2 and a representable fit; H=4 requires n>=8 |
| Conditional variance and its estimator | Acquire the sufficient Wang (1998) relation, assumptions and numerical evaluation; independently verify it before coding the standardized diagnostic |
| Defined selection policy | Inspect the published decision procedure and specify its controls explicitly; do not invent a vendor alpha/search/tie/failure default |
| Observable results | Preserve each candidate H, input identity, sample/fitted tau3/tau4, conditional sigma, standardized z, decision reason and explicit settings |

Acquire and independently verify the original conditional variance, then derive standardized candidate-H diagnostics without changing fixed-H estimator meanings. A source-reviewed mathematical selection procedure with explicit controls can be implemented and tested without the native executable. A raw fourth-moment mismatch or selection of the largest H is not a substitute. If the mathematical specification itself remains insufficient, keep that precise source gap active in #7/#40 rather than classify it as native-testing delay.

Tests must cover independently calculated sigma/z, small/constant samples, minimum sizes, numerical/admissibility failures, and the declared selection/uncertainty procedure. Define tolerances for the asserted quantity and reference precision. Keep F distinct from AEP=1-F. Obtain independent review and normal quality/integration checks before declaring this mathematical scope complete.

## Deferred native selection and comparison — issue #39

The exact Flike test sidedness/critical level, search order, tie/failure/no-acceptable-H policy and bootstrap reselection/RNG/count/interpolation are unknown native choices. Record them in #39 and keep them unsupported rather than guessing. Settings-matched Flike 5.0.306.0 results for fixed and optimized H, including failure cases, belong to later native qualification in #12.

These native requirements do not block the independent standardized diagnostic or a sufficiently specified mathematical selector. The ARR H=4 illustration is retained as source evidence without being promoted into a qualified target-build run. See [development requirements](https://github.com/Chain-Frost/ffa-toolkit/blob/main/docs/development.md) and [development requirements](https://github.com/Chain-Frost/ffa-toolkit/blob/main/docs/development.md).

## References

- [ARR 2019 Book 3 v4.2, eq. 3.2.78, sections 2.6.4.5 and 2.8.10](https://www.arr-software.org/pdfs/ARR_190514_Book3_V4.2.pdf).
- [Wang 1997, LH moments](https://doi.org/10.1029/97WR02134).
- [Wang 1998, fitted-GEV LH goodness-of-fit z](https://doi.org/10.1029/98WR02364).

No third-party code is ported. Repository license remains Sustainable Use License 1.0. The mathematical specification alone does not qualify a native compatibility claim.
