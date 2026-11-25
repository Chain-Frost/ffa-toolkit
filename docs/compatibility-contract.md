# Legacy Flike compatibility contract

## Target and authority

This contract defines the **later native-replication claim** for the existing **TUFLOW/BMT-supplied Flike software**, delivered through a **Python CLI only**. The owner has separated initial mathematical/software delivery [development requirements](https://github.com/Chain-Frost/ffa-toolkit/blob/main/docs/development.md) from unavailable native evidence/comparison [development requirements](https://github.com/Chain-Frost/ffa-toolkit/blob/main/docs/development.md). The summary below is included in the installed contract; the full [development requirements](https://github.com/Chain-Frost/ffa-toolkit/blob/main/docs/development.md) is repository guidance. Final native conformance remains WP09 [development requirements](https://github.com/Chain-Frost/ffa-toolkit/blob/main/docs/development.md).

Mathematical components and explicit-configuration CLI workflows can be completed, reviewed and merged without satisfying this native contract. Do not invent unidentified native defaults or relabel an independent method as equivalent Flike behaviour. The sequencing decision does not waive mathematical correctness, missing implementation or the eventual supported-combination comparison requirements.

The uploaded report evidence identifies **program version 5.0.306.0** and **file version 3.10**. This is the initial compatibility profile, `tuflow-bmt-flike-5.0.306.0-file3.10`. It does not establish custody of a native executable, every default, or that each saved project produced a same-named report. Those are separate evidence fields. See [sample-data-audit.md](sample-data-audit.md).

Replication means the same admitted statistical behaviour and numerical outputs within justified deterministic, rounding and stochastic tolerances. It does not mean identical random draws, copied proprietary source, GUI layout reproduction or byte-identical report formatting. Original Python code and original explanations implement and document the specified mathematics.

ARR and primary literature provide scientific foundations. Legacy reports/configurations establish target behaviour. If evidence exposes a discrepancy between the target and the mathematics, retain the discrepancy and resolve its treatment explicitly; do not silently change the calculation or describe reproduced defects as scientifically endorsed.

## Final native-replication feature matrix

Every row requires status for evidence, implementation, independent verification and legacy comparison. Supported combinations are enumerated; listing a family does not qualify every option combination.

| Required legacy capability | Initial evidence and remaining need | Owner |
| --- | --- | --- |
| LN, LP3, Gumbel, GEV and GP probabilities/quantiles | Public specifications; sample reports exercise LP3 and GEV, not the full matrix | WP03 |
| L-moment fitting for legacy-supported families | ARR legacy examples; exact target-build reference cases still needed | WP04 |
| GEV LH H=1â€“4 and optimized H | Published legacy workflow; numerical/selection evidence still needed | WP04 |
| Bayesian importance sampling | Sample reports and published legacy description; solver/default/repeatability evidence to pin | WP06 |
| Exact, historical and censored observations | Samples include differing excluded-row/grouped-censor representations; complete correspondence still under review | WP02/WP05/WP07 |
| Supplied/regional and no-prior modes | Source and project/report evidence; preserve actual target defaults and coordinates | WP05 |
| LN/LP3 zero threshold | Reports expose logit probability of flow at/below threshold; full estimation/likelihood/default semantics remain to resolve | WP07 |
| Rating-error options | Legacy capability is required; admitted example runs and exact grouping/model evidence needed | WP05 |
| Project/configuration and result interpretation | File3.10 input and versioned reports available; preserve unknown fields | WP02 |
| Legacy quantile/probability outputs and plotting conventions | Reports include expected-parameter results, 90% probability limits and expected-probability/return-period tables; confirm raw non-exceedance versus AEP conventions | WP06/WP08 |
| CLI installation, replay, outputs and method guidance | New delivery layer implementing the same admitted statistical contract | WP01/WP08/WP09 |

The initial mathematical/software milestone has its own acceptance in #40. Missing native rows remain in #39/#12 and do not block that milestone or bounded implementation merges. An initial milestone must describe its supported scope and unqualified native status; it must not be presented as full Flike replication. Publication remains separately authorised.

## Parameter and method fidelity

The observed LP3 report coordinates are:

1. Mean of natural-log flow.
2. Natural logarithm of the standard deviation of natural-log flow.
3. Skew of natural-log flow.
4. Where present, logit of the probability that flow is at or below the zero threshold.

Preserve these coordinates for legacy priors, posterior summaries and expected-parameter output. Internal conversion to a library's standard deviation is permitted, but averaging internal parameters instead of source coordinates can change results. Priors must preserve their full joint density through nonlinear transformations.

Natural-log parameter coordinates are distinct from the reported moments of `log10(q)`. In the supplied reports, the expected-probability column is numerically consistent with non-exceedance probability `F` and return period `1/(1-F)`. The raw column labelled `AEP plot position` also contains values consistent with non-exceedance plotting positions. Preserve raw labels and verify their mathematical meaning; any user-facing AEP value must use the established exceedance convention rather than relabelling `F` directly.

These observed labels do not, by themselves, prove all mathematical or default details. In particular, the zero-threshold probability must not be equated automatically with a newly introduced literal no-flow mixture. Resolve threshold definitions, counts, fitted/fixed parameters, prior contribution and uncertainty from reference evidence.

The eventual replication targets the legacy importance-sampling procedure. Initial development may implement source-reviewed importance-sampling mathematics under explicit documented settings, with independent verification and an unqualified native status. The exact vendor proposal, RNG and stopping/default choices are separately evidenced in #39. Independent MCMC or quadrature may verify the same model; a production MCMC replacement remains future scope. New default priors, mixture components, climate adjustments or estimator-selection policies do not enter the compatibility profile silently.

## Reference-case admission

Keep originals immutable and retain archive/member/file hashes. A staged derivative records its original hash, derivative hash, exact transformation and purpose. Redacting an unrelated client path is different from changing a numeric record or analysis setting.

Use these evidence statuses:

- **Observed artifact:** parsed contents and version are established, but relationships may be unknown.
- **Candidate pair:** data/settings agree to the inspected extent, but exact run correspondence is not yet established.
- **Verified pair:** source/configuration/version and report correspondence are established with a reproducible provenance record.
- **Mismatched or unresolved:** an observed conflict or missing evidence prevents conformance use.
- **Qualified reference:** a verified pair with output semantics and appropriate comparison precision approved for its stated scope.

The supplied examples currently supply observations and candidates, not a complete qualification corpus. Do not infer full coverage from their count. Preserve stale/generic reports separately and do not overwrite original project switches to make a desired pairing appear valid.

## Numerical acceptance

Define deterministic tolerances from independent numerical references and conditioning. Define report comparisons from printed precision and exact output semantics. Define stochastic criteria after assessing reference repeatability and Monte Carlo error. Keep all three distinct from uncertainty caused by short records or model assumptions.

Pin source/build/settings and classify differences before changing code. Comparison inputs include observation flags, thresholds/count groups, priors, parameter order, fit method, H, rating-error model and output options. Matching years and discharges alone is insufficient.

## Future scope and excluded interfaces

WMAwater's new TCEV/covariate/uncertainty/solver features, new production MCMC, additional regional model-fitting and any unverified end-to-end POT workflow are future work. Generalised Pareto distribution support is already part of the legacy family matrix; the complete POT event/exposure workflow is a separate capability.

There is no GUI scope at any phase. Do not create a GUI, TUI, QGIS plugin, browser application or separately supported public API. CLI-generated plots, numerical files and static explanatory HTML remain valid outputs. A static report is not permission to build an application interface.

See [development requirements](https://github.com/Chain-Frost/ffa-toolkit/blob/main/docs/development.md), [development requirements](https://github.com/Chain-Frost/ffa-toolkit/blob/main/docs/development.md) and [the work-package plan](implementation-plan.md). Updated 24 November 2026.
