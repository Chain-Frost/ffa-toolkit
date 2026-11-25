# WP01 shared composition contracts

**Status:** initial interfaces for parallel WP02 and WP03 integration. Nothing in this file is a qualified Flike numerical implementation. The owning [compatibility contract](compatibility-contract.md) and [sample audit](sample-data-audit.md) have priority over guessed conventions.

## Stable module boundaries

| Boundary | Module / owner | Contract |
| --- | --- | --- |
| Package identity and help-only CLI | `ffa_methods.cli` / WP01, later WP08 | `ffa --help`, `ffa --version` succeed. All calculation attempts fail explicitly until implemented. No GUI, TUI, public API or fitting placeholder |
| Legacy target identity | `ffa_methods.compatibility` / WP01 | `LEGACY_FLIKE_PROFILE.identifier` = `tuflow-bmt-flike-5.0.306.0-file3.10`; stores observed vendor and file versions separately from `ffa_methods.__version__` |
| Observation and grouped exposure | `ffa_methods.contracts` / WP01 for interface, WP02 for parsers | `AnnualObservation`, `CensorGroup`, `ObservationSeries`, `ExposureKind`, `ObservationKind`, `EvidenceStatus`; finite measurements, explicit year/unit/record flags and separate bound/count fields |
| Probability and legacy parameter identity | `ffa_methods.contracts` / WP01, WP03 for transforms | `ProbabilityKind`, `ParameterCoordinate`; LP3 source coordinates mean of ln(flow), ln[SD(ln(flow))], skew of ln(flow) and optional logit P(flow at/below the zero threshold) |
| Distribution evaluation | `ffa_methods.contracts.DistributionKernel` / WP03 | Bound-parameter instance methods `logpdf`, `logcdf`, `logsf`, `cdf`, `sf`, `ppf` and `isf`, accepting discharge/returning floats. Parameter validation remains WP03 responsibility. A prediction outside a finite support must still return mathematically correct CDF/SF |
| Result/provenance | `ffa_methods.contracts` / WP01, WP06/WP08 consumers | `RunProvenance`, `ParameterSummary`, `ResultPoint`, `AnalysisResult`, `ResultMeaning`; source profile, method version, package version and evidence status distinct |

The contracts are typed, frozen and intentionally narrow. They are **not** a user-facing public Python API. Do not modify these modules simultaneously from parallel WP02/WP03 branches: propose compatible updates to WP01, land one accepted change, then rebase. Specific numerical method modules should live separately from `contracts.py`. The CLI owns filesystem paths and process exits; distributions and inference do not exit, write files or draw from implicit global random state.

## Observation fidelity and unresolved fields

`AnnualObservation.kind` distinguishes exact, left/right/interval censored, at/below zero threshold, excluded and missing. A zero-valued `EXACT` observation remains zero; `EXCLUDED` may retain its original value and source flag. `CensorGroup` represents a reported grouped threshold and counts, not a list of dropped observations. WP02 must keep uninterpreted file3.10 flags/controls as raw information alongside typed observations. `source_flag` deliberately remains text; the meaning of flag 0/1 or other codes must be established independently.

The threshold group's `count_at_or_below` label is a neutral storage key and **not** a verified statement of source endpoint inclusion; preserve original `source_record` and qualify the actual legacy comparator. WP02/WP05 must reconcile grouped exposures against annual records and history to prevent double counting. `ExposureKind.UNRESOLVED` is available where sample provenance cannot establish systematic versus historical years. `ObservationSeries.zero_threshold` is a measured/report threshold, **not** a licence to assume a fixed zero-inflation model or Beta prior. Quantitative data validity and model-specific positive support are later numerical validation concerns; initial parsers must not silently modify observed evidence.

## Parameter, probability and results

The observed LP3 parameter report has ordered coordinates: `LP3_MEAN_LOGE_FLOW`, `LP3_LOGE_SD_LOGE_FLOW`, `LP3_SKEW_LOGE_FLOW` and, where active, `ZERO_THRESHOLD_LOGIT_PROBABILITY`. Preserve this order for prior densities, posterior summaries and expected-parameter quantiles. WP03 may convert to SciPy for computation, but changes of variable require consistent priors/Jacobians. GEV and GP shape-sign conventions remain unqualified until WP03 verifies original legacy definitions.

`ProbabilityKind.NON_EXCEEDANCE` stores `F`; `ProbabilityKind.ANNUAL_EXCEEDANCE` stores `1-F`. Historical source headings such as `AEP plot position` do not override the interpreted numeric convention. Do not collapse quantiles evaluated at posterior expected parameters, averaged probabilities at a fixed discharge, Monte Carlo quantile limits or log10 quantile posterior moments into one `result` label. `ResultMeaning` distinguishes the meanings; `ResultPoint` carries its input probability or discharge and units.

`RunProvenance` separates method version, package version, reference-case evidence status and exact inputs/settings hashes. The profile identifies the observed **vendor** software, not a claim that its results are reproduced. WP01 does not create `AnalysisResult` values; only a later qualified analysis can report fitted estimates.

## Versioned offline method guidance

`docs/method-registry.json`, `docs/references.md` and `docs/compatibility-contract.md` remain the single authoritative files. The wheel builds **byte-for-byte** copies under `ffa_methods/resources/` using Hatchling force-include; a verifier fails on mismatches and missing resources. The sdist includes only the needed source and these documents. The six candidate reference files and the original archives are excluded from both. The register distinguishes implemented mathematical components from the incomplete full programme; no method is native-qualified. The fixed-H GEV installed workflow is available.

## WP08 report metadata annotations (partial)

`ResultPoint` now carries **optional, backward-compatible annotations** needed to avoid mislabelling downstream report exports: `input_discharge_unit` for expected F at input discharge; `probability_limit_side` and `probability_limit_construction` with `confidence_level` for a quantile probability limit; `log10_moment_kind` for mean/standard-deviation/variance of log10 discharge. None establishes legacy settings or an interval algorithm. WP06 and WP08 must pass *actual* target definitions through these fields; the formatter fails closed for semantically incomplete points. The contract is internal to the CLI, not a separately supported Python API. See `docs/wp08-cli-reporting.md` and WP06 issue #9.

## WP06-to-WP08 weighted output distinctions

The internal `ResultMeaning` enumeration now distinguishes posterior mean/median/predictive quantiles, expected F versus direct expected AEP at fixed discharge, reciprocal-expected-AEP return period, and per-parameter return-period limits. These are **new output identities**, not new numerical estimators or native Flike defaults. `ResultPoint` retains its existing probability-kind/discharge-unit/interval metadata. The WP08 adapter accepts upstream-computed WP06 results and preserves their mathematical identities without introducing a supported public Python API or an unqualified fit command. Infinite limits and undefined log10 moments remain explicit omissions with warnings rather than fabricated finite values.
