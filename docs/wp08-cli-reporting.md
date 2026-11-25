## 24 November 2026 — executable fixed-H GEV mathematical CLI (partial)

**Current stage:** The installed `ffa fit` command implements a **mathematically specified deterministic** GEV calculation from exact source bytes to three report files. This supersedes older statements below that no fitting command is available. It does **not** enable Bayesian fitting, optimized-H selection, censored/zero/historical analyses or native Flike comparison. Ordinary family fits are described below.

```console
ffa fit --ams fixed-gev-ams.csv --settings fixed-gev-settings.json --output fixed-gev-report
ffa explain lh-moments-gev --format markdown
```

The installed wheel bundles an original `resources/examples/` CSV/settings pair, and the exact source copy remains in `docs/examples/`. The JSON settings format **`ffa-gev-fixed-h/1`** has seven mandatory keys and rejects unknown keys, duplicates, NaN/infinity, implicit choices and unsupported flags. They are `schema_version`, `estimator` (`gev_lh_fixed_h`), `lh_shift` (integer 0–4), `discharge_unit`, `year_definition`, `source_classification` (literal `caller_asserted_complete_exact_positive_annual_maxima`) and an ordered non-empty `non_exceedance_probabilities` list of finite values strictly within (0,1). No method/probability/units/year or vendor defaults exist.

The input CSV must use WP02's exact `Year,AMS` header and supply **only exact, strictly positive, nonmissing annual maxima**. Missing, zeros, unresolved flags, censored/grouped/historical data, insufficient/constant samples, unsupported files or overflow fail without deleting rows. Existing output directories are refused, so successful outputs never silently overwrite previous results.

The report records **actual source/settings SHA-256 bytes**, deterministic GEV location/scale/shape `point_estimate` fields rather than posterior parameter means, and `deterministic_fitted_quantile` with explicit F and AEP. The JSON and text files contain computed observation count, LH sample moments and the **unstandardized** fourth-ratio residual when available; that residual is **not** an optimized-H goodness-of-fit z statistic. The existing renderer remains explicitly **not legacy-qualified**. The CLI reports exit 0 for success, 2 for invalid inputs/schema/output collision, and 1 for IO or broken packaged resources. No native executable is accessed.

The scientific basis, exact method scope, limitations and a reproducible synthetic example are included in `ffa explain lh-moments-gev --format markdown` and `ffa references WANG1997`. **Independent mathematical/component tests and installed-wheel verification** establish only that bounded computation, not Flike equivalence. Native comparison/settings remains deferred to [development requirements](https://github.com/Chain-Frost/ffa-toolkit/blob/main/docs/development.md), while broader implementation continues in [development requirements](https://github.com/Chain-Frost/ffa-toolkit/blob/main/docs/development.md)/[development requirements](https://github.com/Chain-Frost/ffa-toolkit/blob/main/docs/development.md).

---

## Ordinary H=0 CLI extension

The same fit command accepts schema `ffa-ordinary-lmoments/1` for Gumbel, raw-flow LN2, log-flow LP3 and explicitly fixed/free-location GP annual maxima. See [installed guidance and worked examples](methods/l-moments.md). Ordinary reports retain the three existing output formats and additionally stage byte-exact source/settings copies. The existing fixed-H GEV schema and bundle are unchanged; fitted GEV support reporting is tracked separately in [issue #12](https://github.com/Chain-Frost/ffa-toolkit/issues/12). Neither workflow implies native qualification or uncertainty inference.

# WP08: CLI guidance and numerical report contract

**Status: partial implementation — not legacy-qualified.** This is the bounded independent part of [development requirements](https://github.com/Chain-Frost/ffa-toolkit/blob/main/docs/development.md). The [compatibility contract](compatibility-contract.md) and [user documentation contract](user-documentation-contract.md) remain authoritative. The target is the legacy **TUFLOW/BMT Flike 5.0.306.0 / file 3.10** behaviour, not later WMAwater features. No GUI, TUI, browser application, QGIS plugin or separately supported Python API is introduced.

## Installed CLI commands

~~~console
ffa --help
ffa --version
ffa explain
ffa explain lp3
ffa explain quantile-estimands --format json
ffa explain quantile-estimands --format markdown
ffa explain mgb-low-flow-screen --format markdown
ffa references ARR3
ffa references KUCZERA1999
ffa references
~~~

- **Offline technical pages:** `ffa explain METHOD_ID --format markdown` reads a bundled practical/technical page (`mgb-low-flow-screen`, `legacy-zero-threshold`, `censored-historical-likelihood`, `quantile-estimands`, `importance-sampling-reference`, `lh-moments-gev`) without a checkout or network. Other method entries currently provide register summaries only; requesting their Markdown page fails with exit 2 rather than pretending a repository-only document is installed. The pages distinguish scientific foundations, inspected target labels, currently implemented comparison components and unresolved native compatibility. No reference PDF or sample data is bundled.
- **explain** lists the bundled method-register entries, or shows the named method's purpose, assumptions, evidence gaps, verification requirements, primary/source keys and recorded implementation/qualification status. The JSON form has schema version **ffa-method-guidance/1**. The registered status is source metadata, not independent proof of a completed numerical implementation or a verified legacy comparison.
- **references** reads the actual packaged scientific/software reference register; an optional exact source key displays one citation entry. No online connection or working-tree checkout is required.
- **Exit codes:** 0 for successful help/version/guidance/reference lookup; 2 for invalid arguments, unknown IDs, absent commands and unimplemented analysis; 1 if the bundled reference data or register cannot be read or validated. **ffa fit** supports explicit-settings fixed-H GEV calculations on complete exact positive Year,AMS input; unsupported workflows fail explicitly. See [fixed-H installed example](wp04-fixed-h-gev-evidence.md).

**Reproducible documentation example:** In a clean Python 3.14 wheel installation, run **ffa explain quantile-estimands --format json**. Inspect the schema, the pinned compatibility profile, the method ID and recorded not-qualified status. Run **ffa references ARR3** to inspect the ARR 2019 Book 3 Chapter 2 section/equation index. This illustrates offline guidance, not flood fitting or a qualified worked numerical example.

## Internal result writer used by the installed fit command

**ffa_methods.reporting** consumes an already computed, typed **AnalysisResult** from the WP01 shared contracts and explicit run settings. It does not read or reinterpret source project files, fit parameters, construct a posterior, choose a default, adopt an estimate or validate source-run correspondence. This is an internal composition boundary for the installed CLI, not a separately supported public API. Files are staged in a sibling directory and published under a **new** output directory; existing output directories are rejected.

| Output file | Contents |
| --- | --- |
| **results.json** | Schema **ffa-analysis-report/1** with pinned compatibility profile, legacy program and file version, actual method/package versions, source-case evidence status, input and settings SHA-256 identities if supplied, run settings, method reference keys and their complete bundled citation entries, source-coordinate parameter summaries, result values, probability conventions and warnings. |
| **results.csv** | Original result order, method and method version, exact numerical values and units, explicit result meaning, supplied input probability and its kind, non-exceedance **F**, annual exceedance **AEP**, input discharge and reported probability level. Each CSV row also records unqualified formatting/conformance status and source-reference evidence status. Values are not silently resorted. |
| **report.txt** | Human-readable source, parameter, settings and result interpretation, explicitly labelled F/AEP, embedded source-register citation text and missing-provenance warnings. Display rounding is for readability; JSON/CSV preserve original Python float values. |

The same formatter keeps these distinct: (1) the distribution quantile evaluated at the **expected source-coordinate parameters**, not the posterior mean of quantiles; (2) **expected non-exceedance probability** at a specified discharge, where AEP for annual maxima is **1-F** and the corresponding annual return period is **1/AEP**, not **1/F**; (3) a named **Monte Carlo quantile probability limit**, not a generic frequentist confidence or simultaneous band; and (4) **posterior log10(discharge) moments**, which are distinct from LP3's natural-log-flow coordinates. Exact bound directions, vendor limit interpolation rules and all uncertainty modes remain unqualified.

The writer also requires **typed, sufficient metadata for every reported estimand**: an F/AEP input for quantiles, an explicitly unit-labelled input discharge (but no input probability) for expected F, and a unit `probability` for expected F. A probability limit requires a level, `lower`/`upper` side and construction descriptor; log10 moment outputs require `mean`, `standard_deviation` or `variance`. These are supplied labels, **not independent verification of the interval or posterior computation**. Every file serialises the unit, limit side/construction and moment kind; missing or incompatible combinations raise `ValueError`. The extra shared `ResultPoint` annotations are optional for other consumers, but mandatory when the corresponding result is exported. No unknown default is imputed.

The writer rejects nonfinite results, impossible probabilities and levels, missing probability kinds, malformed input/settings SHA-256 strings, unsupported method IDs, cross-profile identity, a mismatching installed package version and empty numerical results. It flags absent input or settings hashes, and missing run settings. It never asserts mathematical correctness or engineering suitability. **Both "method_qualification" and "legacy_comparison_status" remain "not_established_by_formatter"**, even for a supplied provenance label indicating a qualified reference case. Reference case admission and solver conformance are separate questions. Recorded SHA-256 strings are not independently authenticated against input files here.

### Synthetic formatting example — explicitly not a Flike benchmark

The focused tests format an artificial discharge quantile of **100 m³/s** at **F=0.8**, with interpreted **AEP≈0.2**. A separate artificial point presents **AEP=0.2** as the supplied input and labels its complement **F=0.8**. The test includes an LP3 natural-log parameter coordinate, a base-10 output moment and an explicit 90% Monte Carlo probability-level field. These values test labelling and serialization only; they are **not** vendor reference values and do not constitute a worked example for a qualified statistical method.

## Historical/censored plotting is explicitly deferred

No empirical plotting positions or frequency plots are generated by this partial scope. The inspected legacy heading **"AEP plot position"** can accompany quantities consistent with non-exceedance **F**, not an AEP; the label and interpreted probability kind must be stored separately. ARR Book 3 Chapter 2 §2.6.2, equations 3.2.35–3.2.36, is the initial mathematical reference. A future plotting module must retain historical perception periods, known non-exceedances, censor thresholds, excluded observations, unknown-year exposure and the source labels. **Deleting censored or historical entries and re-ranking the reduced sample is not an acceptable legacy probability plot.** No such plot is currently represented as qualified.

## Primary foundations and provenance distinctions

See [the maintained reference register](references.md) and [method register](method-registry.json). Source keys include ARR3, KUCZERA1999, HOSKING1990, WANG1997, WANG1998, COHN2013 and KUCZERA1996. ARR 2019 Book 3 version 4.2 Chapter 2 §2.5 covers distinct design estimands; §2.6.2 covers plotting; §§2.6.3.6–2.6.3.7 and equations 3.2.52–3.2.56 cover Monte Carlo inference and reported uncertainty. Kuczera (1999), *Comprehensive at-site flood frequency analysis using Monte Carlo Bayesian inference*, [DOI 10.1029/1999WR900012](https://doi.org/10.1029/1999WR900012), is a primary methodological source. The citations support foundations but cannot establish the exact implementation/settings of Flike 5.0.306.0.

## Initial mathematical/software work — issues #11 and #40

1. Connect reviewed WP02 source adapters to explicit, sufficiently specified mathematical configurations. Preserve unknown legacy controls and reject calculations that actually depend on them. Existing-project/report pairing is unnecessary for an independent calculation using explicitly validated inputs/settings; it remains necessary for a native comparison claim.
2. Integrate independently tested WP04 fits, WP05 densities/rating models, actual WP06 sampling and supported WP07 observation treatment as required by each supported workflow. A bounded deterministic L/LH path can proceed before the Bayesian sampler. Their mathematical/software acceptance is the dependency; native qualification is not. A bridge that formats already calculated outputs does not implement fitting or orchestration.
3. Implement the supported file-to-calculation-to-report CLI, reproducibility/diagnostic metadata, scientific plotting only where its observation/position specification is sufficient, worked independent examples and offline explanations. Preserve distinct estimands and failure status. Do not infer historical plotting by re-ranking a reduced sample.
4. Pass Ruff, strict Pyright, Python 3.14 Ubuntu/Windows CI, build/resource/installed-workflow checks and independent review of the final functional head. Initial issue #11 can close when its agreed software scope is complete and the native remainder is linked separately. Publication is not authorised by these checks.

## Deferred native evidence — issue #39

Unresolved native default/control mappings, exact legacy plotting/display conventions and settings-matched end-to-end Flike comparisons remain in #39, with final conformance reviewed in #12. These do not block an explicit-configuration mathematical CLI or bounded reporting PR. See [development requirements](https://github.com/Chain-Frost/ffa-toolkit/blob/main/docs/development.md) and [development requirements](https://github.com/Chain-Frost/ffa-toolkit/blob/main/docs/development.md).

## WP06 weighted-output bridge — bounded integration (partial)

The internal `ffa_methods.wp06_report_bridge.wp06_output_analysis_result` accepts **already computed** WP06 `QuantileOutputs` and `ProbabilityOutputs`, source-coordinate `ParameterSummary` objects, actual `RunProvenance` and explicit discharge units. It **does not** read source files, choose settings, evaluate priors, sample an importance proposal, fit a distribution or enable `ffa fit`. The originating method version and evidence identity are supplied, not inferred; native Flike qualification remains `not_established_by_formatter`.

This bridge adds distinct typed meanings for **posterior mean and median quantiles**, the **inverse expected-CDF/predictive quantile**, direct **expected AEP** from WP06 survival probabilities, the **reciprocal of expected AEP return period**, and **pointwise limits of per-draw return periods**. It does not silently conflate these with the quantile at expected source parameters or with expected non-exceedance `F`.

WP06's probability limits are labelled `WP06 left-continuous weighted empirical inverse CDF; Flike interpolation unverified`; the central interval mass, side and target quantity appear in JSON/CSV/text. The mean/SD of `log10(discharge)` are reported only where the upstream weighted quantities are defined. If a positive-weight quantile is nonpositive, the undefined base-10 moments are **omitted with an explicit warning**; they are never substituted with zero. Infinite return periods or limits likewise remain **explicit warnings**, not invented finite sentinels. Missing values and numerical precision are not disguised as successful numerical results. Weighted tail-resolution ESS is provided as a warning and explicitly **not** claimed to be a Monte Carlo standard error.

The bridge checks F/AEP complement consistency, unit/probability domains, central interval mass, limit order and the reciprocal return-period identity. Independent synthetic WP06 calculations exercise the actual producer → bridge → `report_document`/CSV/text pipeline. These examples are **mathematical reporting tests**, not replayable native Flike benchmarks. Source-file orchestration requires reviewed inputs and a sufficiently specified mathematical execution path. Explicit-configuration software integration is active under #40 and does not wait for #39 native control/default evidence; an actual Bayesian fit still requires the WP06 sampling engine.

### Signed mathematical quantiles versus physical discharge

Independent review found that a valid, **untruncated** Gumbel/GEV posterior can yield a **negative lower quantile probability limit** even when the expected-parameter, posterior mean, median and predictive quantiles are positive. WP06 correctly returns the signed mathematical result. The bridge and formatter preserve a **finite negative modelled quantile** and emit a clear nonphysical-discharge warning; they do **not** clip it, discard all other estimates, or fabricate a log10 moment. In contrast, **fixed input discharge** for probability/return-period-at-flow remains constrained nonnegative, with units. A synthetic two-component Gumbel regression specifically protects this distinction. The analyst must evaluate physical support and suitability before adopting a discharge estimate; preservation in an unqualified report is not endorsement of negative flows.
