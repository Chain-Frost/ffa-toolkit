# User documentation and report contract

## Purpose

An engineer must be able to understand the statistical basis and limitations of an analysis without inspecting Python source. Documentation is therefore part of each method's acceptance criteria. The same versioned method register should supply packaged method metadata, CLI explanations and analysis reports.

This document specifies the complete interface/documentation requirements. Some guidance and report components are implemented; inspect current CLI help and [WP08 guidance](wp08-cli-reporting.md) for availability. Examples of planned commands or layouts do not establish implemented functionality.

Initial mathematical/software delivery #40 and deferred native evidence/comparison #39 have separate acceptance under [development stages](development-stages.md). The eventual **TUFLOW/BMT Flike 5.0.306.0 / file 3.10** claim remains governed by the [compatibility contract](compatibility-contract.md). Each implemented mathematical calculation needs accessible guidance now: purpose, assumptions, equations/coordinates, explicit settings, uncertainty, limitations and exact sources. Required coverage includes the five families, L/LH methods, importance sampling, historical/censored information, low flows/thresholds, priors, rating models and outputs as implemented. Display mathematical implementation/verification separately from native qualification; unavailable native evidence must not make an implemented component appear absent.

The CLI is the supported user interface. Internal Python functions serve composition and testing; there is no separately supported public API. GUI, TUI, desktop, QGIS and browser application interfaces are excluded at every phase. Static figures, reports and offline method documents are permitted CLI outputs. WMAwater additions and improved production methods remain future scope.

## Three levels of explanation

### Practical explanation

Every selectable method begins with a concise explanation of the question it answers, when it is appropriate and the evidence it needs. Define specialist terms on first use. For example, explain a potentially influential low flow as an annual peak whose magnitude can strongly affect the fitted upper tail, rather than assuming the user understands the acronym PILF.

The explanation should identify the actual legacy option or output. A user reading an expected-parameter curve needs to understand that the distribution is evaluated at posterior mean parameters in the specified coordinates. A user reading an expected-probability table needs to understand that probabilities at a supplied flow are averaged over parameter uncertainty. State whether the column is non-exceedance or exceedance probability; a sampler name alone does not explain the design quantity. Do not invent a new estimator-selection policy while documenting the legacy defaults.

### Technical method page

Each page contains the exact mathematical specification and parameter conventions, assumptions, input schema, supported combinations, numerical procedure, diagnostics, uncertainty construction and known limitations. Include ARR book/chapter/section/equation references where applicable, and primary literature for the method. State whether each rule comes from guidance, published mathematics, observed target behaviour or a project implementation decision. Separate documented foundations from unresolved target defaults.

Describe production importance sampling, legacy no-prior/supplied-prior behaviour and the zero-threshold model according to their actual implementation and qualification status. Planned or unresolved behaviour must remain labelled as such. Independent MCMC or quadrature may provide verification evidence; it must not be presented as a selectable replacement MVP sampler. A newly chosen prior or literal-zero mixture must not be described as legacy behaviour merely because it is statistically plausible.

Provide a reproducible worked example with input data, configuration, expected outputs and an explanation of the decisions. A screenshot alone is insufficient. A user should be able to rerun the example and distinguish numerical rounding from a changed method.

### Analysis evidence

Every report identifies the exact method, algorithm and compatibility-profile versions used, the qualification status of that combination, and links to the corresponding method pages. Include the settings and data decisions for that particular run. Static background documentation cannot explain an undocumented manual threshold or prior override. Distinguish mathematical verification, successful execution and legacy comparison; these are different claims.

## Mandatory method-page fields

| Field | Required content |
| --- | --- |
| Identity and status | Stable method ID, implementation version and qualified observation/estimator combinations |
| Compatibility target | Legacy program/file versions, source controls/defaults, reference cases and unresolved behaviour |
| Engineering purpose | What decision or quantity the method supports |
| When to use | Required evidence, applicable process and practical suitability |
| Assumptions | Independence, stationarity, measurement and observation assumptions as relevant |
| Inputs | Units, transformations, supplied/no-prior mode, thresholds, dates, rating-error controls and required provenance |
| Mathematics | Equations, probability measure, parameter names and supported domain |
| Numerical procedure | Actual algorithm, convergence/precision requirements and failure handling |
| Outputs | Estimand, interval construction/level and numerical error |
| Interpretation | What the result means and which decisions remain with the analyst |
| Limitations | Unsupported combinations and uncertainty sources not represented |
| Sources | Exact ARR sections and primary references; access and licence notes where relevant |
| Verification | Named fixtures, independent checks, legacy comparison evidence and current qualification status |

The initial [method register](method-registry.json) contains planning entries. It is a seed for this structure, not a complete implemented method catalogue or final schema.

## Source conventions users must be able to inspect

The [sample-data audit](sample-data-audit.md) supplies initial report evidence, with input/output settings still awaiting complete qualification. The user documentation and machine-readable results must retain these distinctions:

- **LP3 parameters:** mean of natural-log flow, natural log of the standard deviation of natural-log flow, and skew of natural-log flow. Where active, the fourth coordinate is the logit of probability that flow is at or below the zero threshold. Display source names and explain the conversions to any internal coordinates.
- **Parameter means and priors:** compute the legacy expected-parameter result in the specified source coordinates. Mean standard deviation and the exponential of mean log-standard-deviation are different quantities; nonlinear coordinate changes also affect the full prior density.
- **Output logarithms and limits:** the samples use natural logarithms for LP3 parameter labels and base-10 logarithms for moments of `q`. They show Monte Carlo 90% probability limits. Identify each target, base, level and construction rather than labelling every interval a generic confidence band.
- **Reported probabilities:** the printed expected-probability column is numerically consistent with non-exceedance `F`, paired with return period `1/(1-F)`. Confirm its meaning before exposing AEP as `1-F`. Probability at a specified flow and a quantile obtained by inversion remain different outputs.
- **Plotting labels:** the samples' raw `AEP plot position` label also accompanies values numerically consistent with non-exceedance probability. Retain the source label for traceability and store the interpreted probability kind separately. User-facing axes and canonical output labels must reflect the resolved semantics, with a note explaining any discrepancy in the legacy label.
- **Evidence limits:** a candidate file pair with matching years and flows is not a qualified inference benchmark. Record unresolved priors, flags, thresholds and run controls visibly in the method's evidence record.

The [statistical method specification](statistical-methods.md) gives the equations and supporting references. CLI explanations should make these distinctions understandable without forcing users to read a parser or numerical-library implementation.

## Worked example of a user-facing method explanation

### Potentially influential low flows

**Purpose.** Examine whether very small annual peaks are exerting a disproportionate influence on estimates of larger floods. The method is useful where low annual peaks may arise from a different part of the flood-generating process and the fitted upper tail is sensitive to them.

**Evidence and action.** Run the qualified legacy multiple Grubbs-Beck procedure, show the candidate threshold and affected years, and compare the original and censored fits where that combination is supported. The engineer checks the plot and hydrological context before recording the adopted treatment through the CLI configuration. Censoring preserves the fact that those years had peaks below the threshold; it does not remove the years from the record.

**Limitations.** A flagged value is not automatically an erroneous observation. The test does not justify removal of inconvenient high floods, prove stationarity, or resolve a poor rating curve. A missing year cannot be converted into a censored year without evidence of its bound. The exact test variant and its settings are part of the method identity.

**References.** [ARR Book 3 Chapter 2 section 2.6.3.9](https://www.arr-software.org/pdfs/ARR_190514_Book3_V4.2.pdf); [Cohn et al. (2013), DOI 10.1002/wrcr.20392](https://doi.org/10.1002/wrcr.20392). The explanation above is original project guidance; the numerical algorithm must be pinned and qualified before this becomes a selectable method.

The complete worked example added during implementation must include the original annual series, reference classifications, threshold/count representation, analysis configuration and before/after outputs. Placeholder results must never be presented as a successful calculation.

## Analysis-report contents

1. **Analysis and adopted estimate.** Identify site, purpose, units, reference conditions, requested outputs, selected model and adopted estimand. Preserve the actual legacy defaults and identify any explicit option changes. Show alternatives actually examined where they materially change interpretation.
2. **Evidence used.** Show systematic record coverage, missing years, true zeros, zero-threshold counts, censored records, historical periods and thresholds. Include a timeline so long historical exposure is not confused with continuous measured observations. Preserve original data and their interpretation separately.
3. **Method and justification.** Explain the distribution, estimator and observation model, including the qualified family/zero-treatment combination and enabled/disabled rating-error model. Record prior origin, applicability, original parameter coordinates, effective no-prior setting where selected, and manual changes with the analyst's reason.
4. **Results and uncertainty.** Display legacy quantile and probability-at-flow tables and frequency curves with explicit estimands, interval targets and levels. Preserve log bases, non-exceedance/exceedance labels, source coordinates and displayed precision. Identify pointwise rather than simultaneous bands. Separate parameter uncertainty, numerical integration error and omitted uncertainty sources.
5. **Diagnostics and sensitivity.** Show fit adequacy, inference failures, rare-tail precision, priors and key threshold/distribution sensitivities. Identify the plotting convention and how censored/historical records are displayed; never disguise reduced-record re-ranking as an adjusted historical probability plot. A clean optimiser exit does not automatically make the analysis suitable.
6. **References and replay information.** Include relevant method pages, readable bibliographic citations, data/configuration hashes, compatibility profile, legacy program/file versions, package/method versions, dependency versions and random-stream information. A target build label does not establish that a run has passed legacy comparison.

The report must distinguish the status of a numerical run from the suitability of its scientific model and from the engineer's adoption decision. These should be understandable labels, not an unexplained generic pass/fail indicator.

## Warnings that explain an engineering consequence

Warnings must identify a specific condition, its effect on interpretation and a useful next action. Examples of proposed wording are:

- **Historical coverage unresolved:** the historical period overlaps the gauged years. Reconcile which annual peaks are represented before fitting, because the current inputs would count some evidence twice.
- **Rare-quantile calculation unstable:** the requested tail estimate has not met its numerical precision criterion. Increase the computational budget or review the posterior specification; the current displayed estimate is not a qualified result.
- **Prior configuration unresolved:** the imported legacy controls do not yet establish the effective prior or no-prior setting. Resolve them from a qualified configuration before fitting; the program cannot silently select a replacement prior.
- **Rating uncertainty omitted:** the reported band conditions on recorded discharges. It does not include possible shared errors from the extrapolated rating curve.

Attach method links and run-specific details to each warning. Do not suppress a failure merely because a plausible-looking plot can be drawn.

## Access and release requirements

Store readable Markdown method pages in the repository. Package a generated offline HTML manual and embed the relevant explanations and references in each analysis report. These are static documentation files, not an application interface. Online access must not be necessary to interpret an existing report. Source reference PDFs may remain externally linked; the report must still contain an original explanation, citation and section identifier.

The proposed CLI can expose commands such as `ffa explain lp3`, `ffa explain expected-parameter` and `ffa explain expected-probability`, while structured output files include `method_id`, `method_version`, `compatibility_profile`, `references` and `qualification_status`. Exact command names and method IDs are part of WP08 and must agree with the registry. The installed wheel must locate its packaged explanations and references outside the source checkout; see [repository setup](repository-setup.md).

A method cannot be marked qualified until its mathematics, user explanation, complete worked example and verification evidence refer to the same implementation. Review labels and figure captions against machine-readable results. Documentation must identify unsupported combinations and must not claim ARR endorsement, full Flike equivalence or future-climate suitability from a successful stationary fit.

Track method-documentation acceptance alongside implementation and numerical evidence in the umbrella issue and linked work-package issues. Completion of a planning page does not close a statistical qualification task.

Prepared 23 November 2026.
