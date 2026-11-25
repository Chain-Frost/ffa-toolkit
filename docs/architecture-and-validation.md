# Architecture and validation plan

Status: staged implementation, 24 November 2026. Mathematical components and internal integrations are implemented; the full fitting/sampling CLI is unfinished. [development requirements](https://github.com/Chain-Frost/ffa-toolkit/blob/main/docs/development.md) is accepted independently of [development requirements](https://github.com/Chain-Frost/ffa-toolkit/blob/main/docs/development.md). See [development stages](development-stages.md). These requirements do not claim native Flike equivalence or general ARR compliance.

## 1 Design objective

Build independently verified mathematical components and an explicit-configuration command-line workflow, then qualify the supported combinations against the existing TUFLOW/BMT Flike target when native evidence is available. The [development stages](development-stages.md) define initial acceptance; the [compatibility contract](compatibility-contract.md) retains the eventual five-family, Bayesian, L/LH, historical/censored, prior, zero-threshold, low-flow, rating-error and report requirements. Unknown native choices do not become defaults in the mathematical implementation.

The CLI is the supported user interface. Use internal Python modules and callable functions for composition, testing and reuse within the package; a separately supported public Python API is not a deliverable. CLI-generated figures, reports and static method documentation are output files. A GUI, desktop front end, QGIS interface and web application are outside scope at every phase.

ARR Book 3 Chapter 2 supplies the hydrological framework. Its importance-sampling procedure and the Kuczera formulation provide the scientific foundation; the pinned legacy program, documented controls and paired reference cases establish compatibility details. Unresolved priors, transformations and numerical controls remain evidence gaps. Similar results from differently specified models do not establish equivalence. WMAwater additions and other improved algorithms require a separate future scope and must not change the MVP's legacy calculation.

The [sample-data audit](sample-data-audit.md) currently identifies program version **5.0.306.0** and project format **3.10**, with LP3 report labels and output tables inspected. These distinct identifiers and file names do not establish input/output pairing. Retain that uncertainty until provenance, project settings and reported observation counts are reconciled.

Primary hydrological reference: [ARR 2019 Book 3, version 4.2](https://www.arr-software.org/pdfs/ARR_190514_Book3_V4.2.pdf), particularly sections 2.6.3.6 and 2.6.3.7, equations 3.2.52â€“3.2.56, printed pages 41â€“43.

## 2 Reuse numerical building blocks

| Requirement | Candidate | Project responsibility |
| --- | --- | --- |
| GEV, Pearson III, lognormal, Gumbel and generalized Pareto probabilities | SciPy distributions | Verify family, parameterization, support, transforms and conventions against each method specification. |
| Sample L-moments | SciPy lmoment | Supply reviewed family-specific parameter inversion and admissibility checks. |
| Optimization and numerical integration | SciPy optimize and integrate | Multistart strategies, bounds, support checks and meaningful failure reporting. |
| Exact and censored MLE checks | SciPy CensoredData and distribution fit | Independent likelihood checks and initial values where appropriate; not a substitute for a legacy estimator. |
| Random numbers | NumPy Generator and SeedSequence | Reproducible stream allocation, provenance and versioned replay. |
| Legacy Bayesian calculation | NumPy and SciPy importance-sampling implementation | Match verified posterior, source parameter coordinates, proposal, weighting, controls and output definitions. |
| Independent inference verification | Quadrature; emcee where useful | Check the same posterior and estimands. A verification tool is not an alternative MVP production backend. |
| Reporting and diagnostics | Matplotlib, HTML templates and suitable ArviZ components | Domain interpretation, uncertainty labels, citations and backend-appropriate diagnostics. |

SciPy documents the available [distributions](https://docs.scipy.org/doc/scipy/reference/stats.html), [censored observations](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.CensoredData.html) and [sample L-moments](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.lmoment.html). Sample L-moments alone do not constitute a family-fitting algorithm.

Two mappings require particular care. SciPy's [GEV shape convention](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.genextreme.html) has the opposite sign to conventions used in some other sources. Its [gennorm](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gennorm.html) is a symmetric exponential-power distribution, not the hydrological generalized-normal family. Never select a distribution by similar naming alone.

### Dependency and source admission

Before adding a dependency, record its purpose, exact version, upstream source, licence, transitive requirements, supported platforms and evidence of numerical suitability. Record the provenance and permitted redistribution of example data separately from software licensing.

The repository remains under **Sustainable Use License 1.0**. The [lmoments3 documentation](https://lmoments3.readthedocs.io/stable/readme.html) identifies GPL version 3 or later; under the [agent licence rules](../AGENTS.md#licence-and-third-party-material), treat it only as a separately installed external comparator. Do not copy or translate its implementation/tests or add it as a project dependency. Independently implement the referenced mathematical relationships. Any SciPy-based helper or alternative such as [Lmo](https://github.com/jorenham/Lmo), whose repository identifies BSD-3-Clause licensing, still requires version, suitability and dependency review. Preserve applicable third-party notices; optional packaging does not itself establish licence compatibility.

Keep plotting, document rendering and independent verification samplers outside the compulsory numerical dependency set unless an implemented command needs them. Pin releases for validation; upgrades require focused regression evidence. Apply the Ruff, strict Pyright, Python 3.14 and packaging requirements in [repository setup](repository-setup.md).

## 3 One scientific core

### Observation and analysis schema

Represent an analysis as a versioned specification containing site metadata, units, hydrological-year definition, analysis period, series type, observation records, historical periods, distribution, estimator, priors, zero threshold, low-flow decisions, rating-error controls, requested output quantities and reporting settings. Record both legacy source coordinates/settings and their explicitly named internal mappings.

Input adapters read the legacy project and report formats established by the supplied corpus, alongside column-configurable text/CSV input. Preserve raw files, source hashes, original labels, unrecognised fields and parsing decisions. Do not infer pairing from a shared file stem, rewrite source evidence or silently discard unknown fields that may affect a calculation. A parser may inventory an unqualified project while the fitting command rejects its unresolved statistical settings. Read-only import and comparison support do not imply a byte-identical legacy writer.

Each observation needs a stable identifier, year or date range, source and interpretation. Supported semantics should distinguish exact observations, interval bounds, left censoring, right censoring, known zero flow and missing information. Preserve original values and exclusion reasons. Invalid intervals, duplicate years and overlapping evidence must produce actionable errors.

Historical evidence needs its observation period and applicable perception threshold. A year known not to exceed a threshold contributes information. An unobserved year does not become a non-exceedance. Historical events must not also be counted as additional gauged events for the same annual maximum. Validate coverage before constructing the likelihood.

Low flows, true zero flows, values below a selected zero threshold and missing years are different cases. Do not replace zeros with an arbitrary positive constant. Reproduce the verified legacy LN/LP3 zero-threshold rules; do not invent a new literal-zero mixture or prior. The inspected fourth LP3 coordinate, `logit(P(flow <= zero threshold))`, is evidence of a threshold-probability parameter, not a complete definition of its likelihood or uncertainty treatment. Preserve a PILF decision and its rationale separately from the original observation. Distribution fitting and low-flow selection must retain the target's record-length/count semantics.

Rating-error settings need an explicit component representing the target's error model, grouping and dependence. An uncertain rating segment may affect several floods jointly; independent random perturbations are not an equivalent substitute. Qualify this existing legacy capability within the MVP even if additional reference cases must be acquired.

### Distribution contract

Each of the five distribution adapters should provide support, log density, log CDF, log survival probability, inverse CDF or inverse survival probability, parameter validation and documented transformations. Serialized and reported parameters must have unambiguous names, units, log bases and ordering. Internal optimizer coordinates may differ but need reversible mappings. This is an internal module contract, not a separate public API promise.

For LP3, implement the declared log transformation around Pearson III, including both skew signs and the near-zero-skew limit. [SciPy pearson3](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearson3.html) supplies the standardized distribution. Include transformation Jacobians where required for likelihoods on discharge and priors under reparameterization. A constant irrelevant to one optimizer can still matter when comparing probability models expressed on different measurement scales.

Preserve the inspected LP3 source coordinates: mean of natural-log flow, natural log of the standard deviation of natural-log flow, skew of natural-log flow, and the threshold-probability logit where active. The standard deviation passed to the Pearson III adapter is the exponential of the second coordinate. Preserve prior densities and reported posterior means in the source coordinates before applying this nonlinear map.

Reject invalid parameters and assign zero likelihood to impossible exact observations. A valid prediction point outside a component's support still has a well-defined CDF/SF of zero or one; posterior probability averaging must not reject it. Do not silently truncate a distribution at zero: truncation changes the model. Report and review physically problematic probability mass or parameter support.

### Likelihood contract

One observation-model layer should construct the mathematical log-likelihood used by the legacy importance sampler and independent verification calculations. For continuous models its basic contributions are log density for exact values, log CDF for upper bounds, log survival probability for lower bounds and log interval probability for intervals. The [SciPy fit documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.fit.html) supplies a useful independent statement of these contributions.

The legacy zero-threshold calculation needs a separately specified likelihood component. Establish how threshold probability, retained low-flow counts, transformed observations and prior settings interact before implementing it. If the verified model includes a discrete mass, use probability for an exact atom and define interval endpoint inclusion explicitly. General mixture formulas are mathematical safeguards, not evidence that the legacy program uses a particular newly designed model.

Evaluate interval differences stably in the appropriate tail, using log-domain arithmetic. Do not subtract two rounded CDF values near one. Verify extreme and narrow intervals against independent high-precision calculations.

Historical record selection, rating error, zero-threshold treatment and censoring rules belong in explicit observation-model components. Numerical verification may not introduce a different interpretation of the same data. Preserve any unexplained difference between documented theory and measured target behaviour as a tracked compatibility/scientific discrepancy.

### Result contract

A result should contain input and method identifiers; compatibility-profile version; source and internal parameter coordinates; fitted parameters and supported posterior summaries; uncertainty representation; requested outputs; diagnostics; warnings; exclusions; references; and provenance manifest. Store full numerical precision separately from target report formatting. Preserve target columns and their meanings before adding explanatory output.

Report separately:

- Quantiles evaluated at a declared parameter point estimate.
- Credible or confidence intervals for each flood quantile.
- Expected probabilities at supplied flows, explicitly identified as non-exceedance or exceedance probabilities, and any separately supported predictive quantiles.
- Numerical uncertainty in the estimated outputs.

ARR section 2.6.3.7 discusses weighted posterior parameter summaries and their use. Preserve the source-defined posterior expected parameter vector and the resulting curve, q_p(E[theta | data]), for the legacy calculation. It is generally different from E[q_p(theta) | data] and from a posterior predictive quantile. The averaging parameterization must be fixed in the method specification; averages do not commute with nonlinear transformations. The inspected sample reports include importance-sampling parameter moments, expected-parameter curves with 90% limits, moments of log10(q), and expected-probability-at-flow output. The printed probability column is numerically consistent with non-exceedance F and a return period of 1/(1-F); its convention needs confirmation before conversion to AEP. These are evidence targets, not yet qualified numerical fixtures. See the [sample-data audit](sample-data-audit.md).

Do not relabel a non-exceedance column as AEP or replace a reported probability-at-flow column with an inverted predictive curve under the same name. Quantile inversion must use the generalized inverse if a verified model contains atoms; a continuous survival-equality root need not exist at a probability jump. The target's defaults and supported options determine the production outputs.

A distribution's [interval method](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.interval.html) returns a probability range under fixed parameters. It does not provide sampling uncertainty for an estimated design quantile.

## 4 Inference implementation sequence

### Legacy L/LH-moment methods and deterministic checks

Implement and independently verify H=0 L-moment fitting for the supported five-family configurations, fixed GEV H=1â€“4, and a sufficiently specified published optimized-H procedure. ARR sections 2.6.4 and 2.8.10 supply the foundation and worked examples. Missing primary mathematical definitions remain active source/specification work; missing vendor defaults and exact selection/bootstrap agreement belong to #39. Bootstrap verification must repeat the declared fitting/selection procedure and report failed replications. A raw fourth-moment residual is not a standardized Wang test or an optimized-H rule.

Exact-observation and censored MLE checks can establish likelihood conventions and useful initial values. They are independent verification tools unless established as a target option. They do not substitute for Bayesian treatment of historical information and priors.

Generic SciPy fitting can return a local optimum and uses finite penalties outside support. The project must validate feasibility, compare multiple starts and retain optimizer diagnostics. Do not infer a globally reliable fit from a success flag.

### Importance sampling as the MVP production method

Specify and review the published ARR procedure and every explicitly supplied mathematical control before implementation. Build parameter estimation/initialisation, proposal construction, proposal sampling, log posterior evaluation, log importance ratios, normalized weights, precision controls and weighted summaries as identifiable stages. Supplied-particle summaries alone do not implement sampling. Cite the inspected equation or unresolved mathematical question for each stage. Unknown native proposal/RNG/stopping or prior defaults are separately tracked in #39 and cannot be silently selected. The independent execution contract and its tests can proceed without native runs.

For a proposal g(theta), target proportional to likelihood times prior, and draws theta_i, retain log weights proportional to log likelihood plus log prior minus log g. Use stable normalization. Verify that the proposal covers the admissible posterior support. A narrow local Gaussian approximation may fail on skewed, bounded or multimodal targets. Record such limitations and independently assess their effect; do not silently switch the legacy profile to a new proposal or weighting scheme. Improved production algorithms belong to a future, separately qualified profile.

Store raw proposal draws and weights where practicable. Report concentration of weights, effective sample size, repeat-run stability and output-specific precision. An ESS computed from weights alone is useful but is not a sufficient guarantee for a rare quantile.

The research on [Pareto smoothed importance sampling](https://www.jmlr.org/papers/v25/19-556.html) provides foundations for diagnosing problematic weight tails and Monte Carlo uncertainty. A PSIS diagnostic or modified weighting scheme would be a separately documented enhancement. Do not imply that smoothing reproduces FLIKE, or silently replace the reference estimator.

Unknown default priors or proprietary numerical choices remain explicit compatibility gaps. Numerical agreement cannot validate an unidentified prior. Independently implement sufficiently specified supplied-prior models; an unknown legacy no-prior/default mode remains unsupported until evidenced. Do not choose a proper prior and label it the legacy no-prior model merely to simplify validation.

### Independent inference verification

After establishing the likelihood and source conventions, use independent quadrature or a sampler on a fixed verification case set when it addresses a concrete risk. [emcee](https://emcee.readthedocs.io/en/stable/) accepts a numerical log-posterior and is one candidate. Compare the same priors, transformations and output estimands, with each calculation's numerical uncertainty. A verification dependency does not become a supported CLI fitting method.

The [emcee convergence documentation](https://emcee.readthedocs.io/en/stable/tutorials/autocorr/) warns that walkers in one ensemble are dependent. Do not use them as independent chains for R-hat. Use appropriate autocorrelation analysis, independent ensemble runs and validated diagnostics for reported outputs.

PyMC may become useful for separately approved future hierarchical or richer models. Its [black-box likelihood guidance](https://www.pymc.io/projects/examples/en/latest/howto/blackbox_external_likelihood_numpy.html) explains that gradient-based NUTS requires a differentiable likelihood. A SciPy call does not automatically supply a usable PyTensor gradient. This is background for future work, not an MVP production-backend selection exercise.

## 5 Meaningful acceptance gates

These are project gates, not ARR-prescribed software tolerances. Track **legacy compatibility** and **independent scientific verification** separately. Matching a flawed or unidentified calculation is not scientific validation; a scientifically reasonable alternative is not proof of replication. If they disagree, record the evidence and block an unqualified claim rather than silently changing the model. Optional improved algorithms have a separate future acceptance path.

The [development stages](development-stages.md) separate independent mathematical/software acceptance from the [compatibility contract](compatibility-contract.md). In the following matrix the compatibility column is the later native claim; independent verification and software integration are active initial work. Gates Aâ€“C, admitted independent comparisons in D1 and software checks in E support initial acceptance. Gate D2 and the native-release decision remain deferred.

| Area | Compatibility evidence | Independent verification |
| --- | --- | --- |
| Input and configuration | Paired raw project/report files, pinned build and format, exact option and observation mappings. | Hand-worked observation interpretation, coverage and rejection cases. |
| Five probability families | Source coordinates, parameter conversions and supported estimator combinations. | Support, CDF/SF, inverse probabilities, transforms and limiting cases. |
| L/LH fitting | H=0 for all five; fixed and optimized GEV H; matching selection and uncertainty outputs. | Published moment relationships, independently calculated fits and bootstrap checks. |
| Bayesian calculation and priors | Legacy importance sampling, no-prior/supplied-prior behaviour, controls and report moments. | Likelihood/prior Jacobians, tractable integration or independent posterior checks. |
| Historical, low-flow and zero-threshold options | Matching thresholds, counts, MGB decisions and retained-record semantics. | Disjoint exposure, exact/censored contributions, threshold-probability interpretation. |
| Rating error | Enabled/disabled cases with matched controls, groups and reported uncertainty. | Specified dependence, synthetic recovery and integration checks. |
| Results and figures | Target quantities, levels, plotting positions, scale controls and displayed precision. | Estimand distinctions, weighted summaries, interval construction and tail inversion. |
| CLI and package | Installed commands, documented options, explicit failures and reproducible manifests. | Ruff, strict Pyright, meaningful tests and clean installed-wheel execution. |

An unpaired report, successful parser or visually similar plot does not satisfy a native-conformance row. Existing-file parsing and report-internal identities remain useful independent checks. Missing native coverage stays open in #39/#12; it does not keep a completed mathematical/software issue open. Missing algorithms and internal integration remain active in #40.

### Gate A Mathematical correctness

Build independently derived fixtures for CDFs, quantiles, transformations and observation likelihoods. Test central probabilities and the declared rare-event domain, including bounded tails. Include GEV approaching Gumbel, LP3 approaching lognormal at zero skew, negative LP3 skew, unit conversions, invalid support, tied observations and degenerate records.

Verify the distinction between missing and known non-exceedance years with small hand-worked histories. Check equivalent expanded and aggregated historical representations with the appropriate counting constants: labelled annual indicators and an unlabelled count have different combinatorial factors, even where their parameter posteriors agree. Reject overlaps. A test that calculates both expected and actual values with the same adapter is not independent evidence.

Set numerical tolerances from conditioning and reference precision. Document separate tolerances where extreme-tail inversion is intrinsically harder.

### Gate B Estimation and uncertainty

Use a planned simulation matrix spanning distribution parameters, sample sizes, censoring, historical exposure and prior strength. Evaluate bias, convergence failures and repeated-sample coverage. Account for binomial uncertainty when assessing an observed coverage rate. Do not demand that every simulated interval contains the truth.

For Bayesian implementations, apply [simulation-based calibration](https://arxiv.org/abs/1804.06788) to the complete generative and observation model. Include parameter and flood-quantile summaries. Prior-predictive calibration complements, rather than replaces, investigations at selected difficult parameter values.

Simulation-based calibration requires a proper generative prior. Use declared calibration profiles for that purpose and separately examine posterior propriety and difficult cases in the target's no-prior mode. Do not modify the legacy prior to make a calibration experiment convenient. A bounded prior is a scientific choice, not an invisible numerical fix.

### Gate C Computational adequacy

Agree output-specific precision criteria before release. For example, choose a maximum Monte Carlo standard error relative to the posterior uncertainty or intended reporting precision of each requested flood quantile. Determine the actual threshold from representative engineering use and record the decision.

For production importance sampling, inspect weight tails and repeated independent runs, not only total sample count. For an independent MCMC verification calculation, inspect bulk and tail effective sample sizes, appropriate convergence diagnostics and MCSE for transformed quantiles. The [rank-normalized R-hat research](https://arxiv.org/abs/1903.08008) supplies foundations; any chosen numerical gate must identify its applicability.

A failed diagnostic must prevent an unqualified accepted result. Exhausting a computation budget is not convergence. A fixed percentage comparison borrowed from runoff-routing validation, such as 0.3%, is not a valid universal tolerance for Bayesian flood estimates.

### Gate D1 Independent external mathematical evidence â€” initial development

Use source-pinned, appropriately licensed public examples and independent mathematical implementations where the model, coordinates and asserted quantity match. ARR examples and [USGS Bulletin 17C](https://pubs.usgs.gov/tm/04/b05/tm4b5.pdf), Appendix 10, provide useful evidence. [PeakFQ](https://www.usgs.gov/tools/peakfq) implements Bulletin 17C; its EMA estimates are not Bayesian Flike expected outputs. Preserve source precision, settings, derivations and limitations. Shared numerical backends do not provide independent verification of one another.

Read-only retained-file and report-internal checks also belong in initial development when their inputs are available. Neither these checks nor mathematical comparison vectors require a native executable or promote a candidate into a qualified Flike run.

### Gate D2 Native Flike comparisons â€” deferred to issue #39

Pin input data, executable version, file-format version, distribution, source parameter coordinates, priors, zero/censoring/rating rules, optimizer/sampler controls, output definition and available numerical precision for every Flike comparison. Establish actual input/output pairing before admitting a fixture. Classify differences before judging them: input interpretation, method definition, numerical error, stochastic error or unidentified target behaviour.

Determine deterministic tolerances from conditioning and reference precision, and stochastic tolerances from repeated target runs plus the precision of both calculations. Exact random draws or bitwise agreement are not the default requirement. Do not loosen tolerances after observing an unexplained discrepancy. The current sample audit provides starting evidence, not a passed comparison suite.

Run this gate only when the controlled executable, source/settings correspondence and deliberate native-run authorisation are available. Keep it separately invoked from ordinary mathematical/software CI. A correctly blocked native ledger or deferred comparison is not a failure of initial acceptance and is never a native pass. Synthetic tests of the comparison/admission machinery remain ordinary software tests.

### Gate E Platform and release evidence

Target Python 3.14 using current annotations: built-in collections, union syntax and modern type aliases where useful. Do not use legacy typing.List-style aliases. Require Ruff lint/format checks and strict Pyright for first-party source, tests and maintained scripts, following the [repository setup](repository-setup.md) baseline. Confirm that every selected dependency installs and works on the declared platform matrix before claiming support. Independent verification tools need recorded compatible environments.

Keep CI focused on scientific contracts, essential CLI integration paths and packaging. Verify the built wheel outside the checkout, including installed help/version commands and packaged method/reference resources. Independent review should examine formulas, observation semantics and uncertainty calculations, not only coding style. Release claims must identify which distributions and observation combinations actually passed. A partial working slice may be merged as implementation progress without being labelled the complete replication MVP.

## 6 Reproducibility and performance

Persist source-data hashes, normalized-input hashes, configuration, code revision, method-profile version, dependency versions, RNG family and seed lineage. Save optimizer/sampler controls, diagnostics and relevant weighted draws or checkpoints. Sensitive user data remain separate from distributable public fixtures.

Allocate random streams deterministically by run and task identity so scheduling does not reassign streams. NumPy documents [parallel streams](https://numpy.org/doc/stable/reference/random/parallel.html) and its [compatibility limits](https://numpy.org/doc/stable/reference/random/compatibility.html). A seed alone does not guarantee identical results across builds, machines or versions. Separate reproducible replay within a pinned environment from statistical equivalence across supported platforms.

Measure runtime and memory against an agreed workload: representative record sizes, censoring patterns, distributions, uncertainty outputs and precision requirements. Choose that workload before optimization. Report failure rates and achieved precision alongside timing. There is no justified arbitrary speed target at this stage.

## 7 User documentation is part of acceptance

Maintain a method registry containing purpose, assumptions, applicability, equations, parameter conventions, references, implementation status and validation evidence. Generate help text and report citations from that registry so interface descriptions cannot drift from computation.

Every released method needs a concise explanation, a worked example and clear interpretation of its uncertainty. Reports should show the actual method and prior, observation decisions, alternatives examined, limitations and relevant references. A warning must explain its effect on the result and the next useful action.

WP01 is merged, and the initial reader/kernel foundations are implemented. Continue integration against those shared contracts rather than recreate the scaffold. Agree schema, parameter, error and reference identifiers before integrating their modules; each work package owns separate files. Detailed owners and dependencies belong in the [implementation plan](implementation-plan.md). The umbrella GitHub issue tracks delivery and linked work-package acceptance; these documents retain the enduring specification and evidence rules.
