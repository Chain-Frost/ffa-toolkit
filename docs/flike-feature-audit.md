# Flike feature and method audit

Research snapshot: 23 November 2026. This document distinguishes documented legacy TUFLOW Flike functionality, WMAwater's announced development, and capabilities that remain unverified. It defines evidence for the proposed Python implementation; it does not claim execution or numerical validation of Flike.

The binding MVP is replication of the **existing TUFLOW/BMT-supplied Flike**, delivered through a **CLI only**. The initial compatibility profile is **program 5.0.306.0 / file format 3.10**, established from the supplied artifacts. The [compatibility contract](compatibility-contract.md) defines its required scope. WMAwater additions are future work; GUI delivery and a separately supported public Python API are excluded at every phase. Internal functions support CLI composition and tests, and static reports and figures remain valid command outputs.

## Product baseline and evidence boundary

TUFLOW's current [downloads page](https://www.tuflow.com/downloads) directs FLIKE distribution to WMAwater. The [WMAwater 2026 roadmap](https://flike.wmawater.com.au/) describes a completed core engine with interface work and final testing continuing. A current publicly released build number was not established. The version 5.0.220.0 screenshots in the ARR worked examples identify the illustrated legacy build, not the latest release.

The strongest accessible legacy workflow reference found is [Australian Rainfall and Runoff Book 3, version 4.2](https://www.arr-software.org/pdfs/ARR_190514_Book3_V4.2.pdf), particularly Chapter 2. A standalone current Flike manual and a specification for the new engine were not located. Consequently, each benchmark must retain its actual executable version and configuration. Public guidance from a different illustrated build supplies background; it does not prove every control or numerical default of the chosen target.

The [sample-data audit](sample-data-audit.md) records 16 numerical reports identifying program **5.0.306.0** and file version **3.10**, and 12 projects beginning with `FLIKE data file  310`. This establishes an evidence baseline, not custody of the executable or complete project/report correspondence. Candidate data relationships and conflicting censoring configurations are recorded separately. The samples do not identify the latest available product release.

## Documented legacy features

| Area | Publicly supported capability | Evidence |
| --- | --- | --- |
| Probability families | Lognormal, log-Pearson type III, Gumbel, generalized extreme value, generalized Pareto | [Historical and palaeoflood study using Flike](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2016WR019959), §3.6 |
| Bayesian inference | Importance sampling; multivariate Normal parameter priors | [USACE verification report](https://www.researchgate.net/publication/352414326_Verification_of_the_Bayesian_Estimation_and_Fitting_Software_RMC-BestFit), pp.66–69; supplied posterior-summary reports in the [sample audit](sample-data-audit.md) |
| L-moments | H=0 available for all five listed families | [ARR Book 3](https://www.arr-software.org/pdfs/ARR_190514_Book3_V4.2.pdf), §2.8.10.3 |
| LH-moments | H≥1 restricted to GEV; fixed H=1–4 and optimized H | ARR §2.8.10.3, Figure 3.2.50, §2.8.10.4 |
| Historical information | Thresholds and counts over specified periods; information outside the gauged record | [Flike application study](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2016WR019959), §3.6; USACE report pp.66–67 |
| Low flows | Multiple Grubbs–Beck identification followed by censoring | USACE report pp.67–69 |
| Zero flows | Zero-threshold control labelled for LN and LP3 | ARR Figure 3.2.50 |
| Regional information | Gaussian parameter priors; LP3 regional-information import | ARR §2.8.5 |
| Rating uncertainty | Inbuilt rating-error facility documented in applied regional research | [Tasmanian regional study](https://www.tandfonline.com/doi/full/10.1080/02626667.2014.950583), §3.1 |
| Input and projects | Column-configurable text/CSV import; optional years; append/replace; `.fld` project files | ARR §2.8.3 |
| Results | Probability plots, text reports, parameter summaries, quantiles and uncertainty limits; scale controls | ARR §§2.8.3–2.8.6 |

All listed legacy capabilities belong to the replication MVP, including generalised Pareto, L/LH-moments and rating error. The supplied-prior and no-prior modes must retain the target's verified behaviour. This inventory does not establish that every combination of distribution, estimator, observation type and prior is supported or statistically appropriate. Missing qualification evidence is a tracked gap, not an automatic deferral of a legacy feature. Generalised Pareto support does not by itself establish a complete peaks-over-threshold extraction/exposure workflow.

## Direct evidence from the supplied reports

The [sample-data audit](sample-data-audit.md) is the detailed provenance record. The following observations constrain implementation without establishing a complete numerical oracle:

| Observed convention | Implementation requirement |
| --- | --- |
| LP3 coordinates: mean of natural-log flow, natural log of the standard deviation of natural-log flow, and skew of natural-log flow | Retain the log-standard-deviation coordinate for legacy priors and posterior summaries. Convert to standard deviation explicitly when calling a probability kernel. |
| Additional LP3 coordinate: logit of probability that flow is at or below the zero threshold | Preserve this coordinate and resolve the likelihood, threshold rules and prior. Do not assume a newly designed literal-zero mixture. |
| GEV coordinates labelled location `u`, natural log of scale `a`, and shape `k` | Verify transformations and shape-sign conventions against the target before mapping to a numerical library. |
| Posterior moments explicitly described as coming from importance sampling | Reproduce the legacy importance-sampling production calculation and its supported controls. |
| Quantiles at posterior expected parameters, 90% probability limits and moments of `log10(q)` | Keep natural-log parameter coordinates distinct from base-10 output moments. Qualify the interval construction and expected-parameter convention. |
| Expected probabilities evaluated at specified flow magnitudes | Preserve this output separately from quantiles at specified probabilities. The printed probability is numerically consistent with non-exceedance `F` and return period `1/(1-F)`; confirm before deriving or labelling an AEP. |
| Raw plotting column labelled `AEP plot position` | Sample values are also numerically consistent with non-exceedance probability. Preserve the raw label separately from the interpreted probability kind; do not assign exceedance semantics from the label alone. See the [compatibility contract](compatibility-contract.md). |

Fifteen reports identify LP3 and one identifies GEV. There is no supplied numerical L-moment report, and all candidate pairs still require run-control verification. The available evidence therefore supports parser and specification work immediately while leaving much of the legacy comparison matrix open.

## What the statistical methods require

### Bayesian inference is more than maximum likelihood

[Kuczera (1999)](https://doi.org/10.1029/1999WR900012) develops a Monte Carlo Bayesian framework for gauged and censored flood information, including rating uncertainty, expected probabilities and quantile limits. Its computational approach uses importance sampling. A maximum-likelihood fit provides a parameter estimate; it does not by itself reproduce that posterior calculation.

The Python specification must therefore define the full likelihood and prior, the parameter coordinates in which that prior is specified, and the target quantities reported. Production importance sampling is part of the legacy replication contract. An independent MCMC or quadrature calculation can check the same posterior; it does not become an alternative MVP production method. Changing parameter bounds, ignoring a transformation Jacobian, or silently replacing a prior changes the model even when the resulting curve appears similar.

Reproduce both supplied-prior and no-prior behaviour, including actual numerical bounds/defaults once verified. Do not replace no-prior mode with a newly selected proper prior. A joint Gaussian prior in mean/log-standard-deviation/skew coordinates is not Gaussian in mean/standard-deviation/skew coordinates after exponentiation; preserve the full transformed density. Unknown defaults remain explicit qualification gaps.

For engineering output, distinguish these quantities explicitly:

1. A quantile evaluated at a chosen parameter vector, including the expected parameter vector.
2. The posterior distribution of the population quantile, with its median, mean where finite, and credible limits.
3. The quantile of the posterior predictive distribution, obtained by averaging probabilities over parameter uncertainty and then inverting.
4. The expected non-exceedance or exceedance probability at a supplied discharge, with its convention explicitly identified.

In notation, `q_p(E[theta | D])` is generally different from `E[q_p(theta) | D]`; neither should be used as an undocumented synonym for a predictive quantile. ARR §2.5.1 calls the first the expected-parameter quantile. Average in the source parameter coordinates before transforming: averaging standard deviations instead of their logarithms changes this curve. The result schema must store the estimand and convention alongside every number. These distinctions explain the mathematics; only established legacy outputs become production requirements, and a reported probability-at-flow column must not be replaced by an inverted curve under the same name.

### L-moments and LH-moments serve different purposes

[Wang (1997)](https://doi.org/10.1029/97WR02134) introduces LH-moments to emphasize higher order statistics and reduce the influence of smaller events on rare-flood estimates. The published formulation includes GEV examples and simulation assessment. [Wang (1998)](https://doi.org/10.1029/98WR02364) provides approximate goodness-of-fit tests using sample and theoretical LH-kurtosis.

For this project, H=0 L-moment fitting for all five families and GEV H=1–4 plus optimized H are required legacy functionality. Optimized H must reproduce the legacy selection rule with referenced diagnostic output. It must not mean choosing whichever curve produces the preferred design discharge. Ordinary L-moment fitting, higher-order fitting, and their uncertainty procedures need separate capability flags and verification cases. Match the target's uncertainty procedure and whether selection is repeated during bootstrap fitting.

### Censoring must preserve information

The [Cohn et al. (2013) MGBT paper](https://pubs.usgs.gov/publication/70095255) describes identifying potentially influential low flows and representing them as less-than observations for censored-data inference. The [USGS reference implementation](https://code.usgs.gov/water/stats/MGBT/-/tree/v1.0.3) provides an additional algorithm comparison route.

The planned data model should preserve the original observations, the proposed censoring decision, its reason, the threshold, and the retained exposure/count information. Low-flow censoring must not become irreversible deletion. Historical evidence likewise requires explicit observation periods and perception thresholds; a list of exceptional floods without its observation process is insufficient to define a historical likelihood.

The legacy LN/LP3 zero-threshold option is also required. A true no-flow year, a measurement below detection, an observation below a user threshold and a missing year are different facts. The observed logit coordinate does not establish how each enters the likelihood. Resolve retained counts, activation rules, fitted/fixed parameters and uncertainty from authoritative or controlled reference evidence; do not substitute an arbitrary beta prior, empirical zero probability or positive offset.

### Rating-curve errors can be correlated

[Kuczera (1996)](https://doi.org/10.1029/96WR00804) models systematic errors associated with rating-curve extension and develops a likelihood accepting gauged and binomial-censored data. It also identifies multiple optima and limitations of local approximations in difficult cases.

A common uncertain rating curve can affect several peaks together. Independently perturbing each peak is a different model. The legacy rating-error facility therefore needs a dedicated mathematical specification, matched reference cases and independent verification **within the MVP**. Establish actual grouping, covariance, thresholds and defaults before qualification. An interface tab labelled “errors” is not enough to establish the numerical treatment used by a particular executable.

## WMAwater roadmap versus released capability

The following summarizes the status labels on [WMAwater's roadmap](https://flike.wmawater.com.au/) at the research date. These are vendor development statements, not tests performed by this project or additions to the legacy MVP.

| Announced addition | Roadmap presentation | Treatment in this project |
| --- | --- | --- |
| ARR mode and advanced mode | Planned new mode | Future method scope; await a released specification before any compatibility claim |
| Time or temperature covariates | Engine item marked completed | Separate future nonstationary research and implementation scope |
| TCEV and expanded distributions | Engine item marked completed | Future additions require equations, constraints and validation; the legacy five families are already MVP scope |
| ARR-consistent priors | Engine item marked completed | Future profile; must not replace legacy defaults or supplied-prior behaviour |
| Additional uncertainty methods | Engine item marked completed | Future scope; algorithms not established by the roadmap |
| Full MCMC | Engine item marked completed | Future production method; independent validation MCMC does not change the MVP sampler |
| Updated GUI | Coming soon | Vendor history only; GUI excluded from this project at every phase |
| Python and R scripting | Coming soon | Vendor history only; no separately supported public API deliverable for this project |
| Browser access | Coming soon | Vendor history only; browser applications excluded at every phase |

No date-specific promise of availability should be copied into the implementation schedule.

## Published mathematics and implementation ownership

The foundational papers above disclose methods that can be specified and implemented independently. Public explanation of a method does not supply the commercial program's source, numerical defaults, project parser, documentation assets or branding. This project should use original code and documentation, retain bibliographic attribution, and record the licences of reused software and datasets.

ARR itself identifies a Creative Commons Attribution licence with exceptions for marked third-party material. That is a reason to retain provenance, not to assume every embedded figure or accompanying software asset has identical permissions. A scientific reference register and a dependency/data licence register should remain separate.

The goal is a transparent CLI replication whose methods can be reviewed without access to Flike. Legacy compatibility and independent scientific verification are separate acceptance dimensions. Neither replaces the other. If reference behaviour conflicts with the mathematics, retain and explain the discrepancy rather than silently changing the model or presenting a reproduced defect as scientifically endorsed. Accessible CLI explanations and method pages are required by the [user documentation contract](user-documentation-contract.md).

## Independent benchmark route

The [USACE RMC-TR-2020-02 verification report](https://www.researchgate.net/publication/352414326_Verification_of_the_Bayesian_Estimation_and_Fitting_Software_RMC-BestFit), pp.66–69, compares RMC-BestFit with Flike for Singleton LP3, historical binomial censoring, a regional skew prior, and Wimmera GEV before and after MGBT censoring. These comparisons are valuable because different inference engines are used. The report's curve terminology must still be reconciled to the exact estimand before importing expected values.

The official [RMC-BestFit repository](https://github.com/USACE-RMC/RMC-BestFit) provides a further independent implementation and public technical documentation. Its [LP3 reference](https://github.com/USACE-RMC/RMC-BestFit/blob/main/docs/technical-reference/distributions/log-pearson-type-iii.md) identifies tests for skew signs, the zero-skew lognormal limit, Jacobians, support endpoints and CDF/quantile inversion. Selecting LP3 alone does not implement EMA, regional skew or MGBT.

Proposed benchmark records should include source licence, original data hash, executable/library version, parameterization, log base, units, priors, censoring periods, sampler configuration, estimand, interval definition, and numerical tolerance. Rounded tables and plots should have appropriately limited acceptance precision. Stochastic comparisons should incorporate Monte Carlo error rather than require identical random draws.

## Open questions before compatibility claims

- Can the program 5.0.306.0 executable and its documentation be obtained for controlled reruns of file-format-3.10 projects?
- Which saved projects and reports have verified complete run correspondence, including excluded rows and grouped censoring records?
- What numerical bounds and priors does it use in each parameterization?
- What is the precise probability treatment of the LN/LP3 zero threshold?
- Which observation errors are independent and which share a rating-curve component?
- Is there a documented legacy command-line or batch interface?
- Is a complete, licensed `.fld` format specification available?
- Is there a released end-to-end peaks-over-threshold workflow, including event rate and exposure? Generalized Pareto support alone does not establish this.
- What is the exact convention and numerical construction of each expected-probability, quantile-moment and interval output?
- For separately approved future work, which WMAwater additions have actually shipped with specifications and reference cases?

These gaps do not prevent repository setup, observation adapters, distribution kernels or independently referenced implementation work from commencing. They prevent unsupported claims of complete Flike reproduction. Track remaining legacy evidence and qualification in the [compatibility contract](compatibility-contract.md) and its linked delivery issue; do not move an unresolved required method into future scope.
