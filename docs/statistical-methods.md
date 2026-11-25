# Statistical methods and engineering interpretation

**Status:** mathematical specification and interpretation, updated 24 November 2026. Individual components have implementation and independent-verification evidence recorded in their method pages/register; this document alone does not establish implementation or native Flike equivalence.

The [development stages](development-stages.md) separate initial mathematical/software delivery #40 from deferred native evidence/comparison #39. Implement sufficiently specified published mathematics with explicit settings and independent verification now; unknown vendor defaults are not initial acceptance prerequisites. The [compatibility contract](compatibility-contract.md) retains the later target behaviour and native claim. Do not invent an unidentified law or describe an independent mathematical choice as a Flike default. WMAwater additions remain future scope and GUI development is excluded.

Each implemented calculation must connect its equations, assumptions, numerical algorithm and independent verification evidence to an accessible explanation, with native qualification shown separately. Internal callable functions support CLI composition and tests; a separately supported public Python API is not a deliverable. Generated figures, reports and static documentation are CLI output files.

## 1 Guidance baseline and scope

The initial guidance baseline is [Australian Rainfall and Runoff, Version 4.2, Book 3](https://www.arr-software.org/pdfs/ARR_190514_Book3_V4.2.pdf), Chapter 2, *At-Site Flood Frequency Analysis*, by Kuczera and Franks. Its chapter date is 14 May 2019. Current climate context comes from [Book 1, Chapter 6](https://www.arr-software.org/pdfs/ARR_190514_Book1_V4.2.pdf), updated in 2024. Pin both the guidance version and individual method version in every result.

ARR Chapter 2 §2.1 explicitly anticipates independent software development. Nevertheless, implementing published equations does not establish fitness for engineering use. This specification therefore separates a method's scientific basis from its implementation and qualification status.

**Capability target:** five-family calculations, Bayesian importance sampling, supported L-moment and GEV LH-moment fitting, historical/censored data, supplied and no-prior modes, low-flow/zero-threshold treatment and rating-error options. Initial implementation covers explicitly defined mathematical configurations; unknown native semantics remain unsupported and separately tracked. The later native gate qualifies the target-supported combinations rather than assuming every possible family/estimator/data combination. Complete POT extraction and new nonstationary methods remain future work unless established as part of the legacy target.

The uploaded sample evidence currently identifies reports labelled program version **5.0.306.0** and project format **3.10**. These are distinct identifiers. They do not by themselves establish every project's input/output pairing or all legacy settings. See the [sample-data audit](sample-data-audit.md) for provenance, inspected fields and qualification gaps. Unknown behaviour must be investigated and recorded; it must not be replaced by an independently invented default.

## 2 What is being estimated

Let $Q$ be the maximum instantaneous discharge in an explicitly defined year. For a stationary annual-maximum model with parameters $\theta$, define

$$F(q\mid\theta)=\Pr(Q\leq q\mid\theta),\qquad S(q\mid\theta)=1-F(q\mid\theta).$$

The discharge with annual exceedance probability $p$ is $q_p(\theta)=F^{-1}(1-p\mid\theta)$. Store probabilities as fractions; format percentages only for presentation. A value of `0.01` means 1% AEP.

An annual maximum series has one observation per assessed year. An event series may contain several independent floods per year. Under a Poisson event-arrival model, an exceedance rate $\lambda_q$ gives $\mathrm{AEP}(q)=1-\exp(-\lambda_q)$. Consequently, one exceedance per year corresponds to approximately 63.2% AEP, not 100% AEP. This distinction becomes material for frequent events. See ARR Book 3 §§2.2.2 and 2.7.

**Project requirements:** retain the year definition, time zone, instantaneous versus aggregated measurement, exposure period and event-extraction rules. A daily maximum derived from daily means must not silently become an instantaneous annual peak. Missing years are not zero-flow years.

## 3 Legacy probability families

ARR §2.4.2 identifies LP3 and GEV as reasonable initial families for engineering assessment. The replication MVP must nevertheless implement all five documented legacy families and their supported fitting combinations; it does not reduce product coverage to those two recommendations.

| Family | Intended use and interpretation | Required qualification |
| --- | --- | --- |
| Log Pearson type III, LP3 | Fit a Pearson III distribution to logarithmic positive discharges; allows either sign of log-space skew. | Positive-flow support, log convention, skew limits, regional prior compatibility and rare-tail calculations. |
| Generalised extreme value, GEV | A flexible three-parameter family for block maxima, with different upper-tail behaviours. | Shape convention, support endpoints, zero-shape limit and probability assigned to physically inadmissible discharges. |
| Lognormal and Gumbel | Required legacy families, also useful for simpler models and limiting-case checks. | Match legacy parameter restrictions, zero-threshold applicability and supported estimators. |
| Generalised Pareto | Required legacy distribution family. | Match its actual location, scale, shape and probability conventions; do not substitute a complete POT workflow merely because the family is available. |

**Project recommendation:** compare plausible families rather than silently award a universal winner. Report the engineering implications of different upper tails. A statistically acceptable curve through observed floods does not prove that extrapolation beyond the record is physically sound.

### 3.1 LP3 logarithms and units

Use the dimensionless quantity $Y=\log_b(Q/q_{\mathrm{ref}})$, where the reference discharge fixes the recorded unit convention. Store the base $b$, reference unit, log-space mean, standard deviation and skew explicitly. The [`scipy.stats.pearson3` documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearson3.html) specifies its skew, location and scale interface; the LP3 wrapper must perform the discharge transformation.

Inspected sample reports name LP3 coordinates as **Mean(loge flow)**, **loge[Std dev(loge flow)]** and **Skew(loge flow)**, with a fourth coordinate **logit(P(flow <= zero threshold))**. Preserve those source coordinates in the compatibility profile. If the second coordinate is $\eta=\ln s$, the Pearson III scale is $s=\exp\eta$; averaging $s$ instead of $\eta$ changes the expected-parameter curve. The fourth label establishes a threshold-probability coordinate, not proof of an independently assumed literal-zero mixture. Its activation, estimation and interpretation require the paired reference evidence described in the sample audit.

Changing from base 10 to natural logarithms multiplies the mean and standard deviation by $\ln(10)$ while leaving skew unchanged. Converting both $Q$ and the same physical $q_{\mathrm{ref}}$ to another unit leaves their ratio unchanged. Resetting the reference convention to one of the new units instead changes $q_{\mathrm{ref}}$ and shifts the log mean. Record which operation was performed. Priors and their covariance must undergo any corresponding affine transformation; nonlinear transformations require preserving the full density rather than only adjusting moments.

For an exact discharge observation, the transformed density contains the Jacobian:

$$f_Q(q\mid\theta)=\frac{f_Y(\log_b(q/q_{\mathrm{ref}})\mid\theta)}{q\ln b},\qquad q>0.$$

This term may be constant with respect to some fitted parameters, but retaining a consistent density convention is necessary when comparing likelihoods across models or observation scales. Never insert a small positive replacement for an actual zero merely to permit logarithms.

### 3.2 GEV signs and support

For the convention $1+\xi(q-\mu)/\sigma>0$, with $\sigma>0$, SciPy's shape parameter is $c=-\xi$. [SciPy explicitly warns about opposite shape conventions](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.genextreme.html). Adapters must provide named conversions for each external convention, including ARR and FLIKE, rather than relying on an unlabelled `shape` value.

Test bounded and unbounded tails and the Gumbel limit. Reject invalid parameters; observations outside a valid model's support have zero likelihood. A prediction point outside support is still a valid CDF/SF evaluation: below the lower endpoint use $F=0,S=1$, and above the upper endpoint use $F=1,S=0$. The log density outside support is $-\infty$. This distinction is essential when posterior components have different endpoints. Do not silently truncate negative GEV support: truncation changes the probability model and needs its own method definition.

## 4 Observations and likelihood

Bayesian inference combines the likelihood $L(\theta;D)$ with a prior $\pi(\theta)$:

$$\pi(\theta\mid D)\propto L(\theta;D)\pi(\theta).$$

[Kuczera (1999)](https://doi.org/10.1029/1999WR900012) provides the flood-frequency foundation for combining gauged flows, censoring and rating uncertainty. ARR §§2.6.3.1–2.6.3.4, equations 3.2.40–3.2.49, provide the implementation reference points.

### 4.1 Individual annual observations

For a continuous model, independent observations contribute the following factors:

| Observation | Information actually available | Likelihood contribution |
| --- | --- | --- |
| Exact | The model treats the recorded peak $q$ as known. | $f(q\mid\theta)$ |
| Left censored | The annual peak was no greater than $u$. | $F(u\mid\theta)$ |
| Right censored | The annual peak exceeded $l$. | $S(l\mid\theta)$ |
| Interval censored | The annual peak lay between $l$ and $u$. | $F(u\mid\theta)-F(l\mid\theta)$ |
| Missing | No magnitude or bounding information is available. | No magnitude information; preserve the missing exposure metadata. |

The [`CensoredData` documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.CensoredData.html) confirms these observation categories and their availability for SciPy maximum-likelihood fitting. It does not supply a complete Bayesian flood analysis or historical-period model.

**Project requirements:** compute these contributions in log space, using stable tail and interval calculations. An interval is not equivalent to its midpoint. Uncertain measurement and censoring are different concepts: a stated error distribution should not automatically be replaced by arbitrary hard bounds.

### 4.2 Historical counts and nonoverlapping exposure

Suppose a reliably observed historical period contains $n$ years and exactly $k$ annual peaks above threshold $h$, but no exceedance magnitudes. Its count-only contribution is

$$L_{\mathrm{hist}}(\theta)={n\choose k}S(h\mid\theta)^kF(h\mid\theta)^{n-k}.$$

The threshold means an event above it would have been detected throughout that period. It is not merely the smallest known historical flood. The analyst must justify that perception assumption.

**Project requirements:** represent historical periods separately from individual records and validate their coverage. Do not multiply independent binomial terms for overlapping periods or thresholds. Split exposure into justified disjoint segments or derive a joint observation likelihood.

If the magnitudes of historical exceedances are also supplied, derive their joint contribution. Multiplying the count-only likelihood by their unconditional densities would count exceedance information twice. A conditional-density factor for an observed exceedance is $f(q)/S(h)$; the cancellation must be represented correctly. The same event must not also appear as an independent gauged observation.

The initial release should reject ambiguous overlap, changing thresholds without defined periods, and records with unestablished observation opportunities. Support can be expanded once the observation process is specified.

## 5 Bayesian numerical inference

The scientific target is the posterior distribution, not the output of a particular optimiser. ARR §2.6.3.6, equations 3.2.52–3.2.54, describes importance sampling. [Kuczera (1999)](https://doi.org/10.1029/1999WR900012) also uses Monte Carlo importance sampling; it should not be described as a generic MCMC specification.

**Binding MVP requirement:** reproduce the legacy importance-sampling calculation under matched likelihoods, priors, parameter coordinates and numerical controls. Public ARR/Kuczera mathematics supplies the foundation; sample reports and version-pinned reference runs establish compatibility details. Do not select MCMC, a new importance proposal or modified weighting as a production replacement simply because it appears more robust. Such improvements belong to an explicitly separate future method profile.

The [WP06 local mode and curvature builder](methods/importance-sampling-reference.md) specifies an explicit bounded Newton search and adaptive central differences for ARR steps 1–2; the separate sampler implements step 3. Neither establishes global mode, tail coverage or native controls.

For importance sampling, retain normalised weights, effective sample size, proposal settings and repeat-run stability. Reject results dominated by a few particles. A good posterior-mode fit does not establish that the proposal covers the posterior tail.

An independent MCMC or quadrature calculation may be used as a verification oracle. If MCMC is used for this purpose, retain chains and assess mixing, effective sample size, convergence and sampler-specific failures. Diagnostics must address reported rare quantiles as well as parameters. The oracle does not become the MVP's production method. Thresholds and run-extension policies require documented qualification; an arbitrary fixed sample count is not an accuracy guarantee.

Use reproducible random-number streams and record dependency versions. Changing variables for numerical convenience requires the prior-density Jacobian. An apparently flat prior in one parameterisation is generally not flat in another. An independent quadrature or alternative sampler check on tractable cases should verify the inference target before broad engineering qualification.

## 6 Quantile outputs must remain distinct

The quantities below answer different questions and must not be conflated. ARR §2.5 and §2.6.3.7, particularly equations 3.2.27–3.2.34 and 3.2.55–3.2.56, supply the reference framework. MVP outputs must reproduce the legacy report's actual estimands, defaults and options. Inspected sample reports include posterior parameter moments, an expected-parameter curve with 90% limits and expected-probability-at-flow output. The printed probability column is numerically consistent with non-exceedance probability $F$, accompanied by return period $1/(1-F)$; confirm this convention rather than relabelling the column directly as AEP. Pairing and precise semantics still require qualification. Additional posterior summaries are explanatory or verification quantities unless confirmed as target outputs. See the [sample-data audit](sample-data-audit.md).

### 6.1 Expected-parameter quantile

$$q_p^{\mathrm{EP}}=q_p\!\left(E[\theta\mid D]\right).$$

Compute posterior mean parameters in the legacy source parameterisation, then map that vector to the distribution adapter and evaluate the quantile. This is the meaning of ARR's expected-parameter output. For the inspected LP3 report convention, the scale coordinate is the logarithm of the log-flow standard deviation. An internal alternative coordinate must not replace the source coordinate when forming the mean. Qualification must establish the corresponding conventions for every family and the zero-threshold option.

### 6.2 Posterior mean quantile

$$\overline q_p=E[q_p(\theta)\mid D].$$

Here each parameter sample produces a quantile before averaging. Nonlinearity means it generally differs from the expected-parameter result. A posterior median quantile is another distinct summary. A finite Monte Carlo average does not establish that the underlying mathematical expectation exists; unstable tails must be flagged, not hidden by a run limit.

### 6.3 Expected-AEP or posterior predictive quantile

For continuous models in an attainable probability region, define $q_p^{\mathrm{EAEP}}$ by

$$E[S(q_p^{\mathrm{EAEP}}\mid\theta)\mid D]=p.$$

For posterior particles with weights summing to one, solve $\sum_iw_iS(q\mid\theta_i)=p$ in that continuous region. The general definition, also valid for models with atoms, is

$$F_{\mathrm{pred}}(q)=E[F(q\mid\theta)\mid D],\qquad q_p^{\mathrm{pred}}=\inf\{q:F_{\mathrm{pred}}(q)\geq1-p\}.$$

If an admitted target profile contains a discrete atom, a quantile can lie at that atom without satisfying a survival-probability equality. As a mathematical example, a literal-zero mass of 0.8 gives a zero median although the probability of exceeding zero is 0.2. This example is a numerical safeguard, not the specification of legacy Flike's zero-threshold treatment. In all cases, averaging probabilities and then inverting is different from averaging the particles' quantiles.

This estimator is useful when the interpretation centres on exceedance probability with parameter uncertainty integrated out. Preserve whether the target reports a non-exceedance probability, an AEP at a specified discharge, a quantile at a specified AEP, or several of these quantities. For a confirmed non-exceedance column, the corresponding AEP is its complement, not the original printed value. Do not replace one column by a newly inverted curve under the same label. The CLI must expose the replicated options and explain their meanings; a new design-estimator selection policy must not change legacy defaults.

### 6.4 Intervals and numerical uncertainty

For a fixed AEP, posterior draws of $q_p(\theta)$ define a credible interval for that quantile. This is different from an interval for a future annual flood and from Monte Carlo error in an estimated quantity. Bootstrap confidence intervals have another construction.

**Project requirements:** every interval identifies its target, probability level, construction and included uncertainties. The sample tables report moments of $\log_{10}(q)$ while LP3 parameter labels use natural logarithms; preserve these different conventions and the printed 90% probability limits, then qualify the exact construction. Label pointwise curve bands as pointwise; do not imply simultaneous coverage. Report parameter uncertainty conditional on the chosen model separately from distribution-choice, measurement and future-climate uncertainty. A library distribution's `interval` function is not automatically an interval for an estimated flood quantile.

## 7 Low flows, zeros and PILFs

Small annual peaks can arise from different processes from damaging floods and can strongly influence log-space fitting. [Cohn et al. (2013)](https://doi.org/10.1002/wrcr.20392) develops multiple Grubbs–Beck identification of potentially influential low flows. ARR §2.6.3.9 associates the test with visual appraisal and censoring.

**Binding MVP requirement:** replicate the legacy MGB variant, threshold construction, exclusion/censoring controls and retained-count behaviour. Keep an auditable record of the proposed and adopted treatment and any analyst override. Pin significance settings, ties, zero handling and reference fixtures; different published implementations must not be presumed equivalent. Additional sensitivity output can assist interpretation without changing the target calculation.

Censoring retains the information that a year's annual maximum was below the threshold. Deleting the year changes the estimated frequency of substantial floods. Preserve the original flow and observation status even when the fitting representation is censored.

An actual no-flow year is different from a measurement below detection, a value below a user-specified zero threshold, and a missing year. ARR §2.4.2.4 provides background on zero handling. The MVP must reproduce the legacy LN/LP3 zero-threshold option rather than introduce a new model.

The observed fourth LP3 parameter label is `logit(P(flow <= zero threshold))`. Establish from the paired input and reference output how the threshold is applied, which observations enter the positive/log-flow fit, whether threshold probability is fitted or fixed, its prior/default, and how uncertainty enters reported moments and curves. A label alone cannot establish these rules. Preserve this as an evidence gap until it is resolved; do not substitute a Beta-binomial prior, an empirical probability or an arbitrary small positive flow.

The following mathematical safeguards apply if the verified target model uses discrete mass: an exact atom contributes its probability rather than a continuous density, and intervals must retain endpoint inclusion. For example, $F(u)-F(l)$ represents $l<Q\leq u$; an interval including an atom at $l$ instead requires the corresponding left-limit CDF. A literal-zero atom would belong to $[0,u]$ but not $(0,u]$. These rules prevent numerical mistakes without presuming that all observations below the legacy zero threshold are literally zero. Qualify the actual supported family combinations and reject unsupported ones explicitly.

## 8 L-moments and LH-moments

[Hosking (1990)](https://doi.org/10.1111/j.2517-6161.1990.tb01775.x) establishes L-moments using linear combinations of order statistics. They provide a transparent fit comparator with different sensitivity to individual observations from ordinary product moments.

**Binding MVP requirement:** reproduce legacy H=0 L-moment fitting for all five supported families and qualify the permitted input combinations. Match whether moments are taken in discharge or transformed coordinates, the family inversion, output parameter convention and reported uncertainty. Do not assume that a general L-moment library also handles historical counts, regional priors or rating uncertainty. ARR §2.6.4 provides the relevant scope and equations 3.2.57–3.2.68.

[Wang (1997)](https://doi.org/10.1029/97WR02134) develops LH-moments to emphasise larger events; [Wang (1998)](https://doi.org/10.1029/98WR02364) develops corresponding fitted-GEV adequacy tests. Their use requires a declared shift/order and an assessment of the trade-off between upper-tail emphasis and reduced information. See ARR §2.6.4.5, equations 3.2.69–3.2.78.

GEV LH-moments with fixed H=1–4 and the legacy optimized-H procedure belong to the replication MVP. Reproduce the selection rule, diagnostics and uncertainty calculations rather than selecting H by a new goodness-of-fit score or a preferred design flow. The documented legacy restriction of H>=1 to GEV must remain explicit.

For uncertainty, establish and replicate the target's actual procedure. ARR §2.6.4.6 supplies the parametric-bootstrap foundation: simulate complete samples from the fitted model and refit using the same estimator. Record failures and determine whether method/H selection is repeated in each simulation. Bootstrap replicates must not be relabelled posterior samples or used to invent a Bayesian expected-probability output.

Plain product-moment fitting of logged flows is not the proposed Australian default; its legacy context is discussed in ARR §2.6.5. A future Bulletin 17C profile would implement its own specified Expected Moments Algorithm and related procedures, not simply rename the legacy estimator.

## 9 Regional priors and measurement uncertainty

**Binding MVP requirement:** reproduce the target's supplied-prior and no-prior modes, including LP3 regional-prior import where supported. Store parameter names, log base, units, mean vector, full covariance, source and applicability evidence. ARR §2.6.3.5, equations 3.2.50–3.2.51, and Chapter 3 §3.9 supply scientific guidance; the legacy profile supplies actual program settings and defaults. Do not require a newly chosen prior in place of the existing no-prior option. Resolve and document its effective density, bounds and numerical treatment before qualification. Automated regional-service integration is separate future work unless it is part of the established target.

Preserve the imported joint prior density under transformation, including Jacobians. For example, a Gaussian prior on $(m,\ln s,g)$ is not Gaussian on $(m,s,g)$; silently replacing it by a moment-matched Gaussian changes the model. An ARR reference expected-parameter output must average in its source-specified coordinates and then transform to the distribution adapter, rather than averaging whichever internal coordinates are convenient.

Assess whether the target gauge contributed to the regional estimate. Reusing dependent information as an independent prior can exaggerate precision. Compare prior and likelihood implications and provide sensitivity results. Do not invent a regional prior from an unrelated region. If the legacy no-prior mode uses large Gaussian variances or other numerical bounds, reproduce the verified values and explain their consequences; do not silently replace them with a supposedly better prior.

The legacy rating-error facility is part of the MVP and needs a dedicated qualification package. Errors shared by a rating segment can affect many floods together; independent perturbation of every flow does not represent that dependence. Map the target's error controls and defaults to the specified measurement likelihood, including any groups, thresholds and covariance. Record the input interpretation and compare enabled/disabled reference cases against ARR §§2.3.7 and 2.6.3.4 and the Kuczera formulation. Lack of a reference case is a coverage gap, not a reason to relabel this legacy feature as WMAwater future scope.

## 10 Diagnostics and engineering judgement

**Project requirements:** make the following reviewable before reporting an adopted design estimate:

- Data provenance, gaps, duplicates, units, year coverage and gauge/rating changes.
- Chronological plots and evidence about regulation, land-use change, flood mechanisms, independence and stationarity.
- Distribution fit and upper-tail behaviour, including sensitivity to high observations, censoring and priors.
- Numerical convergence, support violations, unstable moments and failed bootstrap fits.
- Differences between plausible models and estimands, plus extrapolation beyond the observed record.

A nonsignificant trend test does not prove stationarity. Short records can have little power to detect material change. Historical knowledge and physical catchment changes must remain part of the assessment.

Replicate the target's plotting-position convention for each admitted observation type. ARR §2.6.2, equations 3.2.35–3.2.36, provides the Cunnane foundation for complete systematic records and identifies the additional complexity of historical/censored data. A temporary systematic-only diagnostic with threshold/count annotations must be labelled as such, not counted as full plotting parity. Do not re-rank only retained uncensored peaks with a smaller record length or insert historical peaks into ordinary gauged ranks. Keep plotted ranks separate from model-derived event AEPs. Goodness-of-fit calibration must account for parameter estimation and any selection procedure; a test for a completely specified distribution cannot automatically validate a fitted distribution. No automated score should erase hydrological concerns about the rare tail.

## 11 WMAwater and other future extensions

The initial stationary result describes its declared historical/reference conditions. It must not automatically be presented as a future-climate design estimate.

WMAwater's new distributions, climate covariates, updated priors, uncertainty methods and production MCMC are future scope. They cannot redefine the TUFLOW/BMT compatibility target. A GUI, desktop front end, QGIS interface and web application are excluded at every phase; future methods must remain accessible through the command-line workflow if approved.

[ARR V4.2 Book 1 §6.3](https://www.arr-software.org/pdfs/ARR_190514_Book1_V4.2.pdf), printed page 162, identifies unresolved issues in adjusting direct flood-frequency estimates. **Project recommendation:** no generic multiplication of FFA discharges by rainfall climate factors. A later climate method must explain how changing flood drivers, observation period and future assumptions determine its parameters.

Covariate models require declared covariate data and future scenarios, a treatment of parameter uncertainty, and separate conditional and marginal risk outputs. For independent years with varying exceedance probabilities $p_t$, service-life exceedance risk is $1-\prod_t(1-p_t)$; constant-return-period language can conceal this change. These features require additional qualification, not an extra input field alone.

## 12 Accessible method records and evidence

Each method must have an accessible explanation covering: the engineering question; when to use it; assumptions; equations; inputs; numerical implementation; outputs; diagnostics; limitations; references; and qualification evidence. The report should link to the exact version used and include the analyst's adopted choices.

**Source inspection status:** ARR Books 1 and 3 and the SciPy pages linked above were inspected. The publisher's Kuczera (1999), Hosking (1990), and Wang (1997, 1998) abstracts were inspected; acquire and review their full texts before implementing details not reproduced in ARR. The Cohn et al. (2013) publisher full text was inspected. [Micevski and Kuczera (2009)](https://doi.org/10.1029/2008WR007173), a relevant regional-combination foundation, was verified bibliographically in ARR; its full text remains to be acquired.

The [official USGS Bulletin 17C summary](https://www.usgs.gov/publications/guidelines-determining-flood-flow-frequency-bulletin-17c) was inspected. Full version 1.1, May 2019, and its applicable examples must be reviewed before claiming that method's implementation. EMA is a separate estimator; agreement with it is not a universal acceptance test for ARR Bayesian results.

Use ARR §2.8 Examples 3–11 as a planned benchmark sequence covering gauged, historical, prior, PILF, poor-fit, nonhomogeneous, L-moment, LH-moment and POT cases. Obtain original data and settings before assigning numerical tolerances. Published examples are validation targets, not evidence that this proposed package has already passed validation.
