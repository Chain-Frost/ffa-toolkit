# Scientific and implementation reference register

Research snapshot: 23 November 2026. These references support the proposed implementation. They do not constitute evidence that this package has implemented or passed any method.

The binding MVP is the CLI replication profile `tuflow-bmt-flike-5.0.306.0-file3.10`, defined by the [compatibility contract](compatibility-contract.md). ARR and primary literature establish scientific foundations; observed legacy files establish target labels, settings and behavior to the extent verified. Keep those evidence roles separate. New production algorithms and WMAwater additions require distinct future profiles; GUI development is outside scope at every phase.

Use the source keys below in method metadata and qualification records. Keep guidance, original mathematical work, library documentation and software comparisons distinguishable. Record exact versions and retrieved artefact hashes when sources become numerical fixtures. Do not rely on a moving `latest` documentation URL as the only record of a released dependency's behaviour.

## Australian guidance

### ARR3

Ball, J., Babister, M., Nathan, R., Weeks, W., Weinmann, E., Retallick, M. and Testoni, I. (editors). *Australian Rainfall and Runoff: A Guide to Flood Estimation*, Version 4.2, 2019. Commonwealth of Australia (Geoscience Australia). [Book 3 Peak Flow Estimation](https://www.arr-software.org/pdfs/ARR_190514_Book3_V4.2.pdf).

Chapter 2, *At-Site Flood Frequency Analysis*, is by George Kuczera and Stewart Franks; its chapter date is 14 May 2019. The volume version does not imply that every chapter was rewritten during the later climate update. Full relevant guidance was inspected.

| Implementation topic | Exact starting reference |
| --- | --- |
| Scope and independent software | Chapter 2 section 2.1 |
| Data suitability and annual/event series | Sections 2.2 and 2.3 |
| Distribution choices and zeros | Sections 2.4.2 and 2.4.2.4 |
| Different design quantile estimands | Section 2.5, equations 3.2.25â€“3.2.34 |
| Probability plots and plotting-position limitations | Section 2.6.2, equations 3.2.35â€“3.2.36 |
| Exact/censored likelihood and rating errors | Sections 2.6.3.1â€“2.6.3.4, equations 3.2.40â€“3.2.49 |
| Regional priors | Section 2.6.3.5, equations 3.2.50â€“3.2.51; Chapter 3 section 3.9 |
| Importance sampling | Section 2.6.3.6, equations 3.2.52â€“3.2.54, printed pp. 41â€“42 (PDF pp. 58â€“59) |
| Weighted uncertainty and expected probability | Section 2.6.3.7, equations 3.2.55â€“3.2.56, printed p. 43 (PDF p. 60) |
| Low-flow treatment | Sections 2.6.3.8â€“2.6.3.9 |
| L-moments and LH-moments | Section 2.6.4, especially 2.6.4.5 |
| Bootstrap uncertainty | Section 2.6.4.6 |
| POT | Section 2.7 |
| Worked numerical/software examples | Section 2.8, Examples 3â€“11 |
| Legacy Flike LH family restriction | Section 2.8.10.3 |

WP06 #47 uses the modal Taylor approximation and inverse negative Hessian on printed p. 42, with a separately specified local Newton search and adaptive central differences. The source recommends SCE as a search option; these numerical settings are not native defaults. See the [offline method explanation](methods/importance-sampling-reference.md).

ARR is distributed under Creative Commons Attribution 4.0 except for material otherwise indicated or marked. Attribute the Commonwealth of Australia (Geoscience Australia), 2019, and retain separate permissions/provenance for third-party material and datasets. The project should maintain original explanations rather than redistribute large extracts or assume that a linked software asset has the same licence.

### ARR1

The same edited volume series, [Book 1 Scope and Philosophy, Version 4.2](https://www.arr-software.org/pdfs/ARR_190514_Book1_V4.2.pdf). Chapter 6 contains the 2024 climate update. Section 6.3, printed page 162, is the starting point for direct flood-frequency estimation under climate change. Relevant guidance was inspected.

**Project role:** qualify the scope of stationary results and define the research boundary for future climate/covariate methods. A rainfall change factor is not automatically a valid adjustment to an FFA discharge.

## Foundational methods

### KUCZERA1999

Kuczera, G. (1999). *Comprehensive at-site flood frequency analysis using Monte Carlo Bayesian inference*. Water Resources Research, 35(5), 1551â€“1557. [DOI 10.1029/1999WR900012](https://doi.org/10.1029/1999WR900012).

**Role:** Bayesian flood likelihood, gauged and censored data, expected probability and importance sampling. Equations (16)-(17), p. 1554, specify the inverse negative log-posterior Hessian and explicit proposal SD scaling. The ARR presentation supplies an additional mathematical foundation. Exact native priors, defaults and numerical controls require separate qualification evidence; no other detail is admitted by this scaling review.

### KUCZERA1996

Kuczera, G. (1996). *Correlated rating curve error in flood frequency inference*. Water Resources Research, 32(7), 2119â€“2127. [DOI 10.1029/96WR00804](https://doi.org/10.1029/96WR00804).

**Role:** correlated measurement error in rating extensions. Legacy rating-error behavior is required MVP scope under WP05 [development requirements](https://github.com/Chain-Frost/ffa-toolkit/blob/main/docs/development.md). Qualification requires full-text review, the target's exact controls and matched acceptance cases; independent random jitter of peaks is not the specified shared-error model.

### HOSKING1990

Hosking, J.R.M. (1990). *L-Moments: Analysis and Estimation of Distributions Using Linear Combinations of Order Statistics*. Journal of the Royal Statistical Society, Series B, 52(1), 105â€“124. [DOI 10.1111/j.2517-6161.1990.tb01775.x](https://doi.org/10.1111/j.2517-6161.1990.tb01775.x).

**Role:** L-moment foundations and estimator definitions. Publisher abstract inspected; ARR and library documentation provide additional accessible derivations. Full-text algorithm review is required where the implementation depends on details outside those sources.

### WANG1997

Wang, Q.J. (1997). *LH moments for statistical analysis of extreme events*. Water Resources Research, 33(12), 2841â€“2848. [DOI 10.1029/97WR02134](https://doi.org/10.1029/97WR02134).

**Role:** LH-moment estimator foundations. Legacy GEV LH-moments, including fixed H=1â€“4 and optimized H, are MVP scope under WP04 [development requirements](https://github.com/Chain-Frost/ffa-toolkit/blob/main/docs/development.md). The publisher abstract and ARR treatment were inspected; full numerical formulations and target-build examples still require review before qualification.

### WANG1998

Wang, Q.J. (1998). *Approximate goodness-of-fit tests of fitted generalized extreme value distributions using LH moments*. Water Resources Research, 34(12), 3497â€“3502. [DOI 10.1029/98WR02364](https://doi.org/10.1029/98WR02364).

**Role:** statistical assessment used for a GEV LH procedure. The publisher abstract and ARR treatment were inspected. Review the full formulation and match the legacy optimized-H selection, applicable uncertainty and failure behavior under WP04 before qualification.

### COHN2013

Cohn, T.A., England, J.F., Berenbrock, C.E., Mason, R.R., Stedinger, J.R. and Lamontagne, J.R. (2013). *A generalized Grubbs-Beck test statistic for detecting multiple potentially influential low outliers in flood series*. Water Resources Research, 49(8), 5047â€“5058. [DOI 10.1002/wrcr.20392](https://doi.org/10.1002/wrcr.20392).

**Role:** multiple low-flow identification and censoring. Publisher full text inspected. Pin the target's precise algorithm variant and settings before coding. The [USGS MGBT reference implementation](https://code.usgs.gov/water/stats/MGBT/-/tree/v1.0.3) is a candidate independent comparison; record a verified version, licence and fixture outputs during qualification. The cited paper or comparator is not, by itself, evidence of the exact variant implemented by the pinned Flike build.

### MICEVSKI2009

Micevski, T. and Kuczera, G. (2009). *Combining site and regional flood information using a Bayesian Monte Carlo approach*. Water Resources Research, 45, W04405. [DOI 10.1029/2008WR007173](https://doi.org/10.1029/2008WR007173).

**Role:** regional/at-site combination. Bibliographic details were checked in ARR; full text was not acquired. Reproducing an imported regional or supplied prior is MVP WP05 work and requires exact target coordinates, covariance and density. Constructing priors through a new regional model-fitting system is separate future WP10 work. This citation alone establishes neither implementation.

### BULLETIN17C

England, J.F. Jr. and others (2018; version 1.1, May 2019). *Guidelines for Determining Flood Flow Frequencyâ€”Bulletin 17C*. USGS Techniques and Methods, 4-B5. [Official publication page](https://www.usgs.gov/publications/guidelines-determining-flood-flow-frequency-bulletin-17c), [DOI 10.3133/tm4B5](https://doi.org/10.3133/tm4B5).

**Role:** a separately specified future US method profile, interval/censored data representation, EMA, MGBT and external numerical examples. The official summary was inspected; acquire and pin the full current document and Appendix 10 datasets before claiming conformance. EMA is not Bayesian inference. Reproducing a legacy low-flow test does not by itself establish Bulletin 17C conformance.

## Numerical verification and diagnostics

### SBC

Talts, S., Betancourt, M., Simpson, D., Vehtari, A. and Gelman, A. *Validating Bayesian Inference Algorithms with Simulation-Based Calibration*. [arXiv 1804.06788](https://arxiv.org/abs/1804.06788), version 2 (2020).

**Role:** checking computational inference against a full generative and observation model. The published preprint abstract was inspected. Full-method review and implementation-specific checks are part of WP09; a rank plot alone does not establish hydrological suitability.

### RHAT

Vehtari, A., Gelman, A., Simpson, D., Carpenter, B. and BÃ¼rkner, P.-C. (2021). *Rank-normalization, folding, and localization: an improved R-hat for assessing convergence of MCMC*. Bayesian Analysis, 16(2), 667â€“718. [DOI 10.1214/20-BA1221](https://doi.org/10.1214/20-BA1221); [author preprint](https://arxiv.org/abs/1903.08008).

**Role:** appropriate MCMC convergence and quantile precision diagnostics. Applicability depends on the sampler and independent-chain structure. The abstract and official emcee diagnostic guidance were inspected; numerical gates require project qualification. Independent MCMC may support MVP verification; a production MCMC replacement is future scope and cannot silently replace legacy importance sampling.

### PSIS

Vehtari, A., Simpson, D., Gelman, A., Yao, Y. and Gabry, J. (2024). *Pareto Smoothed Importance Sampling*. Journal of Machine Learning Research, 25(72), 1â€“58. [Publication and open paper](https://www.jmlr.org/papers/v25/19-556.html).

**Role:** a diagnostic foundation and possible future importance-sampling enhancement. Abstract inspected. Any smoothing changes the finite-sample calculation and must be specified and qualified separately from the legacy replication engine.

## Library contracts

### SCIPY

Official documentation for [GEV](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.genextreme.html), [Pearson III](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearson3.html), [lognormal](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.lognorm.html), [Gumbel](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gumbel_r.html), [GPD](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.genpareto.html), [CensoredData](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.CensoredData.html), [distribution fitting](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.fit.html) and [sample L-moments](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.lmoment.html).

**Role:** API and numerical behaviour, not hydrological method approval. All five legacy kernelsâ€”lognormal, LP3, Gumbel, GEV and GPâ€”belong to MVP WP03 [development requirements](https://github.com/Chain-Frost/ffa-toolkit/blob/main/docs/development.md). GP kernel support does not establish a complete POT event/exposure workflow. The method documents explain the key sign, support, log transformation, censoring and interval distinctions. Pin the installed release and matching documentation before qualification; a library's fit defaults do not establish the legacy defaults.

### NUMPY

Official [random-number compatibility policy](https://numpy.org/doc/stable/reference/random/compatibility.html) and [parallel random streams](https://numpy.org/doc/stable/reference/random/parallel.html).

**Role:** reproducibility contract. Seed, bit generator, build, method-call pattern and environment can all matter; retaining only a seed is insufficient for strict replay.

### BAYES_LIBRARIES

Official [emcee documentation](https://emcee.readthedocs.io/en/stable/), [autocorrelation and convergence guidance](https://emcee.readthedocs.io/en/stable/tutorials/autocorr/) and [PyMC external-likelihood guidance](https://www.pymc.io/projects/examples/en/latest/howto/blackbox_external_likelihood_numpy.html).

**Role:** independent verification backends and possible separately qualified future production algorithms. The MVP production target remains legacy importance sampling. Do not treat emcee walkers as independent chains for R-hat, or assume a NumPy/SciPy likelihood automatically supplies NUTS-compatible gradients.

### LMOMENT_LIBRARIES

[lmoments3 documentation and licence](https://lmoments3.readthedocs.io/stable/readme.html); [Lmo repository](https://github.com/jorenham/Lmo).

**Role:** comparative implementations and, where permitted, dependency candidates. The inspected lmoments3 documentation identifies GPL v3-or-later; it is restricted to a separate external-comparator role by the project's fixed Sustainable Use License 1.0 policy. Lmo identifies BSD-3-Clause but is not admitted merely by being listed. Review each chosen version, provenance, numerical suitability and dependency obligations; retain required notices under [the agent licence rules](../AGENTS.md#licence-and-third-party-material).

## Product and independent comparison sources

### LEGACY_SAMPLES

Owner-authorized identity-sanitized scenario derivatives. See [admission](sample-data-audit.md). Format and report-internal mathematical evidence only; no settings-matched native qualification.

### FLIKE_STATUS

[TUFLOW downloads](https://www.tuflow.com/downloads/) and [WMAwater Flike roadmap](https://flike.wmawater.com.au/), inspected 23 November 2026. **Role:** current product routing and announced future developments. A roadmap label is not proof of a publicly released feature or full technical specification. WMAwater additions are future WP10 [development requirements](https://github.com/Chain-Frost/ffa-toolkit/blob/main/docs/development.md) scope and do not redefine the pinned legacy target.

### RMC_VERIFICATION

Smith, C.H. (2020). *Verification of the Bayesian Estimation and Fitting Software (RMC-BestFit)*. US Army Corps of Engineers, Risk Management Center, RMC-TR-2020-02. [Author-uploaded report](https://www.researchgate.net/publication/352414326_Verification_of_the_Bayesian_Estimation_and_Fitting_Software_RMC-BestFit), especially the Flike comparisons on printed pages 66â€“69.

**Role:** a published external verification approach and candidate reference cases. Relevant comparison text was inspected. Reconcile the report's curve terminology and exact settings with the mathematical estimands before adopting numerical expectations.

### RMC_SOFTWARE

[Official USACE-RMC/RMC-BestFit repository](https://github.com/USACE-RMC/RMC-BestFit) and [LP3 technical reference](https://github.com/USACE-RMC/RMC-BestFit/blob/main/docs/technical-reference/distributions/log-pearson-type-iii.md).

**Role:** independent software and useful validation specifications. Pin a source commit and inspect its actual configuration before comparing results. This is a comparator, not a required Python runtime dependency or automatic source of validation truth.

### EXTERNAL_REUSE_AUDIT

See [third-party admission](../THIRD_PARTY_NOTICES.md) and [retained mathematical vectors](../reference/public-candidates/numerics-vectors/README.md). External gauge datasets with unresolved rights are excluded; source references do not establish redistribution permission.

### OWNER_SETUP

Engineering policy is defined in [repository setup](repository-setup.md), package metadata and [development](development.md). Historical repository comparisons are not public scientific evidence.
