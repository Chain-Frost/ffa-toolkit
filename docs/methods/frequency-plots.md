# Exact systematic annual maxima frequency plots

**Method ID:** `frequency-plots` — independent bounded static presentation, version 1.
Native Flike comparison, model suitability and engineering adoption are not established.

## Purpose and interpretation

Compare observed annual peaks with the fitted design quantiles. A plotting position is a
rank-based visual diagnostic, not the actual flood probability inferred from a distribution.
Adjacent ranks are statistically dependent; a visual fit is not an independent goodness-of-fit test.

Use the Cunnane position from ARR Book 3 v4.2 Chapter 2 section 2.6.2,
printed pp. 35–36, equations (3.2.35) and (3.2.36):
`AEP_i = (i - 0.4)/(n + 0.2)`, `F_i = 1 - AEP_i`. Here n is the number of
supplied gauged annual maxima and i=1 is the greatest discharge. Equal flows receive
consecutive ranks ordered by ascending year; this is a project presentation decision, not
a native convention or an additional statistical estimator. Rank order uses the original decimal value before conversion to binary64 for the plot.
The source snapshot retains the exact recorded decimal, and the coordinate table retains the
plotted value and source record identity.

The horizontal coordinate is proportional to -log10(AEP), with 100% at the left
and rarer events to the right. For spans of up to three decades, intermediate 2/5-decade
AEP labels aid reading; open fitted markers keep coincident observed points visible.
The vertical axis is linear in the supplied discharge unit
and includes zero and signed fitted quantiles; it does not clip nonphysical predictions.
The deterministic graph shows only the actual requested fitted quantiles; line segments
are visual guides, not interpolated calculations. The point-fit route calculates no intervals.
The exact-record Bayesian route presents already calculated posterior intervals as described below. Support warnings remain visible; refer to the full
analysis and check assumptions before adopting a design discharge.

## Offline worked examples

Use the already bundled, original synthetic examples for each installed deterministic path:

~~~console
ffa fit --ams fixed-gev-ams.csv --settings fixed-gev-settings.json --output gev-plot --frequency-report
ffa fit --ams ordinary-ams.csv --settings ordinary-settings.json --output gumbel-plot --frequency-report
ffa explain frequency-plots --format markdown
ffa manual > method-manual.html
~~~

Copy the source resources from an installed wheel using
`importlib.resources.files("ffa_methods").joinpath("resources", "examples")`.
For fixed GEV, select each explicit H=0–4 in a separate settings file and output directory.
For ordinary LN2, LP3 and GP fixed/free location, use the exact changes in
`ffa explain l-moments --format markdown`; retain each distinct settings file and output directory.

For the three-point ordinary sample (2000:1, 2001:2, 2002:4), descending ranks 1,2,3
have AEP 3/16, 1/2, 13/16; F is the complement. These are independently
hand-calculable plotting positions. Fitted results are the actual run values and are
not additional empirical positions. The deterministic report JSON and CSV are the authoritative
numerical records; their requested probabilities, discharge values and units are unchanged.

The opt-in report adds `frequency.svg` (standalone vector graphic), `frequency-coordinates.csv`
(full numeric F, AEP, Q, row, year and rank plus SVG pixel coordinates), and `report.html`
(self-contained, printable, with the full numerical text report and systematic data table).
It also retains byte-exact input and settings snapshots for both deterministic schemas.
Use an existing, user-controlled parent and a new output directory. Files are staged before
publication. On POSIX the final existence check and rename are separate operations, so concurrent
external creation of an empty directory at the same destination is outside the safety guarantee.
Without the flag, previous outputs and schemas remain unchanged.

## Admission and limitations

The existing explicit deterministic ordinary and fixed-H GEV schemas, and the explicit
lognormal Bayesian Year,AMS schema, require complete, consecutive, positive exact systematic annual maxima. The caller's classification,
year definition and unit remain assertions. The plot refuses unresolved gaps rather than treating
them as measured exposure or non-exceedances. Existing unplotted fits are not restricted further.
No missing, zero, historical, grouped, censored, rating-error or POT plot is admitted here.
Selected-H and parametric-bootstrap visualization remain separate pending work. An annual-maxima GP fit is not a POT workflow. Independence, stationarity,
physical support, model selection, data quality, rating error and engineering adoption still
require review. No native labels or defaults are inferred.

**Sources:** ARR 2019, Book 3 v4.2, section 2.6.2, pp. 35–36, Commonwealth of Australia
(Geoscience Australia), 2019; Cunnane (1978), Journal of Hydrology 37, 205–222,
DOI 10.1016/0022-1694(78)90017-3 (publisher abstract reviewed; full paper not obtained).
Use `ffa references ARR3` and `ffa references CUNNANE1978`.
Independent numeric and installed checks are required before acceptance; native qualification is separate.

## Exact-record Bayesian frequency reporting

The installed explicit lognormal Bayesian schema `ffa-lognormal-bayesian/1` supports
`--frequency-report` for `source_format: year_ams_csv` and consecutive positive exact
systematic annual maxima. Classified-systematic CSV (including censored records) is
intentionally not admitted by this increment. A quantile request is required;
probability-at-flow-only output remains a numerical table.

The renderer consumes the typed quantile results and their canonical report rows, without
running another fit, sampling additional draws or substituting different estimands.
The separately labelled series are quantile at expected source-coordinate parameters,
posterior mean of quantiles, posterior median of quantiles, and posterior predictive
quantile (the inverse averaged CDF). Undefined posterior means remain absent; segments
do not bridge missing requested values. The exact recorded lower and upper central
posterior quantile probability limits are vertical intervals at their own requested F.
These are parameter-induced uncertainty in the model quantile, not predictive intervals,
simultaneous bands, bootstrap intervals or numerical Monte Carlo error. Interval
mass and construction, source unit, canonical one-based result-row identity and each
reported numerical precision state are retained in the coordinate CSV and HTML.
No new precision claim follows from a plot. The other Bayesian probability and
return-period quantities remain in the complete numerical report.

Run the bundled synthetic fixed-scale example (copy the two resources from the wheel
as for the deterministic examples):
~~~console
ffa fit --ams bayesian-fixed-ams.csv --settings bayesian-fixed-settings.json --output bayesian-plot --frequency-report
~~~
For its three log flows 1,2,3, normal prior N(0,1) and known log SD 0 (SD=1),
the independent posterior log-location reference is N(1.5,1/4). At F=0.9 and
z=Phi^-1(F), the at-expected-parameter quantile is exp(1.5+z), the posterior
mean quantile exp(1.5+z+0.125), the posterior median quantile exp(1.5+z), and
the predictive quantile exp(1.5+sqrt(1.25)*z). The central 80% posterior
quantile interval endpoints are exp(1.5+z+0.5*Phi^-1(0.1)) and
exp(1.5+z+0.5*Phi^-1(0.9)). These analytical values are reference quantities;
the plotted values are the actual finite seeded importance-sampling outputs.
The configured AEP precision audit does not qualify those quantile points.
See `ffa explain quantile-estimands --format markdown` and the installed Bayesian
workflow guidance for the source method and limitations. Native qualification remains separate.
