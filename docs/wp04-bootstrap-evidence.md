# WP04 parametric bootstrap evidence

The worked record consists of 50 synthetic annual maxima with year `1950 + i`
and discharge `20 - log(-log((i - 0.35) / 50.3))` m³/s for `i = 1, …, 50`.
The input is complete and exact. Request non-exceedance probability 0.9,
central percentile confidence 0.8, MT19937 seed 42, maximum failure fraction 0,
and endpoint tolerance 1 m³/s. The ordinary and fixed-shift examples attempt 200
replicates each.

Use `ffa fit --ams annual.csv --settings settings.json --output report` with
the `Year,AMS` CSV header, `calendar` year definition and source classification
`caller_asserted_complete_exact_positive_annual_maxima`. The common settings
are `discharge_unit` equal to `"m3/s"` and `non_exceedance_probabilities`
equal to `[0.9]`. Use `ffa-ordinary-lmoments/1` with estimator
`gumbel_lmoments`, or `ffa-gev-fixed-h/1` with estimator `gev_lh_fixed_h`
and `lh_shift` equal to 1. For selection use `ffa-gev-selected-h/1` with
estimator `gev_lh_selected_h` and provide `critical_z` and
`calibration_seed` in place of `lh_shift`. Put `replications`, `seed`,
`confidence`, `maximum_failure_fraction`, and `endpoint_tolerance` in
the `bootstrap` settings object.

| Workflow | Successful / attempted | Point discharge | Percentile interval | Monte Carlo bracket for lower endpoint | Monte Carlo bracket for upper endpoint |
| --- | ---: | ---: | --- | --- | --- |
| Ordinary Gumbel L-moments | 200 / 200 | 22.216673 | [21.737584, 22.684563] | [21.660141, 21.813455] | [22.612114, 22.786193] |
| Fixed GEV H=1 | 200 / 200 | 22.217968 | [21.745864, 22.653794] | [21.653070, 21.780837] | [22.611299, 22.792447] |

The ordinary reference computes the location and scale from the mean and
the linear L-scale rank weights. The fixed H=1 reference computes sample LH
moments from direct rank weights, obtains population LH moments by expanding
each order-statistic CDF into powers of the parent CDF and integrating the
expected maxima analytically, and solves the skew-matching equation. Both
references draw the complete records from their independently fitted laws,
repeat the same estimator and compute the deciles with the inclusive sample
quantile rule. The focused CLI regression checks the point and both percentile
limits against these references. A separate synthetic test checks the numerical
brackets against direct sums of binomial probabilities.

Using the selected-H schema on the same record with cutoff 1.96 and calibration
base seed 7 selects H=0 and gives point discharge 22.216606 m³/s.
An intentionally insufficient budget of four replicates reselected H=0 all
four times, recorded each attempt and refused interval endpoints with
`endpoint_count`. A separately declared budget of 48, using the same settings
and seeds, took 283 seconds with concurrent validation. Its first four
attempts are the four-attempt example. Attempt 7 selected H=1
and attempt 11 failed during refit because the H=4 calibration uncertainty
straddled the cutoff. Each of the other 46 attempts selected H=0. The report
retained all 48 attempts in order and refused the interval with
`failure_fraction` because 1/48 exceeds the declared zero failure tolerance.
The point remains available in both reports. Both runs refused publication of
an 80% interval. The installed smoke separately exercises interval publication
using the existing 24-attempt configuration with 50% confidence, maximum
failure fraction 0.25 and endpoint tolerance 2 m³/s. It uses the same record
and seeds and checks that the installed CLI emits both limits and all 24
attempts for that configuration.

The bootstrap intervals target
the distribution conditional on a successful refit and, for selected H, on
successful reselection. The endpoint brackets measure Monte Carlo error in
estimating the two percentile limits. Failure and precision gates apply before
any interval endpoints are emitted.

The numerical examples were run on Python 3.14. The synthetic controlled
failure test also verifies the order of refits, the selected H on successful
reselection and on a subsequent quantile failure, preservation of refusal
reasons, and the percentile results from the successful records.
