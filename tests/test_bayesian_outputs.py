"""Synthetic, independently calculable WP06 weighted-estimand verification."""

from math import exp, expm1, inf, isclose, log, log1p, log10, nextafter, pi, sqrt
from statistics import NormalDist

import pytest

from ffa_methods.bayesian_outputs import (
    ImportanceParticles,
    posterior_moments,
    predictive_quantile,
    probability_outputs,
    quantile_outputs,
    require_weight_adequacy,
    tail_effective_sample_size,
    weighted_quantile,
)
from ffa_methods.contracts import EvidenceStatus, RunProvenance


def near(expected: float | tuple[float, ...] | list[float]) -> object:
    """Isolate pytest's incomplete `approx` type stub for strict Pyright."""
    return pytest.approx(expected)  # pyright: ignore[reportUnknownMemberType]


class ExponentialKernel:
    """Independent analytic exponential CDF/PPF with log-scale parameter."""

    def __init__(self, log_scale: float) -> None:
        self.scale = exp(log_scale)

    def cdf(self, discharge: float) -> float:
        return -expm1(-discharge / self.scale) if discharge >= 0 else 0.0

    def sf(self, discharge: float) -> float:
        return exp(-discharge / self.scale) if discharge >= 0 else 1.0

    def logpdf(self, discharge: float) -> float:
        return -log(self.scale) - discharge / self.scale if discharge >= 0 else -inf

    def logcdf(self, discharge: float) -> float:
        cdf = self.cdf(discharge)
        return log(cdf) if cdf > 0 else -inf

    def logsf(self, discharge: float) -> float:
        return -discharge / self.scale if discharge >= 0 else 0.0

    def isf(self, survival_probability: float) -> float:
        return -self.scale * log(survival_probability)

    def ppf(self, non_exceedance_probability: float) -> float:
        return -self.scale * log1p(-non_exceedance_probability)


class AtomKernel:
    """Analytic zero-mass component; never assumed to be FLIKE's actual model."""

    def __init__(self, mass: float) -> None:
        self.mass = mass

    def cdf(self, discharge: float) -> float:
        if discharge < 0:
            return 0.0
        return self.mass + (1.0 - self.mass) * (-expm1(-discharge))

    def sf(self, discharge: float) -> float:
        if discharge < 0:
            return 1.0
        return (1.0 - self.mass) * exp(-discharge)

    def logpdf(self, discharge: float) -> float:
        return log(1.0 - self.mass) - discharge if discharge > 0 else -inf

    def logcdf(self, discharge: float) -> float:
        cdf = self.cdf(discharge)
        return log(cdf) if cdf > 0 else -inf

    def logsf(self, discharge: float) -> float:
        sf = self.sf(discharge)
        return log(sf) if sf > 0 else -inf

    def isf(self, survival_probability: float) -> float:
        return self.ppf(1.0 - survival_probability)

    def ppf(self, non_exceedance_probability: float) -> float:
        if non_exceedance_probability <= self.mass:
            return 0.0
        return -log((1.0 - non_exceedance_probability) / (1.0 - self.mass))


def particles(
    *, coords: tuple[str, ...] = ("log_scale",), draws: tuple[tuple[float, ...], ...], log_weights: tuple[float, ...]
) -> ImportanceParticles:
    """Set prior and proposal densities equal: exp(log_likelihood) gives weights."""
    n = len(draws)
    return ImportanceParticles(
        coordinates=coords,
        draws=draws,
        log_likelihood=log_weights,
        log_prior=(0.0,) * n,
        log_proposal=(0.0,) * n,
        provenance=RunProvenance(
            compatibility_profile="synthetic-analytic",
            method_id="importance-weighting-invariant",
            method_version="test-only",
            package_version="0.0.0.dev0",
            evidence_status=EvidenceStatus.OBSERVED_ARTIFACT,
        ),
        proposal_description="synthetic known proposal, not a legacy sampler",
        random_stream_description="no RNG: deterministic weighted test",
    )


def test_normalised_log_weights_extreme_offsets_and_zero_target() -> None:
    p = particles(draws=((1.0,), (2.0,), (3.0,)), log_weights=(-1000.0, -1000.0 - log(2.0), -inf))
    assert p.weights == near((2 / 3, 1 / 3, 0.0))
    m = posterior_moments(p)
    assert m.effective_sample_size == near(1.8)
    assert m.zero_weight_count == 1
    assert m.largest_weight == near(2 / 3)
    assert m.entropy_effective_size == near(exp(-(2 / 3 * log(2 / 3) + 1 / 3 * log(1 / 3))))
    with pytest.raises(ValueError, match="fails"):
        require_weight_adequacy(p, minimum_effective_size=2.0, maximum_single_weight=0.9)
    assert require_weight_adequacy(p, minimum_effective_size=1.7, maximum_single_weight=0.7) == m


def test_parameter_moments_covariance_and_log_coordinate_are_preserved() -> None:
    p = particles(
        coords=("lp3.mean_loge_flow", "lp3.loge_sd_loge_flow"),
        draws=((1.0, 0.0), (3.0, 1.0)),
        log_weights=(log(3.0), log(1.0)),
    )
    m = posterior_moments(p)
    assert [v.mean for v in m.parameters] == near([1.5, 0.25])
    assert m.covariance[0] == near((0.75, 0.375))
    assert m.covariance[1] == near((0.375, 0.1875))
    assert m.correlation[0][1] == near(1.0)
    assert m.parameters[1].mean == 0.25  # NOT log(mean SD) or mean raw SD
    assert m.parameters[1].standard_deviation == near(sqrt(0.1875))


def test_weighted_empirical_inverse_including_endpoint_and_infinite_period() -> None:
    v, w = (10.0, 1.0, inf), (0.3, 0.6, 0.1)
    assert weighted_quantile(v, w, 0.0) == 1.0
    assert weighted_quantile(v, w, 0.5) == 1.0
    assert weighted_quantile(v, w, 0.95) == inf
    assert weighted_quantile(v, w, 1.0) == inf


def test_tail_resolution_avoids_subnormal_weight_squaring_and_endpoint_loss() -> None:
    assert weighted_quantile((1.0, 2.0), (1.0, 1e-300), 1.0) == 2.0
    assert tail_effective_sample_size((1e-300, 2e-300), (1.0, 2.0), 1.0, upper=True) == near(1.8)
    assert tail_effective_sample_size((0.1, 0.9), (1.0, 2.0), 3.0, upper=True) == 0.0


def test_nonlinearity_distinguishes_expected_parameters_quantile_average_median_and_predictive() -> None:
    p = particles(draws=((0.0,), (log(4.0),)), log_weights=(log(3.0), 0.0))
    out = quantile_outputs(p, 0.8, lambda x: ExponentialKernel(x[0]), interval_mass=0.9)
    c = -log1p(-0.8)
    assert out.at_expected_parameters == near(sqrt(2) * c)
    assert out.mean_of_quantiles == near(1.75 * c)
    assert out.median_of_quantiles == near(c)
    assert out.probability_limits == near((c, 4 * c))
    assert out.lower_limit_effective_size == near(1.0)
    assert out.upper_limit_effective_size == near(1.0)
    assert out.at_expected_parameters < out.predictive_quantile < out.mean_of_quantiles
    assert out.mean_log10_quantile == near(log10(c * (4**0.25)))
    assert out.standard_deviation_log10_quantile == near(sqrt(0.75 * 0.25) * log10(4))


def test_probability_at_flow_preserves_f_vs_aep_and_noncommutation() -> None:
    p = particles(draws=((0.0,), (log(4.0),)), log_weights=(log(3.0), 0.0))
    result = probability_outputs(p, 2.0, lambda x: ExponentialKernel(x[0]), interval_mass=0.9)
    s = 0.75 * exp(-2.0) + 0.25 * exp(-0.5)
    assert result.expected_non_exceedance == near(1.0 - s)
    assert result.expected_aep == near(s)
    assert result.return_period == near(1.0 / s)
    assert result.return_period_limits == near((exp(0.5), exp(2.0)))
    assert result.return_period != near(0.75 * exp(2.0) + 0.25 * exp(0.5))


def test_generalised_inverse_returns_zero_at_unattainable_probability_jump() -> None:
    p = particles(draws=((0.8,), (0.4,)), log_weights=(0.0, 0.0))
    assert predictive_quantile(p, 0.5, lambda x: AtomKernel(x[0])) == 0.0
    result = quantile_outputs(p, 0.5, lambda x: AtomKernel(x[0]), interval_mass=0.9)
    assert result.at_expected_parameters == 0.0
    assert result.predictive_quantile == 0.0
    assert result.nonpositive_quantile_weight == 0.5
    assert result.mean_log10_quantile is None
    assert result.standard_deviation_log10_quantile is None
    assert result.probability_limits[0] == 0.0


def test_infinite_component_return_period_is_explicit_not_legacy_sentinel() -> None:
    p = particles(draws=((0.0,), (log(4.0),)), log_weights=(0.0, 0.0))
    result = probability_outputs(p, 1e4, lambda x: ExponentialKernel(x[0]), interval_mass=0.9)
    assert result.return_period == inf
    assert result.return_period_limits == (inf, inf)
    assert result.zero_survival_weight == near(1.0)


@pytest.mark.parametrize(
    ("coordinates", "draws", "ll", "lp", "lg", "reason"),
    [
        (("x", "x"), ((1.0, 2.0),), (0.0,), (0.0,), (0.0,), "unique"),
        (("x",), (), (), (), (), "nonempty"),
        (("x",), ((1.0, 2.0),), (0.0,), (0.0,), (0.0,), "finite coordinates"),
        (("x",), ((1.0,),), (0.0, 0.0), (0.0,), (0.0,), "one entry"),
        (("x",), ((1.0,),), (float("nan"),), (0.0,), (0.0,), "target log"),
        (("x",), ((1.0,),), (-inf,), (0.0,), (0.0,), "all importance"),
        (("x",), ((1.0,),), (0.0,), (0.0,), (-inf,), "proposal log"),
    ],
)
def test_invalid_particle_inputs_fail_instead_of_reweighting(
    coordinates: tuple[str, ...],
    draws: tuple[tuple[float, ...], ...],
    ll: tuple[float, ...],
    lp: tuple[float, ...],
    lg: tuple[float, ...],
    reason: str,
) -> None:
    with pytest.raises(ValueError, match=reason):
        ImportanceParticles(
            coordinates=coordinates,
            draws=draws,
            log_likelihood=ll,
            log_prior=lp,
            log_proposal=lg,
            provenance=RunProvenance(
                compatibility_profile="test",
                method_id="test",
                method_version="1",
                package_version="1",
                evidence_status=EvidenceStatus.OBSERVED_ARTIFACT,
            ),
            proposal_description="test",
            random_stream_description="test",
        )


def test_invalid_levels_and_finite_quantile_fail_explicitly() -> None:
    p = particles(draws=((0.0,),), log_weights=(0.0,))
    for value in (float("nan"), -0.01, 0.0, 1.0, 1.01):
        with pytest.raises(ValueError, match="between zero and one"):
            quantile_outputs(p, value, lambda x: ExponentialKernel(x[0]), interval_mass=0.9)
    with pytest.raises(ValueError, match="between zero and one"):
        probability_outputs(p, 1.0, lambda x: ExponentialKernel(x[0]), interval_mass=1.0)
    with pytest.raises(ValueError, match="quantile probability"):
        weighted_quantile((1.0,), (1.0,), 1.1)


def test_legacy_report_f_return_period_identity_is_an_observation_not_oracle() -> None:
    # Reported Case A and later Case B values: printed precision, NOT paired native reruns.
    assert isclose(1.0 / (1.0 - 0.01599), 1.02, abs_tol=0.005)
    assert isclose(1.0 / (1.0 - 0.12580), 1.14, abs_tol=0.005)
    assert isclose(1.0 / (1.0 - 0.9761832), 41.98716, abs_tol=0.0002)
    assert 41.98716 != 50.0  # at EP q_50, expected probability T need not be 50


class PointMassKernel:
    """Synthetic atom; confirms mathematical generalized inverse, not Flike."""

    def __init__(self, location: float) -> None:
        self.location = location

    def cdf(self, discharge: float) -> float:
        return float(discharge >= self.location)

    def sf(self, discharge: float) -> float:
        return float(discharge < self.location)

    def logpdf(self, discharge: float) -> float:  # noqa: ARG002 - kernel protocol
        return -inf

    def logcdf(self, discharge: float) -> float:
        return 0.0 if discharge >= self.location else -inf

    def logsf(self, discharge: float) -> float:
        return 0.0 if discharge < self.location else -inf

    def ppf(self, non_exceedance_probability: float) -> float:  # noqa: ARG002 - kernel protocol
        return self.location

    def isf(self, survival_probability: float) -> float:  # noqa: ARG002 - kernel protocol
        return self.location


class BadCDFKernel(ExponentialKernel):
    def cdf(self, discharge: float) -> float:  # noqa: ARG002 - invalid CDF test double
        return 1.01


class BadSurvivalKernel(ExponentialKernel):
    def sf(self, discharge: float) -> float:  # noqa: ARG002 - invalid SF test double
        return float("nan")


class InfinitePPFKernel(ExponentialKernel):
    def ppf(self, non_exceedance_probability: float) -> float:  # noqa: ARG002 - invalid PPF test double
        return inf


def test_predictive_mixed_scale_inversion_and_adjacent_float_boundary() -> None:
    # A fixed 180 real-space bisection returned ~4.6e249 instead of 0.7033.
    p = particles(draws=((0.0,), (700.0,)), log_weights=(log(0.99), log(0.01)))
    quantile = predictive_quantile(p, 0.5, lambda x: ExponentialKernel(x[0]))
    expected = log(99.0 / 49.0)
    assert isclose(quantile, expected, rel_tol=2e-15)

    def mixture_cdf(discharge: float) -> float:
        return 0.99 * (-expm1(-discharge)) + 0.01 * (-expm1(-discharge / exp(700.0)))

    assert mixture_cdf(quantile) >= 0.5
    assert mixture_cdf(nextafter(quantile, -inf)) < 0.5


def test_predictive_component_ppf_rounds_below_requested_cdf() -> None:
    # The analytic PPF is valid but CDF(PPF(.25)) rounds to 0.24999999999999997.
    root = -log1p(-0.25)
    assert -expm1(-root) < 0.25
    expected = nextafter(root, inf)
    for draws, log_weights in (
        (((0.0,),), (0.0,)),
        (((0.0,), (0.0,)), (0.0, 0.0)),
    ):
        p = particles(draws=draws, log_weights=log_weights)
        result = predictive_quantile(p, 0.25, lambda x: ExponentialKernel(x[0]))
        assert result == expected
        assert -expm1(-result) >= 0.25
        assert -expm1(-nextafter(result, -inf)) < 0.25
        assert (
            quantile_outputs(p, 0.25, lambda x: ExponentialKernel(x[0]), interval_mass=0.9).predictive_quantile
            == expected
        )


def test_generalized_inverse_with_positive_and_negative_discrete_jumps() -> None:
    p = particles(draws=((-10.0,), (3.0,)), log_weights=(0.0, 0.0))
    assert predictive_quantile(p, 0.5, lambda x: PointMassKernel(x[0])) == -10.0
    assert predictive_quantile(p, 0.75, lambda x: PointMassKernel(x[0])) == 3.0


def test_unnormalized_extreme_weights_and_finite_density_ratio() -> None:
    assert weighted_quantile((1.0, 2.0), (1e308, 1e308), 0.5) == 1.0
    assert weighted_quantile((1.0, 2.0), (1e308, 1e308), 0.75) == 2.0
    p = ImportanceParticles(
        coordinates=("mu",),
        draws=((-1.0,), (1.0,)),
        log_likelihood=(-1e308, -1e308),
        log_prior=(-1e308, -1e308),
        log_proposal=(-1e308, -1e308),
        provenance=RunProvenance(
            compatibility_profile="synthetic-analytic",
            method_id="large-finite-log-ratio",
            method_version="test-only",
            package_version="0.0.0.dev0",
            evidence_status=EvidenceStatus.OBSERVED_ARTIFACT,
        ),
        proposal_description="fixed synthetic nodes",
        random_stream_description="none; deterministic",
    )
    assert p.weights == (0.5, 0.5)
    moments = posterior_moments(p)
    assert moments.covariance == ((1.0,),)
    assert moments.correlation == ((1.0,),)


@pytest.mark.parametrize(
    ("weights", "quantiles", "bound", "reason"),
    [
        ((1.0,), (), 0.0, "equally sized"),
        ((float("nan"),), (1.0,), 0.0, "nonnegative"),
        ((-0.1,), (1.0,), 0.0, "nonnegative"),
        ((1.0,), (1.0,), float("nan"), "negative infinity"),
        ((1.0,), (float("nan"),), 0.0, "negative infinity"),
    ],
)
def test_invalid_tail_resolution_inputs_are_not_silently_ignored(
    weights: tuple[float, ...], quantiles: tuple[float, ...], bound: float, reason: str
) -> None:
    with pytest.raises(ValueError, match=reason):
        tail_effective_sample_size(weights, quantiles, bound, upper=True)


def test_failed_probability_components_are_not_replaced_or_omitted() -> None:
    p = particles(draws=((0.0,),), log_weights=(0.0,))
    with pytest.raises(ValueError, match="invalid non-exceedance"):
        predictive_quantile(p, 0.5, lambda x: BadCDFKernel(x[0]))
    with pytest.raises(ValueError, match="outside"):
        probability_outputs(p, 1.0, lambda x: BadSurvivalKernel(x[0]), interval_mass=0.9)
    with pytest.raises(ValueError, match="non-finite"):
        quantile_outputs(p, 0.5, lambda x: InfinitePPFKernel(x[0]), interval_mass=0.9)


def _normal_conjugate_midpoint_nodes(count: int) -> ImportanceParticles:
    """TEST ONLY: deterministic midpoint quadrature, never iid proposal draws."""
    proposal = NormalDist(mu=0.0, sigma=2.0)
    nodes = tuple(proposal.inv_cdf((i + 0.5) / count) for i in range(count))
    observations = (1.0, 2.0, 3.0)
    log_sqrt_2pi = 0.5 * log(2.0 * pi)
    log_likelihood = tuple(
        -sum(0.5 * ((y - x) / 2.0) ** 2 for y in observations) - len(observations) * (log(2.0) + log_sqrt_2pi)
        for x in nodes
    )
    log_prior = tuple(-0.5 * x * x - log_sqrt_2pi for x in nodes)
    log_proposal = tuple(-0.5 * (x / 2.0) ** 2 - log(2.0) - log_sqrt_2pi for x in nodes)
    return ImportanceParticles(
        coordinates=("mu",),
        draws=tuple((x,) for x in nodes),
        log_likelihood=log_likelihood,
        log_prior=log_prior,
        log_proposal=log_proposal,
        provenance=RunProvenance(
            compatibility_profile="synthetic-analytic",
            method_id="conjugate-normal-midpoint-oracle",
            method_version="test-only",
            package_version="0.0.0.dev0",
            evidence_status=EvidenceStatus.OBSERVED_ARTIFACT,
        ),
        proposal_description="deterministic equal-probability midpoint quadrature under Normal(0, 2)",
        random_stream_description="none; equal-probability nodes, explicitly not iid samples",
    )


def test_conjugate_normal_benchmark_converges_by_deterministic_quadrature() -> None:
    # PR #22's likelihood-bayes-vectors.json supplies the independent benchmark.
    # prior N(0,1), y=[1,2,3], observation sigma=2:
    # variance=1/(1+3/4)=4/7, mean=(4/7)*(6/4)=6/7.
    target_mean, target_variance = 6.0 / 7.0, 4.0 / 7.0
    target_limits = (-0.6244506519246364, 2.338736366210351)
    mean_errors: list[float] = []
    variance_errors: list[float] = []
    interval_errors: list[float] = []
    for count in (256, 1024, 4096):
        draws = _normal_conjugate_midpoint_nodes(count)
        assert draws.log_likelihood != draws.log_prior
        assert draws.log_prior != draws.log_proposal
        assert all(weight > 0.0 for weight in draws.weights)
        moments = posterior_moments(draws)
        assert moments.parameters[0].coordinate == "mu"
        if count == 4096:
            assert isclose(moments.parameters[0].standard_deviation or 0.0, sqrt(target_variance), abs_tol=1e-12)
        values = tuple(row[0] for row in draws.draws)
        limits = (
            weighted_quantile(values, draws.weights, 0.025),
            weighted_quantile(values, draws.weights, 0.975),
        )
        mean_errors.append(abs(moments.parameters[0].mean - target_mean))
        variance_errors.append(abs(moments.covariance[0][0] - target_variance))
        interval_errors.append(max(abs(x - y) for x, y in zip(limits, target_limits, strict=True)))
    assert mean_errors[0] < 1e-8
    assert mean_errors[-1] < 1e-12
    assert variance_errors[0] < 1e-8
    assert variance_errors[-1] < 1e-12
    # Step limits resolve more slowly than moments; their error is not
    # required to decrease monotonically with each refinement.
    assert interval_errors[-1] < 0.0011
    assert interval_errors[-1] < interval_errors[0] / 10.0
