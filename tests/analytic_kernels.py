"""Analytic exponential-law test oracle, not Flike evidence."""

from math import exp, expm1, log, log1p


class ExponentialKernel:
    """Unit exponential: independent closed-form likelihood oracle."""

    def logpdf(self, discharge: float) -> float:
        return -discharge if discharge >= 0 else float("-inf")

    def logcdf(self, discharge: float) -> float:
        return log(-expm1(-discharge)) if discharge > 0 else float("-inf")

    def logsf(self, discharge: float) -> float:
        return -discharge if discharge >= 0 else 0.0

    def cdf(self, discharge: float) -> float:
        return -expm1(-discharge) if discharge > 0 else 0.0

    def sf(self, discharge: float) -> float:
        return exp(-discharge) if discharge >= 0 else 1.0

    def ppf(self, non_exceedance_probability: float) -> float:
        return -log1p(-non_exceedance_probability)

    def isf(self, survival_probability: float) -> float:
        return -log(survival_probability)
