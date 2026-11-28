"""Install and exercise the wheel from a clean venv outside the checkout."""

import json
import os
import subprocess
import sys
import tempfile
from hashlib import sha256
from math import exp, expm1, isclose, log, log1p, pi, sqrt
from pathlib import Path
from statistics import NormalDist


def _invoke(
    command: list[str], *, cwd: Path, env: dict[str, str], expected_code: int
) -> subprocess.CompletedProcess[str]:
    """Run a fixed executable/argument vector without a shell."""
    result = subprocess.run(command, check=False, text=True, capture_output=True, cwd=cwd, env=env)
    if result.returncode != expected_code:
        msg = (
            f"command failed: {command!r}; status={result.returncode}, "
            f"expected={expected_code}; stdout={result.stdout!r}; stderr={result.stderr!r}"
        )
        raise RuntimeError(msg)
    return result


def _check_installed_method_pages(console: Path, output: Path, env: dict[str, str]) -> None:
    """Require all declared offline method pages in the installed wheel."""
    for method_id in (
        "mgb-low-flow-screen",
        "legacy-zero-threshold",
        "censored-historical-likelihood",
        "quantile-estimands",
        "importance-sampling-reference",
        "lh-moments-gev",
        "l-moments",
    ):
        page = _invoke(
            [str(console), "explain", method_id, "--format", "markdown"],
            cwd=output,
            env=env,
            expected_code=0,
        )
        if not page.stdout.startswith("# ") or f"Method ID:** `{method_id}`" not in page.stdout:
            msg = f"installed wheel has missing or incorrect offline technical page: {method_id}"
            raise RuntimeError(msg)
    _invoke([str(console), "explain", "--format", "markdown"], cwd=output, env=env, expected_code=2)
    _invoke([str(console), "explain", "lp3", "--format", "markdown"], cwd=output, env=env, expected_code=2)


def _ordinary_reference(
    estimator: str, extra: dict[str, object], probabilities: tuple[float, ...]
) -> tuple[dict[str, float], tuple[float, ...]]:
    """Independent identities for the three-point [1, 2, 4] synthetic example."""
    first, second = 7.0 / 3.0, 1.0
    if estimator == "gumbel_lmoments":
        scale = second / log(2.0)
        location = first - 0.5772156649015329 * scale
        return (
            {"gumbel.location": location, "gumbel.scale": scale},
            tuple(location - scale * log(-log(p)) for p in probabilities),
        )
    if estimator == "lognormal_lmoments":
        sd = sqrt(2.0) * NormalDist().inv_cdf((1.0 + second / first) / 2.0)
        mean = log(first / 2.0) - sd**2 / 2.0
        return (
            {"lognormal.mean_loge_flow": mean, "lognormal.loge_sd_loge_flow": log(sd)},
            tuple(2.0 * exp(mean + sd * NormalDist().inv_cdf(p)) for p in probabilities),
        )
    if estimator == "lp3_lmoments":
        sd = sqrt(pi) * 2.0 * log(2.0) / 3.0
        return (
            {"lp3.mean_loge_flow": 0.0, "lp3.loge_sd_loge_flow": log(sd), "lp3.skew_loge_flow": 0.0},
            tuple(2.0 * exp(sd * NormalDist().inv_cdf(p)) for p in probabilities),
        )
    if extra["gp_location_mode"] == "fixed":
        location, scale, kappa = 0.25, 325.0 / 144.0, 1.0 / 12.0
        return (
            {"gp.location": location, "gp.scale": scale, "gp.shape_xi": -kappa},
            tuple(location - scale * expm1(kappa * log1p(-p)) / kappa for p in probabilities),
        )
    location, scale = 1.0 / 3.0, 2.0
    return (
        {"gp.location": location, "gp.scale": scale, "gp.shape_xi": 0.0},
        tuple(location - scale * log1p(-p) for p in probabilities),
    )


def _check_ordinary_result(
    destination: Path,
    settings_bytes: bytes,
    ams_bytes: bytes,
    expected: dict[str, float],
    quantiles: tuple[float, ...],
    probabilities: tuple[float, ...],
) -> None:
    """Check the installed report against independent values and byte custody."""
    doc = json.loads((destination / "results.json").read_text(encoding="utf-8"))
    provenance = doc["provenance"]
    if (
        provenance["method_id"] != "l-moments"
        or provenance["input_sha256"] != sha256(ams_bytes).hexdigest()
        or provenance["settings_sha256"] != sha256(settings_bytes).hexdigest()
        or (destination / "source-ams.csv").read_bytes() != ams_bytes
        or (destination / "source-settings.json").read_bytes() != settings_bytes
    ):
        msg = "ordinary installed example source identity changed"
        raise RuntimeError(msg)
    if doc["method_qualification"] != "not_established_by_formatter":
        msg = "ordinary result incorrectly qualified"
        raise RuntimeError(msg)
    parameters = doc["parameters"]
    if set(expected) != {parameter["coordinate"] for parameter in parameters}:
        msg = "ordinary installed example parameter coordinates differ"
        raise RuntimeError(msg)
    for parameter in parameters:
        if (
            parameter["summary_kind"] != "point_estimate"
            or parameter["mean"] is not None
            or parameter["standard_deviation"] is not None
            or not isclose(parameter["point_estimate"], expected[parameter["coordinate"]], rel_tol=2e-11, abs_tol=2e-12)
        ):
            msg = f"ordinary installed example parameter mismatch: {parameter}"
            raise RuntimeError(msg)
    for point, quantile, probability in zip(doc["results"], quantiles, probabilities, strict=True):
        if (
            point["meaning"] != "deterministic_fitted_quantile"
            or point["unit"] != "m3/s"
            or point["input_non_exceedance_probability"] != probability
            or not isclose(point["input_annual_exceedance_probability"], 1.0 - probability, abs_tol=1e-15)
            or not isclose(point["value"], quantile, rel_tol=2e-11, abs_tol=2e-12)
        ):
            msg = f"ordinary installed example quantile mismatch: {point}"
            raise RuntimeError(msg)


def _check_ordinary_installed(console: Path, output: Path, env: dict[str, str]) -> None:
    """Execute every documented configuration through the actual installed console."""
    config = json.loads((output / "ordinary-settings.json").read_text(encoding="utf-8"))
    ams = output / "ordinary-ams.csv"
    ams_bytes = ams.read_bytes()
    probabilities = (0.5, 0.9)
    cases: tuple[tuple[str, str, dict[str, object]], ...] = (
        ("gumbel", "gumbel_lmoments", {}),
        ("ln2", "lognormal_lmoments", {"reference_discharge": 2}),
        ("lp3", "lp3_lmoments", {"reference_discharge": 2}),
        ("gp-fixed", "gp_lmoments", {"gp_location_mode": "fixed", "gp_location": 0.25}),
        ("gp-free", "gp_lmoments", {"gp_location_mode": "free"}),
    )
    for name, estimator, extra in cases:
        settings_path = output / f"{name}-settings.json"
        settings_bytes = json.dumps({**config, "estimator": estimator, **extra}).encode("utf-8")
        settings_path.write_bytes(settings_bytes)
        destination = output / f"{name}-report"
        _invoke(
            [str(console), "fit", "--ams", str(ams), "--settings", str(settings_path), "--output", str(destination)],
            cwd=output,
            env=env,
            expected_code=0,
        )
        expected, quantiles = _ordinary_reference(estimator, extra, probabilities)
        _check_ordinary_result(destination, settings_bytes, ams_bytes, expected, quantiles, probabilities)


def _check_gev_support_report(console: Path, output: Path, env: dict[str, str]) -> None:
    """Exercise a support-violating fit using the installed entry point."""
    values = (1, 100, 101, 102, 103, 104, 105, 106, 107, 108)
    (output / "support-ams.csv").write_text(
        "Year,AMS\n" + "".join(f"{2000 + i},{q}\n" for i, q in enumerate(values)), encoding="utf-8"
    )
    destination = output / "support-report"
    _invoke(
        [
            str(console),
            "fit",
            "--ams",
            "support-ams.csv",
            "--settings",
            "fixed-gev-settings.json",
            "--output",
            str(destination),
        ],
        cwd=output,
        env=env,
        expected_code=0,
    )
    doc = json.loads((destination / "results.json").read_text(encoding="utf-8"))
    support = doc["gev_fitted_support"]
    if (
        support["kind"] != "finite_upper"
        or support["outside"] != [{"year": 2008, "discharge": 107}, {"year": 2009, "discharge": 108}]
        or support["at_endpoint_count"] != 0
        or "gev_fitted_support_json" not in (destination / "results.csv").read_text(encoding="utf-8")
        or "outside: year=2009," not in (destination / "report.txt").read_text(encoding="utf-8")
    ):
        msg = "installed GEV support reporting did not retain the synthetic outside observations"
        raise RuntimeError(msg)


def _check_bayesian_installed(console: Path, output: Path, env: dict[str, str]) -> None:
    """Check real console execution from admitted synthetic bytes outside checkout."""
    for kind in ("fixed", "censored"):
        ams = output / f"bayesian-{kind}-ams.csv"
        settings = output / f"bayesian-{kind}-settings.json"
        destination = output / f"bayesian-{kind}-report"
        args = [str(console), "fit", "--ams", str(ams), "--settings", str(settings), "--output", str(destination)]
        _invoke(args, cwd=output, env=env, expected_code=0)
        doc = json.loads((destination / "results.json").read_text(encoding="utf-8"))
        if (
            doc["method_qualification"] != "not_established_by_formatter"
            or doc["provenance"]["input_sha256"] != sha256(ams.read_bytes()).hexdigest()
            or doc["provenance"]["settings_sha256"] != sha256(settings.read_bytes()).hexdigest()
            or (destination / "source-ams.csv").read_bytes() != ams.read_bytes()
            or (destination / "source-settings.json").read_bytes() != settings.read_bytes()
        ):
            msg = "installed Bayesian workflow lost source custody or qualification status"
            raise RuntimeError(msg)
        rows = {point["meaning"]: point for point in doc["results"]}
        aep = rows["expected_annual_exceedance_probability"]
        if kind == "fixed":
            reference = 1 - NormalDist(1.5, sqrt(1.25)).cdf(2.0)
            z = NormalDist().inv_cdf(0.9)
            if (
                not isclose(doc["parameters"][0]["mean"], 1.5, abs_tol=0.16)
                or not isclose(aep["value"], reference, abs_tol=0.09)
                or not isclose(rows["posterior_predictive_quantile"]["value"], exp(1.5 + sqrt(1.25) * z), abs_tol=9)
                or not isclose(rows["posterior_mean_quantile"]["value"], exp(1.5 + z + 0.125), abs_tol=6)
                or rows["posterior_mean_quantile"]["value"] == rows["quantile_at_expected_parameters"]["value"]
                or aep["precision_state"] != "conditional_pass"
                or doc["precision_audit"]["completed_count"] != 3
            ):
                msg = "installed fixed-scale Bayesian fit failed conjugate reference or selected-output audit"
                raise RuntimeError(msg)
        elif (
            not isclose(doc["parameters"][0]["mean"], 0.003598304886871672, abs_tol=0.032)
            or not isclose(doc["parameters"][1]["mean"], -0.01338564667751898, abs_tol=0.012)
            or not isclose(aep["value"], 0.3686203803796388, abs_tol=0.012)
            or "posterior_mean_quantile" in rows
            or aep["precision_state"] != "not_assessed"
            or not any("UNDEFINED" in warning and "diverges" in warning for warning in doc["warnings"])
        ):
            msg = "installed estimated-scale censored Bayesian fit failed independent quadrature reference"
            raise RuntimeError(msg)
        replay = output / f"bayesian-{kind}-replay"
        _invoke([*args[:-1], str(replay)], cwd=output, env=env, expected_code=0)
        if any(item.read_bytes() != (replay / item.name).read_bytes() for item in destination.iterdir()):
            msg = "installed Bayesian same-environment seeded replay differs"
            raise RuntimeError(msg)
        _invoke(args, cwd=output, env=env, expected_code=2)
    bad_ams = output / "bayesian-invalid-ams.csv"
    bad_ams.write_bytes(b"Year,AMS\n2000,0\n")
    invalid_destination = output / "bayesian-invalid-report"
    _invoke(
        [
            str(console),
            "fit",
            "--ams",
            str(bad_ams),
            "--settings",
            str(output / "bayesian-fixed-settings.json"),
            "--output",
            str(invalid_destination),
        ],
        cwd=output,
        env=env,
        expected_code=2,
    )
    if invalid_destination.exists():
        msg = "invalid installed Bayesian input created an output bundle"
        raise RuntimeError(msg)


def _check_selected_h_installed(console: Path, output: Path, env: dict[str, str], settings: dict[str, object]) -> None:
    ams = output / "selected-h-ams.csv"
    ams.write_text(
        "Year,AMS\n" + "\n".join(f"{1950 + i},{20 - log(-log((i - 0.35) / 50.3))}" for i in range(1, 51)),
        encoding="utf-8",
    )
    settings.pop("lh_shift")
    settings.update(
        schema_version="ffa-gev-selected-h/1",
        estimator="gev_lh_selected_h",
        critical_z=1.96,
        calibration_seed=7,
        non_exceedance_probabilities=[0.9],
        bootstrap={
            "replications": 24,
            "seed": 42,
            "confidence": 0.5,
            "maximum_failure_fraction": 0.25,
            "endpoint_tolerance": 2.0,
        },
    )
    config = output / "selected-h-settings.json"
    config.write_text(json.dumps(settings), encoding="utf-8")
    target = output / "selected-h-report"
    _invoke(
        [str(console), "fit", "--ams", str(ams), "--settings", str(config), "--output", str(target)],
        cwd=output,
        env=env,
        expected_code=0,
    )
    report = json.loads((target / "results.json").read_text(encoding="utf-8"))
    if (
        report["settings"]["lh_shift"] != 0
        or set(report["candidate_gev_support"]) != set("01234")
        or [point["probability_limit_side"] for point in report["results"]] != [None, "lower", "upper"]
    ):
        msg = "installed selected-H CLI did not report five fitted supports, H=0 and bootstrap limits"
        raise RuntimeError(msg)


def smoke_installed(dist_dir: Path) -> None:
    """Check version/help/rejection, import origin and offline packaged resources."""
    wheels = sorted(dist_dir.glob("ffa_toolkit-*.whl"))
    if len(wheels) != 1:
        msg = f"expected exactly one project wheel in {dist_dir}"
        raise ValueError(msg)
    wheel = wheels[0].resolve()
    repo_root = Path(__file__).resolve().parents[1]

    with tempfile.TemporaryDirectory(prefix="ffa-installed-") as folder:
        work = Path(folder)
        environment = work / "venv"
        output = work / "outside-checkout"
        output.mkdir()
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        env["PYTHONNOUSERSITE"] = "1"
        _invoke([sys.executable, "-m", "venv", str(environment)], cwd=output, env=env, expected_code=0)
        interpreter = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        console = environment / ("Scripts/ffa.exe" if os.name == "nt" else "bin/ffa")
        _invoke(
            [str(interpreter), "-m", "pip", "install", str(wheel)],
            cwd=output,
            env=env,
            expected_code=0,
        )

        probe = """
import json
import sys
from importlib.metadata import version
from importlib.resources import files
from pathlib import Path
import ffa_methods
site_path = Path(ffa_methods.__file__).resolve()
assert site_path.is_relative_to(Path(sys.argv[1]).resolve()), site_path
assert not site_path.is_relative_to(Path(sys.argv[2]).resolve()), site_path
assert version("ffa-toolkit") == ffa_methods.__version__
root = files("ffa_methods")
registry = json.loads(root.joinpath("resources/method-registry.json").read_text(encoding="utf-8"))
assert registry["compatibility_profile"] == "tuflow-bmt-flike-5.0.306.0-file3.10"
assert "LEGACY_SAMPLES" in root.joinpath("resources/references.md").read_text(encoding="utf-8")
assert "TUFLOW/BMT" in root.joinpath("resources/compatibility-contract.md").read_text(encoding="utf-8")
print(f"wheel import: {site_path}")
"""
        _invoke(
            [str(interpreter), "-c", probe, str(environment), str(repo_root)],
            cwd=output,
            env=env,
            expected_code=0,
        )
        if not console.is_file():
            msg = f"wheel did not install the expected console entry point: {console}"
            raise RuntimeError(msg)
        version_result = _invoke(
            [str(interpreter), "-m", "ffa_methods", "--version"],
            cwd=output,
            env=env,
            expected_code=0,
        )
        metadata = _invoke(
            [str(interpreter), "-c", "from importlib.metadata import version; print(version('ffa-toolkit'))"],
            cwd=output,
            env=env,
            expected_code=0,
        ).stdout.strip()
        if f"ffa {metadata}" not in version_result.stdout:
            msg = f"CLI version does not match installed distribution: {version_result.stdout!r}"
            raise RuntimeError(msg)
        help_result = _invoke(
            [str(interpreter), "-m", "ffa_methods", "--help"],
            cwd=output,
            env=env,
            expected_code=0,
        )
        if "--version" not in help_result.stdout:
            msg = "installed CLI help does not describe version option"
            raise RuntimeError(msg)
        console_version = _invoke([str(console), "--version"], cwd=output, env=env, expected_code=0)
        console_help = _invoke([str(console), "--help"], cwd=output, env=env, expected_code=0)
        if console_version.stdout != version_result.stdout or console_help.stdout != help_result.stdout:
            msg = "installed ffa console entry point differs from python -m ffa_methods"
            raise RuntimeError(msg)
        explain_result = _invoke(
            [str(console), "explain", "quantile-estimands", "--format", "json"],
            cwd=output,
            env=env,
            expected_code=0,
        )
        guidance = json.loads(explain_result.stdout)
        if (
            guidance["schema_version"] != "ffa-method-guidance/1"
            or len(guidance["methods"]) != 1
            or guidance["methods"][0]["method_id"] != "quantile-estimands"
        ):
            msg = "installed guidance output does not match bundled method register"
            raise RuntimeError(msg)
        _check_installed_method_pages(console, output, env)
        copy_examples = """
from importlib.resources import files
from pathlib import Path
root = files("ffa_methods").joinpath("resources", "examples")
for name in ("fixed-gev-ams.csv", "fixed-gev-settings.json", "ordinary-ams.csv", "ordinary-settings.json",
             "bayesian-fixed-ams.csv", "bayesian-fixed-settings.json",
             "bayesian-censored-ams.csv", "bayesian-censored-settings.json"):
    Path(name).write_bytes(root.joinpath(name).read_bytes())
"""
        _invoke([str(interpreter), "-c", copy_examples], cwd=output, env=env, expected_code=0)
        destination = output / "fixed-gev-report"
        fit_args = [
            str(console),
            "fit",
            "--ams",
            "fixed-gev-ams.csv",
            "--settings",
            "fixed-gev-settings.json",
            "--output",
            str(destination),
        ]
        _invoke(fit_args, cwd=output, env=env, expected_code=0)
        published = json.loads((destination / "results.json").read_text(encoding="utf-8"))
        if (
            published["method_qualification"] != "not_established_by_formatter"
            or len(published["results"]) != 4
            or published["results"][0]["meaning"] != "deterministic_fitted_quantile"
            or published["parameters"][0]["summary_kind"] != "point_estimate"
            or not published["provenance"]["input_sha256"]
            or not published["provenance"]["settings_sha256"]
        ):
            msg = "installed wheel fixed-H mathematical fit has invalid quantities or provenance"
            raise RuntimeError(msg)
        _invoke(fit_args, cwd=output, env=env, expected_code=2)
        _check_ordinary_installed(console, output, env)
        _check_gev_support_report(console, output, env)
        _check_bayesian_installed(console, output, env)
        _check_selected_h_installed(
            console, output, env, json.loads((output / "fixed-gev-settings.json").read_text(encoding="utf-8"))
        )
        reference = _invoke([str(console), "references", "ARR3"], cwd=output, env=env, expected_code=0)
        if "### ARR3" not in reference.stdout or "2.6.3.6" not in reference.stdout:
            msg = "installed offline reference register lacks the pinned ARR citation"
            raise RuntimeError(msg)
        _invoke([str(console), "explain", "no-such-method"], cwd=output, env=env, expected_code=2)
        _invoke([str(console), "references", "no-such-source"], cwd=output, env=env, expected_code=2)
        for arguments in ([], ["fit"], ["--unknown-option"]):
            _invoke([str(interpreter), "-m", "ffa_methods", *arguments], cwd=output, env=env, expected_code=2)
            _invoke([str(console), *arguments], cwd=output, env=env, expected_code=2)
        print(f"installed wheel and console entry point passed outside checkout: {wheel.name}")


def main(args: list[str] | None = None) -> int:
    """Accept the build output directory as the sole argument."""
    argv = sys.argv[1:] if args is None else args
    if len(argv) != 1:
        msg = "usage: python scripts/smoke_installed.py DIST_DIRECTORY"
        raise SystemExit(msg)
    smoke_installed(Path(argv[0]).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
