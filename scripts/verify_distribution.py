"""Fail closed if a wheel/sdist contains private evidence or lacks packaged guidance."""

import hashlib
import sys
import tarfile
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESOURCE_FILES: tuple[tuple[str, str], ...] = (
    ("docs/method-registry.json", "ffa_methods/resources/method-registry.json"),
    ("docs/references.md", "ffa_methods/resources/references.md"),
    ("docs/wp08-bayesian-cli.md", "ffa_methods/resources/wp08-bayesian-cli.md"),
    ("docs/compatibility-contract.md", "ffa_methods/resources/compatibility-contract.md"),
    ("docs/distribution-kernels.md", "ffa_methods/resources/distribution-kernels.md"),
    ("docs/methods/mgb-low-flow-screen.md", "ffa_methods/resources/methods/mgb-low-flow-screen.md"),
    ("docs/methods/legacy-zero-threshold.md", "ffa_methods/resources/methods/legacy-zero-threshold.md"),
    (
        "docs/methods/censored-historical-likelihood.md",
        "ffa_methods/resources/methods/censored-historical-likelihood.md",
    ),
    ("docs/methods/quantile-estimands.md", "ffa_methods/resources/methods/quantile-estimands.md"),
    ("docs/methods/importance-sampling-reference.md", "ffa_methods/resources/methods/importance-sampling-reference.md"),
    ("docs/methods/lh-moments-gev.md", "ffa_methods/resources/methods/lh-moments-gev.md"),
    ("docs/examples/fixed-gev-ams.csv", "ffa_methods/resources/examples/fixed-gev-ams.csv"),
    ("docs/examples/fixed-gev-settings.json", "ffa_methods/resources/examples/fixed-gev-settings.json"),
    ("docs/examples/ordinary-settings.json", "ffa_methods/resources/examples/ordinary-settings.json"),
    ("docs/examples/ordinary-ams.csv", "ffa_methods/resources/examples/ordinary-ams.csv"),
    ("docs/methods/l-moments.md", "ffa_methods/resources/methods/l-moments.md"),
    ("docs/examples/bayesian-fixed-ams.csv", "ffa_methods/resources/examples/bayesian-fixed-ams.csv"),
    ("docs/examples/bayesian-fixed-settings.json", "ffa_methods/resources/examples/bayesian-fixed-settings.json"),
    ("docs/examples/bayesian-censored-ams.csv", "ffa_methods/resources/examples/bayesian-censored-ams.csv"),
    ("docs/examples/bayesian-censored-settings.json", "ffa_methods/resources/examples/bayesian-censored-settings.json"),
)
BLOCKED_COMPONENTS = frozenset({"reference", "legacy-candidates", "private-evidence", "local-evidence"})
BLOCKED_SUFFIXES = (".fld", ".fid", ".7z")


def check_member_names(names: list[str]) -> None:
    """Reject retained evidence, archive traversal and source Flike project files."""
    for name in names:
        parts = name.replace("\\", "/").strip("/").split("/")
        if ".." in parts or any(part in BLOCKED_COMPONENTS for part in parts):
            msg = f"disallowed archive member: {name}"
            raise ValueError(msg)
        basename = parts[-1].casefold()
        if (
            basename.endswith(BLOCKED_SUFFIXES)
            or "_bayes_out" in basename
            or (basename.startswith("ams_") and basename.endswith(".csv"))
        ):
            msg = f"Flike input, unqualified report or original archive in distribution: {name}"
            raise ValueError(msg)


def _one_file(folder: Path, suffix: str) -> Path:
    """Identify exactly one expected artifact for reproducible verification."""
    candidates = sorted(p for p in folder.iterdir() if p.is_file() and p.name.endswith(suffix))
    if len(candidates) != 1:
        msg = f"expected one {suffix} in {folder}, got {len(candidates)}"
        raise ValueError(msg)
    return candidates[0]


def verify_distribution(dist_dir: Path) -> tuple[Path, Path]:
    """Inspect both archives read-only and check documentation is byte-for-byte current."""
    if not dist_dir.is_dir():
        msg = f"distribution directory does not exist: {dist_dir}"
        raise ValueError(msg)
    wheel = _one_file(dist_dir, ".whl")
    source_dist = _one_file(dist_dir, ".tar.gz")

    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        check_member_names(names)
        required = {"ffa_methods/py.typed", "ffa_methods/cli.py", "ffa_methods/contracts.py"}
        required.update(target for _, target in RESOURCE_FILES)
        missing = required.difference(names)
        if missing:
            msg = f"wheel missing mandatory content: {sorted(missing)}"
            raise ValueError(msg)
        for source, target in RESOURCE_FILES:
            if archive.read(target) != (REPO_ROOT / source).read_bytes():
                msg = f"packaged guidance differs from maintained source: {source}"
                raise ValueError(msg)

    with tarfile.open(source_dist, mode="r:gz") as archive:
        names = archive.getnames()
        check_member_names(names)
        for source, _ in RESOURCE_FILES:
            if not any(name.endswith("/" + source) for name in names):
                msg = f"sdist missing required guidance: {source}"
                raise ValueError(msg)
        if not any(name.endswith("/src/ffa_methods/py.typed") for name in names):
            msg = "sdist missing typed package marker"
            raise ValueError(msg)

    for artifact in (wheel, source_dist):
        checksum = hashlib.sha256(artifact.read_bytes()).hexdigest()
        print(f"{artifact.name}: sha256={checksum}")
    return wheel, source_dist


def main(args: list[str] | None = None) -> int:
    """Report verification failures through a nonzero process exit code."""
    argv = sys.argv[1:] if args is None else args
    if len(argv) != 1:
        msg = "usage: python scripts/verify_distribution.py DIST_DIRECTORY"
        raise SystemExit(msg)
    verify_distribution(Path(argv[0]).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
