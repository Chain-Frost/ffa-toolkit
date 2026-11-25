"""Exercise CI skip decisions against real Git patches and packaging consumers."""

import shutil
import subprocess
from pathlib import Path

import pytest

from scripts import ci_changes

ROOT = Path(__file__).resolve().parents[1]


def _git(root: Path, *args: str) -> str:
    executable = shutil.which("git")
    assert executable is not None
    command = [
        executable,
        "-c",
        "user.name=CI scope test",
        "-c",
        "user.email=ci-scope@example.invalid",
        "-c",
        "commit.gpgsign=false",
        "-c",
        "core.autocrlf=false",
        *args,
    ]
    # Fixed Git executable and argument vector; no shell or downloaded code.
    return subprocess.run(command, cwd=root, check=True, capture_output=True, encoding="utf-8").stdout.strip()  # noqa: S603


def _write(root: Path, name: str, text: str) -> None:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def _commit(root: Path) -> str:
    _git(root, "add", "--all")
    _git(root, "commit", "--quiet", "-m", "CI scope fixture")
    return _git(root, "rev-parse", "HEAD")


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    """Use the maintained package manifest rather than duplicate classifier configuration."""
    root = tmp_path / "repository"
    root.mkdir()
    _git(root, "init", "--quiet", "--initial-branch=main")
    _write(root, "pyproject.toml", (ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    _write(root, "README.md", "# Package\n")
    _write(root, "docs/development.md", "# Development\n")
    _write(root, "docs/work/receipt.md", "# Research receipt\n")
    _write(root, "src/ffa_methods/example.py", "VALUE = 1\n")
    _commit(root)
    return root


def test_documentation_shortcut_and_manual_full_validation(repository: Path) -> None:
    _write(repository, "docs/development.md", "# Updated development\n\n[Package](../README.md)\n")
    _commit(repository)

    assert ci_changes.check_changes(repository, "pull_request") is False
    assert ci_changes.check_changes(repository, "workflow_dispatch") is True


def test_shallow_pr_merge_checks_earlier_code_change(repository: Path, tmp_path: Path) -> None:
    """The last PR commit is prose, but the whole merge patch contains numerical code."""
    _git(repository, "checkout", "--quiet", "-b", "feature")
    _write(repository, "src/ffa_methods/example.py", "VALUE = 2\n")
    _commit(repository)
    _write(repository, "docs/development.md", "# Final PR prose\n")
    _commit(repository)
    _git(repository, "checkout", "--quiet", "main")
    _write(repository, "README.md", "# Base advanced independently\n")
    _commit(repository)
    _git(repository, "merge", "--quiet", "--no-ff", "--no-edit", "feature")

    checkout = tmp_path / "shallow-checkout"
    _git(tmp_path, "clone", "--quiet", "--no-local", "--depth=2", repository.as_uri(), str(checkout))
    assert _git(checkout, "rev-parse", "--is-shallow-repository") == "true"
    assert ci_changes.check_changes(checkout, "pull_request") is True


def test_rename_into_documentation_cannot_hide_deleted_code(repository: Path) -> None:
    (repository / "src/ffa_methods/example.py").rename(repository / "docs/work/moved-code.md")
    _commit(repository)

    assert ci_changes.check_changes(repository, "pull_request") is True


def test_main_whitespace_check_covers_all_pushed_commits(repository: Path) -> None:
    before = _git(repository, "rev-parse", "HEAD")
    _write(repository, "src/ffa_methods/example.py", "VALUE = 2   \n")
    _commit(repository)
    _write(repository, "docs/development.md", "# Clean final commit\n")
    _commit(repository)
    assert _git(repository, "diff", "--check", "HEAD^1", "HEAD") == ""

    with pytest.raises(subprocess.CalledProcessError):
        ci_changes.check_changes(repository, "push", before)


@pytest.mark.parametrize(
    "path",
    [
        "docs/methods/lh-moments-gev.md",
        "docs/method-registry.json",
        "docs/references.md",
        "docs/examples/input.csv",
        "reference/fixture.json",
        "qualification/evidence.json",
        ".github/workflows/python-quality.yml",
        "docs/unreviewed-document.md",
    ],
)
def test_protected_or_unreviewed_input_runs_full_suite(repository: Path, path: str) -> None:
    _write(repository, path, "Changed input\n")
    _commit(repository)

    assert ci_changes.check_changes(repository, "pull_request") is True


def test_mixed_code_and_administrative_prose_runs_full_suite(repository: Path) -> None:
    _write(repository, "src/ffa_methods/example.py", "VALUE = 2\n")
    _write(repository, "docs/development.md", "# Updated instructions\n")
    _commit(repository)

    assert ci_changes.check_changes(repository, "pull_request") is True


@pytest.mark.parametrize(
    ("metadata_input", "document"),
    [
        ("readme", "docs/development.md"),
        ("readme", "./docs/development.md"),
        ("readme", "docs//development.md"),
        ("license-files", "docs/development.md"),
    ],
)
def test_metadata_document_cannot_use_shortcut_after_admission(
    repository: Path, metadata_input: str, document: str
) -> None:
    manifest = repository / "pyproject.toml"
    text = manifest.read_text(encoding="utf-8")
    if metadata_input == "readme":
        text = text.replace('readme = "README.md"', f'readme = "{document}"')
    else:
        text = text.replace('license-files = ["LICENSE"]', f'license-files = ["LICENSE", "{document}"]')
    manifest.write_text(text, encoding="utf-8", newline="\n")
    _commit(repository)
    _write(repository, "docs/development.md", "# Updated installed metadata\n")
    _commit(repository)

    assert ci_changes.check_changes(repository, "pull_request") is True


@pytest.mark.parametrize("pattern", ["/docs/**", "/docs/**/*.md"])
def test_broader_package_glob_vetoes_previously_safe_document(repository: Path, pattern: str) -> None:
    manifest = repository / "pyproject.toml"
    text = manifest.read_text(encoding="utf-8").replace('    "/docs/compatibility-contract.md",', f'    "{pattern}",')
    manifest.write_text(text, encoding="utf-8", newline="\n")
    _commit(repository)
    _write(repository, "docs/development.md", "# Now packaged by a broad glob\n")
    _commit(repository)

    assert ci_changes.check_changes(repository, "pull_request") is True


@pytest.mark.parametrize("change", ["custom-hook", "default-sdist", "default-wheel"])
def test_uncertain_packaging_layout_runs_full_suite(repository: Path, change: str) -> None:
    manifest = repository / "pyproject.toml"
    text = manifest.read_text(encoding="utf-8")
    if change == "custom-hook":
        text += '\n[tool.hatch.build.hooks.custom]\npath = "hooks.py"\n'
    elif change == "default-wheel":
        text = text.replace('packages = ["src/ffa_methods"]', "packages = []")
    else:
        start = text.index("include = [", text.index("[tool.hatch.build.targets.sdist]"))
        end = text.index("]", start)
        text = text[:start] + "include = []" + text[end + 1 :]
    manifest.write_text(text, encoding="utf-8", newline="\n")
    _commit(repository)
    _write(repository, "docs/development.md", "# Hook inputs need review\n")
    _commit(repository)

    assert ci_changes.check_changes(repository, "pull_request") is True


def test_first_party_consumer_vetoes_documentation_shortcut(repository: Path) -> None:
    _write(
        repository,
        "tests/test_resource.py",
        'from pathlib import Path\n\nINPUT = Path("docs") / "development.md"\n',
    )
    _commit(repository)
    _write(repository, "docs/development.md", "# A test now consumes these bytes\n")
    _commit(repository)

    assert ci_changes.check_changes(repository, "pull_request") is True


def test_unreadable_consumer_cannot_authorize_shortcut(repository: Path) -> None:
    (repository / "src/ffa_methods/input.bin").write_bytes(b"\xff\xfe")
    _commit(repository)
    _write(repository, "docs/development.md", "# Consumer scan is uncertain\n")
    _commit(repository)

    assert ci_changes.check_changes(repository, "pull_request") is True


def test_push_without_usable_before_sha_fails_closed(repository: Path) -> None:
    for before in ("", "not-a-commit"):
        with pytest.raises(ValueError, match="before SHA"):
            ci_changes.check_changes(repository, "push", before)


def test_failed_documentation_validation_never_emits_skip_output(
    repository: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write(repository, "docs/development.md", "# Updated prose\n\n[Missing](missing.md)\n")
    _commit(repository)
    output = tmp_path / "github-output"
    monkeypatch.setattr(ci_changes, "REPO_ROOT", repository)
    monkeypatch.setenv("CI_EVENT_NAME", "pull_request")
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))

    with pytest.raises(ValueError, match="Markdown target"):
        ci_changes.main()
    assert not output.exists()

    _write(repository, "docs/development.md", "# Fixed prose\n\n[Package](../README.md)\n")
    _commit(repository)
    assert ci_changes.main() == 0
    assert output.read_text(encoding="utf-8") == "full=false\n"


def test_document_deletion_checks_inbound_links(repository: Path) -> None:
    _write(repository, "docs/development.md", "# Instructions\n\n[Receipt](work/receipt.md)\n")
    _commit(repository)
    (repository / "docs/work/receipt.md").unlink()
    _commit(repository)

    with pytest.raises(ValueError, match="Markdown target"):
        ci_changes.check_changes(repository, "pull_request")


def test_initial_root_commit_gets_full_push_and_manual_validation(repository: Path) -> None:
    assert ci_changes.check_changes(repository, "push", "0" * 40) is True
    assert ci_changes.check_changes(repository, "workflow_dispatch") is True
    _write(repository, "other.txt", "another commit\n")
    _commit(repository)
    with pytest.raises(ValueError, match="initial root commit"):
        ci_changes.check_changes(repository, "push", "0" * 40)
