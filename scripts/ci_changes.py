"""Check the whole incoming patch and admit only reviewed documentation to cheap PR CI."""

import os
import re
import shutil
import subprocess
import sys
import tomllib
from itertools import chain
from pathlib import Path
from typing import cast
from urllib.parse import unquote, urlsplit

REPO_ROOT = Path(__file__).resolve().parents[1]
# Keep this list narrow. Remove a document when it becomes a package/test input.
DOCUMENTATION_ONLY = frozenset(
    {
        "AGENTS.md",
        "docs/agent-work-scopes.md",
        "docs/development-stages.md",
        "docs/development.md",
        "docs/external-reference-audit.md",
        "docs/flike-feature-audit.md",
        "docs/implementation-plan.md",
        "docs/primary-literature-acquisition.md",
        "docs/repository-setup.md",
        "docs/sample-data-audit.md",
    }
)
_INLINE_LINK = re.compile(r"\[[^\]\n]+\]\(<?([^\s)>]+)>?(?:\s+\"[^\"]*\")?\)")
_REFERENCE_LINK = re.compile(r"^\[[^\]\n]+\]:\s*<?([^\s>]+)>?", re.MULTILINE)


def _git(root: Path, *args: str) -> str:
    executable = shutil.which("git")
    if executable is None:
        msg = "git is required for CI patch validation"
        raise RuntimeError(msg)
    result = subprocess.run([executable, *args], cwd=root, check=False, capture_output=True, encoding="utf-8", input="")
    if result.returncode:
        print(result.stdout, end="")
        print(result.stderr, file=sys.stderr, end="")
    result.check_returncode()
    return result.stdout


def _table(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        msg = "unrecognised packaging table"
        raise TypeError(msg)
    return cast("dict[str, object]", value)


def _strings(value: object) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in cast("list[object]", value)):
        msg = "unrecognised packaging input list"
        raise TypeError(msg)
    return cast("list[str]", value)


def _package_inputs(root: Path) -> list[str]:
    """Recognise the current explicit Hatch layout; uncertain layouts cannot skip tests."""
    config = cast("dict[str, object]", tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8")))
    build_system = _table(config["build-system"])
    hatch = _table(_table(config["tool"])["hatch"])
    build = _table(hatch["build"])
    targets = _table(build["targets"])
    wheel = _table(targets["wheel"])
    sdist = _table(targets["sdist"])
    wheel_packages = _strings(wheel["packages"])
    sdist_inputs = _strings(sdist["include"])
    project = _table(config["project"])
    readme = project["readme"]
    if (
        build_system["build-backend"] != "hatchling.build"
        or set(build_system) != {"requires", "build-backend"}
        or not all(item.startswith("hatchling==") for item in _strings(build_system["requires"]))
        or set(hatch) != {"build"}
        or set(build) != {"targets"}
        or set(targets) != {"wheel", "sdist"}
        or set(wheel) != {"packages", "force-include"}
        or set(sdist) != {"include", "exclude"}
        or not wheel_packages
        or not sdist_inputs
        or not isinstance(readme, str)
    ):
        msg = "packaging configuration needs a fresh documentation-scope review"
        raise ValueError(msg)
    # Exclusions are deliberately ignored: an overinclusive match runs full CI.
    sources = [
        readme,
        *_strings(project["license-files"]),
        *wheel_packages,
        *_table(wheel["force-include"]),
    ]
    inputs = [*sources, *sdist_inputs]
    if any(source.startswith("/") for source in sources) or any(
        re.search(r"[*?\[!\\]", pattern) or any(part in {"", ".", ".."} for part in pattern.strip("/").split("/"))
        for pattern in inputs
    ):
        # Hatch uses gitwildmatch, including zero-directory ** matches. Do not
        # approximate its glob rules with pathlib/fnmatch and risk skipping a resource.
        msg = "packaging patterns need a fresh documentation-scope review"
        raise ValueError(msg)
    return inputs


def documentation_only(root: Path, changed: list[str]) -> bool:
    """Require an explicit allowlist, no packaged input and no first-party consumer."""
    if not changed or any(
        path not in DOCUMENTATION_ONLY and not (path.startswith("docs/work/") and path.endswith(".md"))
        for path in changed
    ):
        return False
    try:
        package_inputs = _package_inputs(root)
        consumers = _git(root, "ls-files", "-z", "--", "src", "tests", "scripts").split("\0")
        text = "\n".join(
            (root / path).read_text(encoding="utf-8")
            for path in consumers
            if path and path not in {"scripts/ci_changes.py", "tests/test_ci_changes.py"}
        )
    except (KeyError, OSError, TypeError, ValueError, subprocess.CalledProcessError):
        return False
    for path in changed:
        for package_input in package_inputs:
            pattern = package_input.lstrip("/").rstrip("/")
            if path == pattern or path.startswith(pattern + "/"):
                return False
        if path in text or Path(path).name in text or (path.startswith("docs/work/") and "docs/work" in text):
            return False
    return True


def check_documentation(root: Path) -> None:
    """Check tracked Markdown text and relative link targets, including deleted targets."""
    for name in _git(root, "ls-files", "-z", "--", "*.md").split("\0"):
        if not name:
            continue
        path = root / name
        text = path.read_text(encoding="utf-8")
        if "\0" in text:
            msg = f"NUL byte in Markdown document: {name}"
            raise ValueError(msg)
        text = re.sub(r"^```[^\n]*\n.*?^```[^\n]*$", "", text, flags=re.MULTILINE | re.DOTALL)
        for match in chain(_INLINE_LINK.finditer(text), _REFERENCE_LINK.finditer(text)):
            target = match.group(1)
            url = urlsplit(target)
            if url.scheme or url.netloc or not url.path or url.path.startswith("/"):
                continue
            destination = (path.parent / unquote(url.path)).resolve()
            if not destination.is_relative_to(root.resolve()) or not destination.exists():
                msg = f"missing or outside-checkout Markdown target in {name}: {target}"
                raise ValueError(msg)


def check_changes(root: Path, event_name: str, before_sha: str = "") -> bool:
    """Return whether full CI is needed, after mandatory patch/documentation checks."""
    if event_name == "push":
        if re.fullmatch(r"[0-9a-f]{40}", before_sha) is None:
            msg = "main push needs its nonzero before SHA for whole-push whitespace validation"
            raise ValueError(msg)
        base = before_sha
        if before_sha == "0" * 40:
            if _git(root, "rev-list", "--count", "HEAD").strip() != "1":
                msg = "zero before SHA is valid only for the initial root commit"
                raise ValueError(msg)
            base = _git(root, "hash-object", "-t", "tree", "--stdin").strip()
    elif event_name in {"pull_request", "workflow_dispatch"}:
        # Default PR checkout is the merge result: its first parent is the target base.
        base = "HEAD^1"
        if event_name == "workflow_dispatch" and _git(root, "rev-list", "--count", "HEAD").strip() == "1":
            base = _git(root, "hash-object", "-t", "tree", "--stdin").strip()
    else:
        msg = f"unsupported CI event: {event_name}"
        raise ValueError(msg)
    _git(root, "diff", "--check", base, "HEAD", "--")
    changed = _git(root, "diff", "--name-only", "--no-renames", "-z", base, "HEAD", "--").split("\0")
    changed = [path for path in changed if path]
    full = event_name != "pull_request" or not documentation_only(root, changed)
    if not full:
        check_documentation(root)
    print(f"CI scope: {'full quality' if full else 'documentation (UTF-8, local links, patch whitespace)'}")
    print(f"Compared {base}..HEAD; changed paths: {len(changed)}")
    return full


def main() -> int:
    """Publish a skip output only after every applicable preflight check has succeeded."""
    full = check_changes(REPO_ROOT, os.environ["CI_EVENT_NAME"], os.environ.get("CI_BEFORE_SHA", ""))
    with Path(os.environ["GITHUB_OUTPUT"]).open("a", encoding="utf-8") as output:
        output.write(f"full={str(full).lower()}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
