"""Offline explanation of methods from the authoritative packaged register.

The register is a research/status catalogue, not an admission of legacy conformance.
This module is internal to the CLI and reporting workflow.
"""

import json
from importlib.resources import files
from pathlib import Path
from typing import cast

from ffa_methods.compatibility import LEGACY_FLIKE_PROFILE


def _read_resource(resource_name: str) -> str:
    """Use bundled wheel data, or the authoritative docs in an editable source checkout."""
    resource = files("ffa_methods").joinpath("resources", resource_name)
    if resource.is_file():
        return resource.read_text(encoding="utf-8")
    source_root = Path(__file__).resolve().parents[2]
    if (source_root / "pyproject.toml").is_file() and (source_root / "src/ffa_methods/guidance.py").is_file():
        return (source_root / "docs" / resource_name).read_text(encoding="utf-8")
    msg = f"required packaged reference resource is absent: {resource_name}"
    raise FileNotFoundError(msg)


def _read_catalogue() -> tuple[dict[str, object], list[dict[str, object]]]:
    """Load and check the bundled register, never a writable checkout-relative path."""
    raw: object = json.loads(_read_resource("method-registry.json"))
    if not isinstance(raw, dict):
        msg = "packaged method register is not a JSON object"
        raise TypeError(msg)
    catalogue = cast(dict[str, object], raw)
    if catalogue.get("compatibility_profile") != LEGACY_FLIKE_PROFILE.identifier:
        msg = "packaged method register has a different compatibility target"
        raise ValueError(msg)
    raw_methods = catalogue.get("methods")
    if not isinstance(raw_methods, list):
        msg = "packaged method register has no method list"
        raise TypeError(msg)
    methods: list[dict[str, object]] = []
    seen: set[str] = set()
    for entry in cast(list[object], raw_methods):
        if not isinstance(entry, dict):
            msg = "packaged method register contains a non-object method"
            raise TypeError(msg)
        method = cast(dict[str, object], entry)
        identifier = method.get("method_id")
        if not isinstance(identifier, str) or not identifier or identifier in seen:
            msg = "packaged method register contains missing or repeated method IDs"
            raise ValueError(msg)
        for required in ("title", "purpose", "scope", "implementation_status", "qualification_status"):
            if not isinstance(method.get(required), str):
                msg = f"packaged method {identifier!r} lacks a valid {required}"
                raise TypeError(msg)
        seen.add(identifier)
        methods.append(method)
    return catalogue, methods


def get_method(method_id: str) -> dict[str, object]:
    """Return recorded method guidance, without promoting the implementation status."""
    _, methods = _read_catalogue()
    for method in methods:
        if method["method_id"] == method_id:
            return method
    msg = f"unknown method {method_id!r}; use 'ffa explain' to list registered IDs"
    raise LookupError(msg)


def _strings(method: dict[str, object], field: str) -> list[str]:
    raw = method.get(field, [])
    if not isinstance(raw, list):
        msg = f"method {method['method_id']!r} has an invalid {field} list"
        raise TypeError(msg)
    raw_items = cast(list[object], raw)
    if any(not isinstance(value, str) for value in raw_items):
        msg = f"method {method['method_id']!r} has an invalid {field} list"
        raise TypeError(msg)
    return cast(list[str], raw)


def _offline_page(method: dict[str, object]) -> str:
    """Read one explicitly registered original Markdown page from the installed wheel."""
    method_id = method["method_id"]
    page = method.get("documentation")
    if not isinstance(page, str) or page != f"methods/{method_id}.md":
        msg = f"no bundled technical page for {method_id!r}; use the text or JSON register summary"
        raise LookupError(msg)
    text = _read_resource(page)
    return text if text.endswith("\n") else text + "\n"


def method_manual() -> str:
    """Read the self-contained HTML manual included in the distribution."""
    return _read_resource("methods/manual.html")


def explain(method_id: str | None = None, *, output_format: str = "text") -> str:
    """Render method guidance in text or JSON; neither form asserts qualification."""
    if output_format not in ("text", "json", "markdown"):
        msg = f"unsupported explanation format: {output_format}"
        raise ValueError(msg)
    catalogue, methods = _read_catalogue()
    selected = next((method for method in methods if method["method_id"] == method_id), None)
    if method_id is not None and selected is None:
        msg = f"unknown method {method_id!r}; use 'ffa explain' to list registered IDs"
        raise LookupError(msg)
    if output_format == "markdown":
        if selected is None:
            msg = "Markdown explanation requires a method ID; use 'ffa explain' to list methods"
            raise LookupError(msg)
        return _offline_page(selected)
    if output_format == "json":
        data = {
            "schema_version": "ffa-method-guidance/1",
            "compatibility_profile": catalogue["compatibility_profile"],
            "status_notice": "Register status is not evidence of an implemented or legacy-qualified calculation.",
            "methods": methods if selected is None else [selected],
        }
        return json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    return _render_text_explanation(selected, methods)


def _render_text_explanation(selected: dict[str, object] | None, methods: list[dict[str, object]]) -> str:
    """Render a readable summary without requiring all method pages to be bundled."""
    header = (
        "FFA method catalogue — legacy TUFLOW/BMT Flike 5.0.306.0 (file 3.10)\n"
        "GUIDANCE ONLY: register entries do not establish that calculations are implemented or legacy-qualified."
    )
    if selected is None:
        lines = [header, "", "Registered methods (use 'ffa explain METHOD_ID' for details):"]
        lines.extend(
            f"  {method['method_id']} — {method['title']} "
            f"[scope={method['scope']}; implementation={method['implementation_status']}; "
            f"qualification={method['qualification_status']}]"
            for method in methods
        )
        return "\n".join(lines) + "\n"
    lines = [
        header,
        "",
        f"{selected['title']} ({selected['method_id']})",
        f"Scope: {selected['scope']}",
        f"Register implementation status: {selected['implementation_status']}",
        f"Register qualification status: {selected['qualification_status']}",
        f"Purpose: {selected['purpose']}",
    ]
    for field, title in (
        ("key_assumptions", "Assumptions"),
        ("evidence_gaps", "Outstanding evidence gaps"),
        ("validation_required", "Required verification"),
    ):
        values = _strings(selected, field)
        if values:
            lines.append(f"{title}:")
            lines.extend(f"  - {value}" for value in values)
    lines.append(f"Primary/source identifiers: {', '.join(_strings(selected, 'source_keys')) or 'not supplied'}")
    source = selected.get("source_location")
    if isinstance(source, str) and source:
        lines.append(f"Source locations: {source}")
    lines.append("Full source citations: packaged resources/references.md (also docs/references.md in the repository)")
    page = selected.get("documentation")
    if isinstance(page, str) and page:
        if page == f"methods/{selected['method_id']}.md":
            lines.append(f"Bundled technical explanation: ffa explain {selected['method_id']} --format markdown")
        else:
            lines.append(f"Method specification: docs/{page} (source repository only; not bundled)")
    lines.append("Independent mathematical checks do not by themselves qualify legacy Flike compatibility.")
    return "\n".join(lines) + "\n"


def references(source_key: str | None = None) -> str:
    """Read an offline bibliographic entry by its exact registered heading/key."""
    register = _read_resource("references.md")
    if source_key is None:
        return register if register.endswith("\n") else register + "\n"
    lines = register.splitlines(keepends=True)
    heading = f"### {source_key}"
    start = next((index for index, line in enumerate(lines) if line.strip() == heading), None)
    if start is None:
        msg = f"reference key {source_key!r} not found; use 'ffa references' for the register"
        raise LookupError(msg)
    end = next(
        (index for index in range(start + 1, len(lines)) if lines[index].startswith(("# ", "## ", "### "))),
        len(lines),
    )
    return "".join(lines[start:end]).rstrip() + "\n"
