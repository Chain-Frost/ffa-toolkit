"""Installed FFA CLI: offline method guidance and bounded mathematical L/LH fits."""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from ffa_methods import __version__
from ffa_methods.deterministic_workflow import ORDINARY_SCHEMA, calculate_from_bytes
from ffa_methods.guidance import explain, references
from ffa_methods.reporting import write_report_bundle


def make_parser() -> argparse.ArgumentParser:
    """Expose only validated, explicit-settings mathematical calculations."""
    parser = argparse.ArgumentParser(
        prog="ffa",
        allow_abbrev=False,
        description="FFA Toolkit: independent flood frequency analysis CLI.",
        epilog=(
            "Explicit settings for complete positive annual maxima support ordinary L-moments, "
            "fixed-H and selected-H GEV, and parametric bootstrap."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__} (legacy target: Flike 5.0.306.0; file 3.10)",
    )
    commands = parser.add_subparsers(dest="command", required=False)
    explain_parser = commands.add_parser(
        "explain",
        allow_abbrev=False,
        help="Read packaged method guidance and recorded qualification status",
        description="Show offline guidance; a registered method is not proof of legacy qualification.",
    )
    explain_parser.add_argument("method_id", nargs="?", help="registered method ID; omit to list all entries")
    explain_parser.add_argument("--format", choices=("text", "json", "markdown"), default="text", dest="output_format")
    references_parser = commands.add_parser(
        "references",
        allow_abbrev=False,
        help="Read the bundled scientific and software reference register offline",
    )
    references_parser.add_argument("source_key", nargs="?", help="optional citation key, e.g. ARR3 or WANG1997")
    fit_parser = commands.add_parser(
        "fit",
        allow_abbrev=False,
        help="Fit explicit deterministic L/LH mathematics to a complete positive Year,AMS CSV (unqualified)",
        description="Requires explicit JSON settings and complete exact positive annual maxima; see ffa explain l-moments.",
    )
    fit_parser.add_argument("--ams", required=True, type=Path, metavar="YEAR_AMS_CSV")
    fit_parser.add_argument("--settings", required=True, type=Path, metavar="SETTINGS_JSON")
    fit_parser.add_argument("--output", required=True, type=Path, metavar="NEW_REPORT_DIRECTORY")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Return 0 on success, 2 for invalid inputs, 1 for IO/broken resources."""
    parser = make_parser()
    args = parser.parse_args(argv)
    if args.command in ("explain", "references"):
        try:
            content = (
                explain(args.method_id, output_format=args.output_format)
                if args.command == "explain"
                else references(args.source_key)
            )
            print(content, end="")
        except LookupError as exc:
            parser.error(str(exc))
        except (OSError, TypeError, ValueError) as exc:
            print(f"ffa: failed to read packaged method guidance: {exc}", file=sys.stderr)
            return 1
        return 0
    if args.command == "fit":
        try:
            raw_ams = args.ams.read_bytes()
            raw_settings = args.settings.read_bytes()
            result, settings = calculate_from_bytes(raw_ams, raw_settings)
            source_inputs = (raw_ams, raw_settings) if settings["schema_version"] == ORDINARY_SCHEMA else None
            paths = write_report_bundle(result, args.output, settings=settings, source_inputs=source_inputs)
        except (ValueError, TypeError, OverflowError, FileExistsError) as exc:
            parser.error(str(exc))
        except OSError as exc:
            print(f"ffa: fit I/O failure: {exc}", file=sys.stderr)
            return 1
        print(f"Independent mathematical fit only — not native Flike-qualified. Reports: {paths[0].parent}")
        return 0
    parser.error("no command selected; use fit, explain, references or --help")
    return 2  # Unreachable: parser.error raises SystemExit.
