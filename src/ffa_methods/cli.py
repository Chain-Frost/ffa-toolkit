"""Installed FFA CLI: offline method guidance and a bounded mathematical GEV fit."""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from ffa_methods import __version__
from ffa_methods.deterministic_workflow import calculate_fixed_gev_from_files
from ffa_methods.guidance import explain, references
from ffa_methods.reporting import write_report_bundle


def make_parser() -> argparse.ArgumentParser:
    """Expose only validated, explicit-settings mathematical calculations."""
    parser = argparse.ArgumentParser(
        prog="ffa",
        allow_abbrev=False,
        description="FFA Toolkit: independent flood frequency analysis CLI.",
        epilog=(
            "Only explicit-settings, complete-positive-AMS fixed-H GEV fitting is available; "
            "no Bayesian fitting, other distributions or native Flike qualification."
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
        help="Fit explicit fixed-H GEV L/LH mathematics to a complete positive Year,AMS CSV (unqualified)",
        description="One deterministic GEV fit; requires explicit JSON settings and complete exact positive annual maxima.",
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
            result, settings = calculate_fixed_gev_from_files(args.ams, args.settings)
            paths = write_report_bundle(result, args.output, settings=settings)
        except (ValueError, TypeError, FileExistsError) as exc:
            parser.error(str(exc))
        except OSError as exc:
            print(f"ffa: fit I/O failure: {exc}", file=sys.stderr)
            return 1
        print(
            f"Independent fixed-H GEV mathematical fit only â€” not native Flike-qualified. Reports: {paths[0].parent}"
        )
        return 0
    parser.error("no command selected; use fit, explain, references or --help")
    return 2  # Unreachable: parser.error raises SystemExit.
