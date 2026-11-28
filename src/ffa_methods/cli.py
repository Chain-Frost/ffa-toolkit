"""Installed FFA CLI: offline guidance and explicitly configured mathematical fits."""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from ffa_methods import __version__
from ffa_methods.bayesian_workflow import BAYESIAN_SCHEMA, BAYESIAN_SCHEMAS, SHAPE_SCHEMA
from ffa_methods.deterministic_workflow import ORDINARY_SCHEMA, calculate_from_bytes
from ffa_methods.guidance import explain, method_manual, references
from ffa_methods.reporting import validate_output_path, write_report_bundle


def make_parser() -> argparse.ArgumentParser:
    """Expose only validated, explicit-settings mathematical calculations."""
    parser = argparse.ArgumentParser(
        prog="ffa",
        allow_abbrev=False,
        description="FFA Toolkit: independent flood frequency analysis CLI.",
        epilog=(
            "Explicit settings for complete positive annual maxima support fixed-H GEV and ordinary L-moment fits, "
            "selected-H GEV and parametric bootstrap. Explicit lognormal and bounded-shape GEV/LP3 Bayesian fits are available."
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
    commands.add_parser("manual", allow_abbrev=False, help="Write the offline HTML method manual to standard output")
    references_parser = commands.add_parser(
        "references",
        allow_abbrev=False,
        help="Read the bundled scientific and software reference register offline",
    )
    references_parser.add_argument("source_key", nargs="?", help="optional citation key, e.g. ARR3 or WANG1997")
    fit_parser = commands.add_parser(
        "fit",
        allow_abbrev=False,
        help="Fit explicit deterministic or configured Bayesian mathematics (unqualified)",
        description="Requires explicit JSON settings and admitted annual observations; see ffa explain quantile-estimands.",
    )
    fit_parser.add_argument("--ams", required=True, type=Path, metavar="ANNUAL_OBSERVATIONS_CSV")
    fit_parser.add_argument("--settings", required=True, type=Path, metavar="SETTINGS_JSON")
    fit_parser.add_argument("--output", required=True, type=Path, metavar="NEW_REPORT_DIRECTORY")
    fit_parser.add_argument(
        "--frequency-report", action="store_true", help="Add static frequency SVG, coordinate CSV and HTML report"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Return 0 on success, 2 for invalid inputs, 1 for IO/broken resources."""
    parser = make_parser()
    args = parser.parse_args(argv)
    if args.command == "manual":
        try:
            sys.stdout.buffer.write(method_manual().encode("utf-8"))
        except (OSError, TypeError, ValueError) as exc:
            print(f"ffa: failed to write packaged method manual: {exc}", file=sys.stderr)
            return 1
        return 0
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
            validate_output_path(args.output)
            result, settings = calculate_from_bytes(raw_ams, raw_settings)
            source_inputs = (
                (raw_ams, raw_settings)
                if args.frequency_report or settings["schema_version"] in (ORDINARY_SCHEMA, *BAYESIAN_SCHEMAS)
                else None
            )
            paths = write_report_bundle(
                result,
                args.output,
                settings=settings,
                source_inputs=source_inputs,
                frequency_report=args.frequency_report,
            )
        except (ValueError, TypeError, OverflowError, FileExistsError) as exc:
            parser.error(str(exc))
        except OSError as exc:
            print(f"ffa: fit I/O failure: {exc}", file=sys.stderr)
            return 1
        if settings.get("bootstrap_status") == "refused":
            print(
                f"ffa: {settings['bootstrap_refusal_detail']}. Point fit and attempt ledger: {paths[0].parent}",
                file=sys.stderr,
            )
            return 2
        mode = "lognormal Bayesian" if settings["schema_version"] == BAYESIAN_SCHEMA else "deterministic"
        if settings["schema_version"] == SHAPE_SCHEMA:
            mode = "bounded-shape Bayesian"
        print(f"Independent {mode} mathematical fit only — not native Flike-qualified. Reports: {paths[0].parent}")
        return 0
    parser.error("no command selected; use fit, explain, references or --help")
    return 2  # Unreachable: parser.error raises SystemExit.
