from __future__ import annotations

import argparse
from pathlib import Path
import sys

from harness.drop_review import DEFAULT_CONFIG_PATH, run_drop_review_packet
from harness.triangle_markdown import render_triangle_markdown


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "drop-review":
        return _run_drop_review(args)
    if args.command == "triangle-to-markdown":
        return _run_triangle_to_markdown(args)
    parser.print_help()
    return 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Harness Native Reserving Studio commands for deterministic reserving workflows."
    )
    subparsers = parser.add_subparsers(dest="command")
    drop_review = subparsers.add_parser(
        "drop-review",
        help="Run deterministic drop review and write a markdown review packet.",
    )
    drop_review.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Config path. Defaults to examples/config_quarterly.yml.",
    )
    drop_review.add_argument(
        "--candidate-limit",
        type=int,
        default=5,
        help="Maximum drop candidates to evaluate/display. Defaults to 5.",
    )
    drop_review.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Markdown output path. Defaults to harness/artifacts with timestamp.",
    )
    triangle_markdown = subparsers.add_parser(
        "triangle-to-markdown",
        help="Render a triangle as a markdown table.",
    )
    triangle_markdown.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Config path. No default; required.",
    )
    triangle_markdown.add_argument(
        "--triangle-type",
        required=True,
        help="Triangle type to render (for example: a2a, incurred, premium).",
    )
    triangle_markdown.add_argument(
        "--view",
        choices=["cumulative", "incremental"],
        default="cumulative",
        help="Triangle view for value triangles. Defaults to cumulative. A2A supports cumulative only.",
    )
    triangle_markdown.add_argument(
        "--bolt",
        action="append",
        default=[],
        metavar="ORIGIN:DEVELOPMENT",
        help="Optional bolted cell to strike out, repeated as needed.",
    )
    triangle_markdown.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional markdown output path. If omitted, print to stdout.",
    )
    return parser


def _run_drop_review(args: argparse.Namespace) -> int:
    try:
        result = run_drop_review_packet(
            config_path=args.config,
            output_path=args.output,
            candidate_limit=args.candidate_limit,
        )
    except Exception as exc:  # CLI boundary: keep failures concise for harnesses.
        print(f"drop-review failed: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote drop review packet: {result.output_path}")
    print(f"Candidates: {result.candidate_count}")
    if result.candidate_id:
        print(f"Recommendation: {result.candidate_id} ({result.recommendation_class})")
    else:
        print("Recommendation: none")
    if result.warnings:
        print("Warnings:")
        for warning in result.warnings:
            print(f"- {warning}")
    return 0


def _run_triangle_to_markdown(args: argparse.Namespace) -> int:
    try:
        bolted_drops = _parse_bolt_arguments(args.bolt)
        result = render_triangle_markdown(
            config_path=args.config,
            triangle_type=args.triangle_type,
            triangle_view=args.view,
            bolted_drops=bolted_drops,
            output_path=args.output,
        )
    except Exception as exc:  # CLI boundary: keep failures concise for harnesses.
        print(f"triangle-to-markdown failed: {exc}", file=sys.stderr)
        return 1

    if result.output_path is not None:
        print(f"Wrote triangle markdown: {result.output_path}")
    else:
        print(result.markdown)
    return 0


def _parse_bolt_arguments(values: list[str]) -> list[tuple[str, int]]:
    bolted_drops: list[tuple[str, int]] = []
    for value in values:
        if ":" not in value:
            raise ValueError(
                f"Invalid bolt value '{value}'. Expected ORIGIN:DEVELOPMENT"
            )
        origin, development_age = value.split(":", 1)
        origin = origin.strip()
        development_age = development_age.strip()
        if not origin or not development_age:
            raise ValueError(
                f"Invalid bolt value '{value}'. Expected ORIGIN:DEVELOPMENT"
            )
        bolted_drops.append((origin, int(development_age)))
    return bolted_drops


if __name__ == "__main__":
    raise SystemExit(main())
