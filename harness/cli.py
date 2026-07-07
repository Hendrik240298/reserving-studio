from __future__ import annotations

import argparse
from pathlib import Path
import sys

from harness.drop_review import DEFAULT_CONFIG_PATH, run_drop_review_packet
from harness.final_report import (
    DEFAULT_CONFIG_PATH as DEFAULT_FINAL_REPORT_CONFIG_PATH,
    write_final_report,
)
from harness.ldf_compare import run_ldf_compare
from harness.triangle_markdown import render_triangle_markdown


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "drop-review":
        return _run_drop_review(args)
    if args.command == "ldf-compare":
        return _run_ldf_compare(args)
    if args.command == "triangle-to-markdown":
        return _run_triangle_to_markdown(args)
    if args.command == "final-report":
        return _run_final_report(args)
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
    ldf_compare = subparsers.add_parser(
        "ldf-compare",
        help="Compare LDF vectors for explicit reserving scenario settings.",
    )
    ldf_compare.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Config path. Defaults to examples/config_quarterly.yml.",
    )
    ldf_compare.add_argument(
        "--scenarios-file",
        type=Path,
        default=None,
        help="YAML file containing a scenarios mapping.",
    )
    ldf_compare.add_argument(
        "--scenario-json",
        action="append",
        default=[],
        metavar="NAME:JSON",
        help="Scenario settings as NAME:JSON. Repeat as needed.",
    )
    ldf_compare.add_argument(
        "--delta-threshold",
        type=float,
        default=0.01,
        help="Absolute LDF delta threshold for red plot markers. Defaults to 0.01.",
    )
    ldf_compare.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Markdown output path. Defaults to harness/artifacts with timestamp. PNG is written beside it.",
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
    final_report = subparsers.add_parser(
        "final-report",
        help="Compose and write one final markdown report for one conversation.",
    )
    final_report.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_FINAL_REPORT_CONFIG_PATH,
        help="Config path. Defaults to examples/config_quarterly.yml.",
    )
    final_report.add_argument(
        "--conversation-id",
        required=True,
        help="Stable conversation identifier used for the report filename.",
    )
    final_report.add_argument(
        "--title",
        required=True,
        help="Markdown title for the final report.",
    )
    final_report.add_argument(
        "--body-file",
        type=Path,
        default=None,
        help="Optional AI-authored markdown body content. Legacy direct-body mode.",
    )
    final_report.add_argument(
        "--scaffold-file",
        type=Path,
        default=None,
        help="Optional AI-authored scaffold markdown with {{input:label}} tokens.",
    )
    final_report.add_argument(
        "--input-file",
        action="append",
        default=[],
        metavar="LABEL:PATH",
        help="Input markdown file to compose into the final report, repeated as needed.",
    )
    final_report.add_argument(
        "--artifact",
        action="append",
        default=[],
        metavar="LABEL:PATH",
        help="Artifact reference to append to the report appendix, repeated as needed.",
    )
    final_report.add_argument(
        "--warning",
        action="append",
        default=[],
        help="Warning line to record in the report, repeated as needed.",
    )
    output_group = final_report.add_mutually_exclusive_group()
    output_group.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Explicit markdown output path.",
    )
    output_group.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for <conversation-id>.md. Defaults to ai.final_reports.path.",
    )
    return parser


def _run_drop_review(args: argparse.Namespace) -> int:
    try:
        output_path = run_drop_review_packet(
            config_path=args.config,
            candidate_limit=args.candidate_limit,
            output_path=args.output,
        )
    except Exception as exc:  # CLI boundary: keep failures concise for harnesses.
        print(f"drop-review failed: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote drop review packet: {output_path}")
    return 0


def _run_ldf_compare(args: argparse.Namespace) -> int:
    try:
        output_path = run_ldf_compare(
            config_path=args.config,
            scenarios_file=args.scenarios_file,
            scenario_json=args.scenario_json,
            delta_threshold=args.delta_threshold,
            output_path=args.output,
            command=_command_for_ldf_compare(
                config_path=args.config,
                scenarios_file=args.scenarios_file,
                scenario_json=args.scenario_json,
                delta_threshold=args.delta_threshold,
                output_path=args.output,
            ),
        )
    except Exception as exc:  # CLI boundary: keep failures concise for harnesses.
        print(f"ldf-compare failed: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote ldf compare packet: {output_path}")
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


def _run_final_report(args: argparse.Namespace) -> int:
    try:
        if args.body_file is not None:
            body_path = args.body_file.resolve() if args.body_file.is_absolute() else args.body_file
            body_file = Path(body_path)
            if not body_file.exists():
                raise FileNotFoundError(f"Body file '{body_file}' does not exist")
            body_markdown = body_file.read_text(encoding="utf-8")
        else:
            body_markdown = None
        if args.scaffold_file is not None:
            scaffold_path = (
                args.scaffold_file.resolve()
                if args.scaffold_file.is_absolute()
                else args.scaffold_file
            )
            scaffold_file = Path(scaffold_path)
            if not scaffold_file.exists():
                raise FileNotFoundError(f"Scaffold file '{scaffold_file}' does not exist")
            scaffold_markdown = scaffold_file.read_text(encoding="utf-8")
        else:
            scaffold_markdown = None
        input_files = _parse_input_file_arguments(args.input_file)
        artifact_references = _parse_artifact_arguments(args.artifact)
        result = write_final_report(
            config_path=args.config,
            conversation_id=args.conversation_id,
            title=args.title,
            body_markdown=body_markdown,
            scaffold_markdown=scaffold_markdown,
            input_files=input_files,
            artifact_references=artifact_references,
            warnings=args.warning,
            output_path=args.output,
            output_dir=args.output_dir,
            command=_command_for_final_report(
                config_path=args.config,
                conversation_id=args.conversation_id,
                title=args.title,
                body_file=args.body_file,
                scaffold_file=args.scaffold_file,
                input_files=args.input_file,
                artifacts=args.artifact,
                warnings=args.warning,
                output_path=args.output,
                output_dir=args.output_dir,
            ),
        )
    except Exception as exc:  # CLI boundary: keep failures concise for harnesses.
        print(f"final-report failed: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote final report: {result.output_path}")
    print(f"Inputs: {result.input_count}")
    print(f"Artifacts: {result.artifact_count}")
    if result.warnings:
        print("Warnings:")
        for warning in result.warnings:
            print(f"- {warning}")
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


def _parse_artifact_arguments(values: list[str]) -> list[tuple[str, Path]]:
    return _parse_labeled_path_arguments(values, field_name="artifact")


def _parse_input_file_arguments(values: list[str]) -> list[tuple[str, Path]]:
    return _parse_labeled_path_arguments(values, field_name="input file")


def _command_for_final_report(
    *,
    config_path: Path,
    conversation_id: str,
    title: str,
    body_file: Path | None,
    scaffold_file: Path | None,
    input_files: list[str],
    artifacts: list[str],
    warnings: list[str],
    output_path: Path | None,
    output_dir: Path | None,
) -> str:
    parts = [
        "uv run python -m harness.cli final-report",
        f"--config {_shell_quote(str(config_path))}",
        f"--conversation-id {_shell_quote(conversation_id)}",
        f"--title {_shell_quote(title)}",
    ]
    if body_file is not None:
        parts.append(f"--body-file {_shell_quote(str(body_file))}")
    if scaffold_file is not None:
        parts.append(f"--scaffold-file {_shell_quote(str(scaffold_file))}")
    for input_file in input_files:
        parts.append(f"--input-file {_shell_quote(input_file)}")
    for artifact in artifacts:
        parts.append(f"--artifact {_shell_quote(artifact)}")
    for warning in warnings:
        parts.append(f"--warning {_shell_quote(warning)}")
    if output_path is not None:
        parts.append(f"--output {_shell_quote(str(output_path))}")
    if output_dir is not None:
        parts.append(f"--output-dir {_shell_quote(str(output_dir))}")
    return " ".join(parts)


def _command_for_ldf_compare(
    *,
    config_path: Path,
    scenarios_file: Path | None,
    scenario_json: list[str],
    delta_threshold: float,
    output_path: Path | None,
) -> str:
    parts = [
        "uv run python -m harness.cli ldf-compare",
        f"--config {_shell_quote(str(config_path))}",
    ]
    if scenarios_file is not None:
        parts.append(f"--scenarios-file {_shell_quote(str(scenarios_file))}")
    for value in scenario_json:
        parts.append(f"--scenario-json {_shell_quote(value)}")
    parts.extend(
        [
            f"--delta-threshold {delta_threshold}",
        ]
    )
    if output_path is not None:
        parts.append(f"--output {_shell_quote(str(output_path))}")
    return " ".join(parts)


def _parse_labeled_path_arguments(
    values: list[str], *, field_name: str
) -> list[tuple[str, Path]]:
    items: list[tuple[str, Path]] = []
    for value in values:
        if ":" not in value:
            raise ValueError(
                f"Invalid {field_name} value '{value}'. Expected LABEL:PATH"
            )
        label, path = value.split(":", 1)
        label = label.strip()
        path = path.strip()
        if not label or not path:
            raise ValueError(
                f"Invalid {field_name} value '{value}'. Expected LABEL:PATH"
            )
        items.append((label, Path(path)))
    return items


def _shell_quote(value: str) -> str:
    if value == "":
        return "''"
    if all(char.isalnum() or char in "._/-:" for char in value):
        return value
    return "'" + value.replace("'", "'\"'\"'") + "'"


if __name__ == "__main__":
    raise SystemExit(main())
