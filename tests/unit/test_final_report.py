from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harness.cli import main
from harness.final_report import write_final_report


def test_write_final_report_uses_configured_output_dir(tmp_path: Path) -> None:
    output_dir = tmp_path / "final_reports"
    input_path = tmp_path / "drop_review.md"
    input_path.write_text("# Drop Review\n\nUse this packet.\n", encoding="utf-8")
    config_path = _write_config(tmp_path, output_dir)

    result = write_final_report(
        config_path=config_path,
        conversation_id="quarterly/drop review",
        title="Quarterly Drop Review",
        input_files=[("drop-review", input_path)],
        command="uv run python -m harness.cli final-report",
    )

    assert result.output_path == output_dir / "quarterly_drop_review.md"
    packet = result.output_path.read_text(encoding="utf-8")
    assert "# Quarterly Drop Review" in packet
    assert "## Results" in packet
    assert "### Drop review" in packet
    assert "Use this packet." in packet


def test_write_final_report_supports_scaffold_tokens(tmp_path: Path) -> None:
    output_dir = tmp_path / "final_reports"
    drop_review_path = tmp_path / "drop_review.md"
    triangle_path = tmp_path / "triangle.md"
    drop_review_path.write_text("# Drop Review\n\nDrop review content.\n", encoding="utf-8")
    triangle_path.write_text("| Origin | 3 |\n| --- | --- |\n| 2000 | 1.000 |\n", encoding="utf-8")
    config_path = _write_config(tmp_path, output_dir)

    result = write_final_report(
        config_path=config_path,
        conversation_id="scaffolded-report",
        title="Scaffolded Report",
        scaffold_markdown="## Summary\n\nSee the drop review below.\n\n{{input:drop-review}}\n\n## Tables\n\n{{input:a2a-triangle}}",
        input_files=[("drop-review", drop_review_path), ("a2a triangle", triangle_path)],
        command="uv run python -m harness.cli final-report",
    )

    packet = result.output_path.read_text(encoding="utf-8")
    assert "## Summary" in packet
    assert "Drop review content." in packet
    assert "| Origin | 3 |" in packet


def test_final_report_cli_rejects_missing_artifact(tmp_path: Path) -> None:
    output_dir = tmp_path / "final_reports"
    body_path = tmp_path / "body.md"
    body_path.write_text("## Recommendation\n\nStore the final result only.\n", encoding="utf-8")
    config_path = _write_config(tmp_path, output_dir)

    exit_code = main(
        [
            "final-report",
            "--config",
            str(config_path),
            "--conversation-id",
            "missing-artifact",
            "--title",
            "Missing Artifact",
            "--body-file",
            str(body_path),
            "--artifact",
            f"Missing:{tmp_path / 'missing.md'}",
        ]
    )

    assert exit_code == 1


def _write_config(tmp_path: Path, output_dir: Path) -> Path:
    config_path = tmp_path / "config.yml"
    config_path.write_text(
        "\n".join(
            [
                "paths:",
                "  results: results/",
                "  plots: plots/",
                "  data: data/",
                "  sessions: sessions/",
                "first date: 1900",
                'last date: "December 2006"',
                "segment: quarterly",
                "granularity: quarterly",
                "workflow:",
                "  dataset: quarterly",
                "  quarterly_premium_csv: data/quarterly_premium.csv",
                "session:",
                "  path: sessions/quarterly.yml",
                "ai:",
                "  final_reports:",
                f"    path: {output_dir}",
            ]
        ),
        encoding="utf-8",
    )
    return config_path
