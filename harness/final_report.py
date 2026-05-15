from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import re
import subprocess
from typing import Iterable

from harness.markdown import render_final_report
from source.config_manager import ConfigManager


REPO_ROOT = Path(__file__).parents[1]
DEFAULT_CONFIG_PATH = Path("examples/config_quarterly.yml")
DEFAULT_FINAL_REPORT_DIR = Path("harness/artifacts/final_reports")


@dataclass(frozen=True)
class InputDocument:
    label: str
    path: Path
    content: str


@dataclass(frozen=True)
class FinalReportArtifact:
    label: str
    path: Path
    kind: str = "artifact"


@dataclass(frozen=True)
class FinalReportResult:
    output_path: Path
    conversation_id: str
    input_count: int
    artifact_count: int
    warnings: list[str] = field(default_factory=list)


def write_final_report(
    *,
    config_path: Path = DEFAULT_CONFIG_PATH,
    conversation_id: str,
    title: str,
    body_markdown: str | None = None,
    scaffold_markdown: str | None = None,
    input_files: Iterable[tuple[str, Path]] | None = None,
    artifact_references: Iterable[tuple[str, Path]] | None = None,
    warnings: list[str] | None = None,
    output_path: Path | None = None,
    output_dir: Path | None = None,
    command: str,
) -> FinalReportResult:
    """Compose and write one final markdown report for a conversation."""

    normalized_conversation_id = _normalize_required_text(
        conversation_id, field_name="conversation_id"
    )
    normalized_title = _normalize_required_text(title, field_name="title")
    if output_path is not None and output_dir is not None:
        raise ValueError("Specify either output_path or output_dir, not both")
    if body_markdown is not None and scaffold_markdown is not None:
        raise ValueError("Specify either body_markdown or scaffold_markdown, not both")

    resolved_input_documents = _resolve_input_documents(input_files or [])
    if body_markdown is not None and resolved_input_documents:
        raise ValueError("body_markdown cannot be combined with input_files")
    if scaffold_markdown is not None and not resolved_input_documents:
        raise ValueError("scaffold_markdown requires at least one input file")
    if body_markdown is None and not resolved_input_documents:
        raise ValueError("Provide body_markdown or at least one input file")

    resolved_config_path = _resolve_repo_path(config_path)
    config = ConfigManager.from_yaml(resolved_config_path)
    resolved_output_path = _resolve_output_path(
        config,
        conversation_id=normalized_conversation_id,
        output_path=output_path,
        output_dir=output_dir,
    )
    resolved_artifacts = _resolve_artifact_references(artifact_references or [])
    all_artifacts = _collect_artifacts(resolved_input_documents, resolved_artifacts)
    warning_lines = [str(item).strip() for item in (warnings or []) if str(item).strip()]

    if body_markdown is not None:
        normalized_body_markdown = body_markdown.strip()
        if not normalized_body_markdown:
            raise ValueError("body_markdown is empty")
        composed_body = normalized_body_markdown
        composition_mode = "legacy_body"
    else:
        composed_body = _compose_body(resolved_input_documents, scaffold_markdown)
        composition_mode = "scaffold_compose" if scaffold_markdown is not None else "default_compose"

    requested_inputs = {
        "config_path": _display_path(resolved_config_path),
        "conversation_id": normalized_conversation_id,
        "title": normalized_title,
        "input_count": len(resolved_input_documents),
        "artifact_count": len(all_artifacts),
        "composition_mode": composition_mode,
    }
    effective_inputs = {
        "segment": config.get_segment(),
        "granularity": config.get_granularity(),
        "dataset": config.get_workflow_dataset(),
        "final_report_dir": _display_path(resolved_output_path.parent),
    }
    report = render_final_report(
        title=normalized_title,
        conversation_id=normalized_conversation_id,
        body_markdown=composed_body,
        artifact_references=[
            {"label": item.label, "path": _display_path(item.path), "kind": item.kind}
            for item in all_artifacts
        ],
        requested_inputs=requested_inputs,
        effective_inputs=effective_inputs,
        included_inputs=[
            f"- {item.label}: `{_display_path(item.path)}`" for item in resolved_input_documents
        ] or ["- No explicit input files recorded."],
        command=command,
        warnings=warning_lines,
        timestamp=datetime.now(timezone.utc).isoformat(),
        code_version=_git_revision(),
    )

    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_output_path.write_text(report, encoding="utf-8")
    return FinalReportResult(
        output_path=resolved_output_path,
        conversation_id=normalized_conversation_id,
        input_count=len(resolved_input_documents),
        artifact_count=len(all_artifacts),
        warnings=warning_lines,
    )


def _resolve_output_path(
    config: ConfigManager,
    *,
    conversation_id: str,
    output_path: Path | None,
    output_dir: Path | None,
) -> Path:
    if output_path is not None:
        return _resolve_repo_path(output_path)

    target_dir = output_dir
    if target_dir is None:
        target_dir = config.get_ai_final_report_path()
        if not str(target_dir).strip():
            target_dir = DEFAULT_FINAL_REPORT_DIR
    resolved_dir = _resolve_repo_path(target_dir)
    return resolved_dir / f"{_safe_conversation_id(conversation_id)}.md"


def _resolve_input_documents(input_files: Iterable[tuple[str, Path]]) -> list[InputDocument]:
    resolved: list[InputDocument] = []
    labels_seen: set[str] = set()
    for label, path in input_files:
        normalized_label = _normalize_required_text(label, field_name="input label")
        normalized_key = _normalize_input_key(normalized_label)
        if normalized_key in labels_seen:
            raise ValueError(f"Duplicate input label '{normalized_label}'")
        labels_seen.add(normalized_key)
        resolved_path = _resolve_repo_path(path)
        if not resolved_path.exists():
            raise FileNotFoundError(
                f"Input file '{_display_path(resolved_path)}' does not exist."
            )
        resolved.append(
            InputDocument(
                label=normalized_label,
                path=resolved_path,
                content=resolved_path.read_text(encoding="utf-8").strip(),
            )
        )
    return resolved


def _resolve_artifact_references(
    artifact_references: Iterable[tuple[str, Path]],
) -> list[FinalReportArtifact]:
    resolved: list[FinalReportArtifact] = []
    for label, path in artifact_references:
        normalized_label = _normalize_required_text(label, field_name="artifact label")
        resolved_path = _resolve_repo_path(path)
        if not resolved_path.exists():
            raise FileNotFoundError(
                f"Artifact path '{_display_path(resolved_path)}' does not exist."
            )
        resolved.append(FinalReportArtifact(label=normalized_label, path=resolved_path))
    return resolved


def _collect_artifacts(
    input_documents: list[InputDocument],
    manual_artifacts: list[FinalReportArtifact],
) -> list[FinalReportArtifact]:
    artifacts: list[FinalReportArtifact] = []
    seen: set[tuple[str, str, str]] = set()
    for item in [
        *manual_artifacts,
        *[
            FinalReportArtifact(label=document.label, path=document.path, kind="input")
            for document in input_documents
        ],
    ]:
        key = (item.label, str(item.path), item.kind)
        if key in seen:
            continue
        seen.add(key)
        artifacts.append(item)
    return artifacts


def _compose_body(
    input_documents: list[InputDocument],
    scaffold_markdown: str | None,
) -> str:
    if scaffold_markdown is not None:
        normalized_scaffold = scaffold_markdown.strip()
        if not normalized_scaffold:
            raise ValueError("scaffold_markdown is empty")
        return _compose_body_from_scaffold(normalized_scaffold, input_documents)
    return _default_body_from_inputs(input_documents)


def _compose_body_from_scaffold(
    scaffold_markdown: str,
    input_documents: list[InputDocument],
) -> str:
    rendered = scaffold_markdown
    input_map = {_normalize_input_key(item.label): item for item in input_documents}
    for raw_token in re.findall(r"\{\{input:([^}]+)\}\}", rendered):
        normalized_token = _normalize_input_key(raw_token)
        input_document = input_map.get(normalized_token)
        if input_document is None:
            continue
        rendered = rendered.replace(
            f"{{{{input:{raw_token}}}}}", input_document.content
        )
    rendered = rendered.replace("{{inputs}}", _default_body_from_inputs(input_documents))
    missing_tokens = re.findall(r"\{\{input:([^}]+)\}\}", rendered)
    if missing_tokens:
        missing = ", ".join(sorted(set(token.strip() for token in missing_tokens)))
        raise ValueError(f"Scaffold references unknown input(s): {missing}")
    return rendered


def _default_body_from_inputs(input_documents: list[InputDocument]) -> str:
    if not input_documents:
        return "## Results\n\nNo input files were available to compose."
    lines: list[str] = ["## Results", ""]
    for input_document in input_documents:
        lines.append(f"### {_label_to_title(input_document.label)}")
        lines.append("")
        lines.append(input_document.content)
        lines.append("")
    return "\n".join(lines).strip()


def _normalize_required_text(value: str, *, field_name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{field_name} is required")
    return normalized


def _normalize_input_key(label: str) -> str:
    return _safe_conversation_id(label).lower().replace("-", "_")


def _label_to_title(label: str) -> str:
    stripped = re.sub(r"[_\-]+", " ", label).strip()
    return stripped[:1].upper() + stripped[1:] if stripped else label


def _safe_conversation_id(conversation_id: str) -> str:
    safe_value = re.sub(r"[^A-Za-z0-9_\-]", "_", conversation_id).strip("_")
    if not safe_value:
        raise ValueError("conversation_id does not contain any safe filename characters")
    return safe_value


def _git_revision() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    revision = result.stdout.strip()
    return revision or "unknown"


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _resolve_repo_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    candidate = Path.cwd() / path
    if candidate.exists() or candidate.parent.exists():
        return candidate
    return REPO_ROOT / path
