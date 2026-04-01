from __future__ import annotations

from pathlib import Path


def load_segment_note(segment: str | None) -> str:
    if not isinstance(segment, str) or not segment.strip():
        return ""
    note_path = (
        Path(__file__).resolve().parents[1]
        / "AI_SEGMENT_NOTES"
        / f"{segment.strip()}.md"
    )
    if not note_path.exists():
        return ""
    try:
        return note_path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""
