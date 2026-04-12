from __future__ import annotations

from datetime import datetime, timezone
import uuid
from typing import Any

from source.config_manager import ConfigManager
from source.services.segment_memory_service import SegmentMemoryService


class MemoryAuthoringService:
    _RECENT_QUARTER_CONTEXT_LIMIT = 4
    _SUPPORTED_PROPOSAL_FIELDS = {
        "segment_overview",
        "known_issues",
        "recent_quarter_notes",
        "open_items",
    }

    def __init__(
        self,
        *,
        memory_service: SegmentMemoryService | None = None,
    ) -> None:
        self._memory_service = memory_service or SegmentMemoryService()

    def load_for_segment(
        self,
        *,
        config: ConfigManager | None,
        segment: str | None,
    ) -> dict[str, Any]:
        return self._load_memory(config=config, segment=segment)

    def save_manual_update(
        self,
        *,
        config: ConfigManager | None,
        segment: str | None,
        fields: dict[str, Any] | None,
        editor: str | None,
    ) -> dict[str, Any]:
        current = self._load_memory(config=config, segment=segment)
        updates = self._normalize_ui_fields(fields, existing_memory=current)
        merged = self._memory_service.merge(
            existing_memory=current,
            incoming_memory=updates,
            segment=segment,
        )
        merged = self._memory_service.append_memory_change(
            memory=merged,
            entry={
                "field": "manual_edit",
                "action": "save_manual_update",
                "source": "human",
                "updated_by": str(editor or "").strip() or "user",
                "summary": self._manual_update_summary(updates),
                "updated_at": self._utc_now(),
            },
        )
        self._save_memory(config=config, segment=segment, memory=merged)
        return merged

    def build_context_packet(self, memory: dict[str, Any] | None) -> dict[str, Any]:
        normalized = self._memory_service.load(
            memory if isinstance(memory, dict) else {},
            segment=(memory or {}).get("segment_id")
            if isinstance(memory, dict)
            else None,
        )
        return {
            "segment_overview": str(normalized.get("segment_overview", "")).strip(),
            "known_issues": list(normalized.get("known_issues", [])),
            "house_preferences": list(normalized.get("house_preferences", [])),
            "open_items": list(normalized.get("open_items", [])),
            "recent_quarter_notes": [
                dict(item)
                for item in normalized.get("recent_quarter_notes", [])[
                    : self._RECENT_QUARTER_CONTEXT_LIMIT
                ]
            ],
        }

    def build_ui_payload(self, memory: dict[str, Any] | None) -> dict[str, Any]:
        normalized = self._memory_service.load(
            memory if isinstance(memory, dict) else {},
            segment=(memory or {}).get("segment_id")
            if isinstance(memory, dict)
            else None,
        )
        text_preferences = [
            item
            for item in normalized.get("house_preferences", [])
            if isinstance(item, str) and str(item).strip()
        ]
        structured_preferences = [
            dict(item)
            for item in normalized.get("house_preferences", [])
            if isinstance(item, dict)
        ]
        return {
            "segment_overview": str(normalized.get("segment_overview", "")).strip(),
            "known_issues_text": self._join_lines(normalized.get("known_issues", [])),
            "house_preferences_text": self._join_lines(text_preferences),
            "structured_house_preferences": structured_preferences,
            "recent_quarter_notes_text": self._format_recent_quarter_notes(
                normalized.get("recent_quarter_notes", [])
            ),
            "open_items_text": self._join_lines(normalized.get("open_items", [])),
            "memory_change_log": [
                dict(item) for item in normalized.get("memory_change_log", [])[:12]
            ],
            "segment_id": str(normalized.get("segment_id", "")).strip(),
        }

    def apply_memory_proposal(
        self,
        *,
        config: ConfigManager | None,
        segment: str | None,
        proposal: dict[str, Any] | None,
        approver: str | None,
        edited_value: Any | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        current = self._load_memory(config=config, segment=segment)
        normalized_proposal = self.normalize_proposal(proposal)
        if not normalized_proposal:
            return current, {}
        candidate = dict(normalized_proposal)
        if edited_value is not None:
            candidate["value"] = edited_value
        updates = self._proposal_updates(candidate, existing_memory=current)
        merged = self._memory_service.merge(
            existing_memory=current,
            incoming_memory=updates,
            segment=segment,
        )
        merged = self._memory_service.append_memory_change(
            memory=merged,
            entry={
                "field": candidate.get("field"),
                "action": "approve_proposal",
                "source": candidate.get("source") or "assistant",
                "approved_by": str(approver or "").strip() or "user",
                "summary": candidate.get("rationale")
                or f"Approved memory proposal for {candidate.get('field')}",
                "updated_at": self._utc_now(),
            },
        )
        self._save_memory(config=config, segment=segment, memory=merged)
        candidate["status"] = "accepted"
        candidate["approved_by"] = str(approver or "").strip() or "user"
        return merged, candidate

    def reject_memory_proposal(
        self,
        proposal: dict[str, Any] | None,
        *,
        reviewer: str | None,
    ) -> dict[str, Any]:
        normalized = self.normalize_proposal(proposal)
        if not normalized:
            return {}
        rejected = dict(normalized)
        rejected["status"] = "rejected"
        rejected["reviewed_by"] = str(reviewer or "").strip() or "user"
        return rejected

    def normalize_proposal(self, proposal: dict[str, Any] | None) -> dict[str, Any]:
        if not isinstance(proposal, dict):
            return {}
        field = str(proposal.get("field", "")).strip()
        operation = str(proposal.get("operation", "")).strip().lower() or "append"
        if field not in self._SUPPORTED_PROPOSAL_FIELDS:
            return {}
        if operation not in {"append", "replace"}:
            return {}
        value = proposal.get("value")
        if field in {"known_issues", "open_items"}:
            items = self._list_value(value)
            if not items:
                return {}
            normalized_value: Any = items
        elif field == "segment_overview":
            text = str(value or "").strip()
            if not text:
                return {}
            normalized_value = text
        else:
            notes = self._notes_value(value)
            if not notes:
                return {}
            normalized_value = notes
        return {
            "proposal_id": str(proposal.get("proposal_id") or self._proposal_id()),
            "field": field,
            "operation": operation,
            "value": normalized_value,
            "rationale": str(proposal.get("rationale", "")).strip(),
            "source": str(proposal.get("source", "assistant")).strip() or "assistant",
            "evidence_ids": [
                str(item).strip()
                for item in proposal.get("evidence_ids", [])
                if str(item).strip()
            ]
            if isinstance(proposal.get("evidence_ids"), list)
            else [],
            "status": str(proposal.get("status", "pending")).strip() or "pending",
        }

    def render_context_text(self, memory: dict[str, Any] | None) -> str:
        packet = self.build_context_packet(memory)
        parts: list[str] = []
        if packet.get("segment_overview"):
            parts.append(f"Segment overview: {packet['segment_overview']}")
        if packet.get("known_issues"):
            parts.append(
                "Known issues: "
                + "; ".join(str(item) for item in packet["known_issues"])
            )
        if packet.get("house_preferences"):
            rendered_preferences = []
            for item in packet["house_preferences"]:
                if isinstance(item, dict):
                    pref_type = str(item.get("type", "")).strip()
                    pref_value = item.get("value")
                    rendered_preferences.append(f"{pref_type}={pref_value}")
                else:
                    rendered_preferences.append(str(item))
            parts.append("House preferences: " + "; ".join(rendered_preferences))
        if packet.get("open_items"):
            parts.append(
                "Open items: " + "; ".join(str(item) for item in packet["open_items"])
            )
        if packet.get("recent_quarter_notes"):
            parts.append(
                "Recent quarter notes: "
                + "; ".join(
                    f"{item.get('period')}: {item.get('note')}"
                    for item in packet["recent_quarter_notes"]
                    if isinstance(item, dict)
                )
            )
        return "\n".join(part for part in parts if part)

    def _normalize_ui_fields(
        self,
        fields: dict[str, Any] | None,
        *,
        existing_memory: dict[str, Any],
    ) -> dict[str, Any]:
        values = dict(fields or {})
        structured_preferences = [
            dict(item)
            for item in existing_memory.get("house_preferences", [])
            if isinstance(item, dict)
        ]
        return {
            "segment_overview": str(values.get("segment_overview", "")).strip(),
            "known_issues": self._split_lines(values.get("known_issues_text")),
            "house_preferences": structured_preferences
            + self._split_lines(values.get("house_preferences_text")),
            "recent_quarter_notes": self._parse_recent_quarter_notes(
                values.get("recent_quarter_notes_text")
            ),
            "open_items": self._split_lines(values.get("open_items_text")),
        }

    def _proposal_updates(
        self,
        proposal: dict[str, Any],
        *,
        existing_memory: dict[str, Any],
    ) -> dict[str, Any]:
        field = str(proposal.get("field", "")).strip()
        operation = str(proposal.get("operation", "append")).strip().lower()
        value = proposal.get("value")
        if field == "segment_overview":
            return {"segment_overview": str(value or "").strip()}
        if field in {"known_issues", "open_items"}:
            items = self._list_value(value)
            if operation == "append":
                items = list(existing_memory.get(field, [])) + items
            return {field: items}
        if field == "recent_quarter_notes":
            notes = self._notes_value(value)
            if operation == "append":
                notes = list(existing_memory.get(field, [])) + notes
            return {field: notes}
        return {}

    def propose_updates_for_turn(
        self,
        *,
        user_prompt: str,
        deterministic_packet: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        prompt = str(user_prompt or "").strip().lower()
        if not any(
            token in prompt
            for token in (
                "remember",
                "carry forward",
                "add to memory",
                "save this note",
                "save this in memory",
                "note this",
                "update memory",
            )
        ):
            return []
        packet = deterministic_packet if isinstance(deterministic_packet, dict) else {}
        presentation = (
            packet.get("presentation")
            if isinstance(packet.get("presentation"), dict)
            else {}
        )
        key_caveat = str(presentation.get("key_caveat", "")).strip()
        if not key_caveat:
            return []
        return [
            {
                "proposal_id": self._proposal_id(),
                "field": "open_items",
                "operation": "append",
                "value": [key_caveat],
                "rationale": "The latest AI review identified a carry-forward item worth preserving in segment memory.",
                "source": "assistant",
                "evidence_ids": [
                    str(item).strip()
                    for item in presentation.get("evidence_used", [])
                    if str(item).strip()
                ]
                if isinstance(presentation.get("evidence_used"), list)
                else [],
                "status": "pending",
            }
        ]

    @staticmethod
    def _format_recent_quarter_notes(notes: list[dict[str, Any]]) -> str:
        rows: list[str] = []
        for item in notes:
            if not isinstance(item, dict):
                continue
            period = str(item.get("period", "")).strip()
            note = str(item.get("note", "")).strip()
            if not period and not note:
                continue
            rows.append(f"{period} | {note}" if period else note)
        return "\n".join(rows)

    @staticmethod
    def _parse_recent_quarter_notes(value: Any) -> list[dict[str, Any]]:
        notes: list[dict[str, Any]] = []
        for raw_line in str(value or "").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if "|" in line:
                period, note = line.split("|", 1)
                notes.append(
                    {
                        "period": str(period).strip(),
                        "note": str(note).strip(),
                        "source": "human",
                        "updated_at": MemoryAuthoringService._utc_now(),
                    }
                )
                continue
            notes.append(
                {
                    "period": "",
                    "note": line,
                    "source": "human",
                    "updated_at": MemoryAuthoringService._utc_now(),
                }
            )
        return notes

    @staticmethod
    def _manual_update_summary(updates: dict[str, Any]) -> str:
        populated = [key for key, value in updates.items() if value]
        return (
            "Updated memory fields: " + ", ".join(populated)
            if populated
            else "Manual memory save"
        )

    @staticmethod
    def _join_lines(values: list[Any]) -> str:
        return "\n".join(str(item).strip() for item in values if str(item).strip())

    @staticmethod
    def _split_lines(value: Any) -> list[str]:
        return [line.strip() for line in str(value or "").splitlines() if line.strip()]

    @staticmethod
    def _list_value(value: Any) -> list[str]:
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        text = str(value or "").strip()
        return [text] if text else []

    @staticmethod
    def _notes_value(value: Any) -> list[dict[str, Any]]:
        if not isinstance(value, list):
            return []
        notes: list[dict[str, Any]] = []
        for item in value:
            if not isinstance(item, dict):
                continue
            period = str(item.get("period", "")).strip()
            note = str(item.get("note", "")).strip()
            if not period and not note:
                continue
            notes.append(
                {
                    "period": period,
                    "note": note,
                    "source": str(item.get("source", "assistant")).strip()
                    or "assistant",
                    "updated_at": str(
                        item.get("updated_at") or MemoryAuthoringService._utc_now()
                    ),
                }
            )
        return notes

    def _load_memory(
        self,
        *,
        config: ConfigManager | None,
        segment: str | None,
    ) -> dict[str, Any]:
        if config is None:
            return self._memory_service.load({}, segment=segment)
        raw_memory = config.load_ai_segment_memory(segment=segment)
        return self._memory_service.load(raw_memory, segment=segment)

    @staticmethod
    def _save_memory(
        *,
        config: ConfigManager | None,
        segment: str | None,
        memory: dict[str, Any],
    ) -> None:
        if config is None:
            return
        config.save_ai_segment_memory(memory, segment=segment)

    @staticmethod
    def _proposal_id() -> str:
        return f"mem-{uuid.uuid4().hex[:12]}"

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
