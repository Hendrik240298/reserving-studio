from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from typing import Any


class SegmentMemoryService:
    SCHEMA_VERSION = 3
    _VALUATION_HISTORY_LIMIT = 12
    _SCENARIO_DISPOSITION_LIMIT = 40
    _RECENT_QUARTER_NOTES_LIMIT = 12
    _MEMORY_CHANGE_LOG_LIMIT = 80

    def load(
        self, raw_memory: dict[str, Any] | None, *, segment: str | None
    ) -> dict[str, Any]:
        normalized = self._base_memory(segment=segment)
        if not isinstance(raw_memory, dict):
            return normalized

        for key, value in raw_memory.items():
            if key == "updated_at":
                continue
            normalized[key] = deepcopy(value)

        normalized["schema_version"] = self.SCHEMA_VERSION
        normalized["segment_id"] = str(segment or normalized.get("segment_id") or "")
        normalized["segment_overview"] = str(
            normalized.get("segment_overview") or ""
        ).strip()
        normalized["house_preferences"] = self._house_preferences_list(
            normalized.get("house_preferences")
        )
        normalized["known_issues"] = self._string_list(normalized.get("known_issues"))
        normalized["open_items"] = self._string_list(normalized.get("open_items"))
        normalized["last_selection"] = self._dict(normalized.get("last_selection"))
        normalized["last_human_decision"] = self._dict(
            normalized.get("last_human_decision")
        )
        normalized["last_recommendation"] = self._dict(
            normalized.get("last_recommendation")
        )
        normalized["last_analysis_basis"] = self._dict(
            normalized.get("last_analysis_basis")
        )
        normalized["last_review"] = self._dict(normalized.get("last_review"))
        normalized["scenario_ledger"] = self._dict_list(
            normalized.get("scenario_ledger")
        )
        normalized["scenario_dispositions"] = self._migrate_scenario_dispositions(
            raw_memory
        )
        normalized["recent_quarter_notes"] = self._trim_dict_list(
            self._recent_quarter_notes_list(normalized.get("recent_quarter_notes")),
            limit=self._RECENT_QUARTER_NOTES_LIMIT,
        )
        normalized["memory_change_log"] = self._trim_dict_list(
            self._memory_change_log_list(normalized.get("memory_change_log")),
            limit=self._MEMORY_CHANGE_LOG_LIMIT,
        )
        normalized["valuation_history"] = self._trim_dict_list(
            normalized.get("valuation_history"),
            limit=self._VALUATION_HISTORY_LIMIT,
        )
        return normalized

    def merge(
        self,
        *,
        existing_memory: dict[str, Any] | None,
        incoming_memory: dict[str, Any] | None,
        segment: str | None,
    ) -> dict[str, Any]:
        current = self.load(existing_memory, segment=segment)
        merged = dict(current)
        raw_incoming = incoming_memory if isinstance(incoming_memory, dict) else {}
        for key, value in raw_incoming.items():
            if key in {"schema_version", "segment_id", "updated_at"}:
                continue
            normalized_piece = self.load({key: value}, segment=segment)
            merged[key] = deepcopy(normalized_piece.get(key))

        merged["schema_version"] = self.SCHEMA_VERSION
        merged["segment_id"] = str(segment or merged.get("segment_id") or "")
        merged["scenario_dispositions"] = self._trim_dict_list(
            merged.get("scenario_dispositions"),
            limit=self._SCENARIO_DISPOSITION_LIMIT,
        )
        merged["valuation_history"] = self._trim_dict_list(
            merged.get("valuation_history"),
            limit=self._VALUATION_HISTORY_LIMIT,
        )
        merged["recent_quarter_notes"] = self._trim_dict_list(
            merged.get("recent_quarter_notes"),
            limit=self._RECENT_QUARTER_NOTES_LIMIT,
        )
        merged["memory_change_log"] = self._trim_dict_list(
            merged.get("memory_change_log"),
            limit=self._MEMORY_CHANGE_LOG_LIMIT,
        )
        return merged

    def append_valuation_snapshot(
        self,
        *,
        memory: dict[str, Any],
        snapshot: dict[str, Any] | None,
    ) -> dict[str, Any]:
        normalized = self.load(memory, segment=memory.get("segment_id"))
        if not isinstance(snapshot, dict) or not snapshot:
            return normalized
        history = self._dict_list(normalized.get("valuation_history"))
        fingerprint = str(snapshot.get("data_fingerprint", "")).strip()
        basis = str(snapshot.get("comparison_basis", "")).strip()
        history = [
            item
            for item in history
            if not (
                str(item.get("data_fingerprint", "")).strip() == fingerprint
                and str(item.get("comparison_basis", "")).strip() == basis
            )
        ]
        history.insert(0, deepcopy(snapshot))
        normalized["valuation_history"] = history[: self._VALUATION_HISTORY_LIMIT]
        return normalized

    def continuity_summary(self, memory: dict[str, Any] | None) -> dict[str, Any]:
        normalized = self.load(
            memory if isinstance(memory, dict) else {},
            segment=(memory or {}).get("segment_id")
            if isinstance(memory, dict)
            else None,
        )
        return {
            "segment_id": normalized.get("segment_id", ""),
            "memory_schema_version": normalized.get("schema_version"),
            "segment_overview": str(normalized.get("segment_overview", "")),
            "known_issues": list(normalized.get("known_issues", [])),
            "house_preferences": list(normalized.get("house_preferences", [])),
            "open_items": list(normalized.get("open_items", [])),
            "recent_quarter_notes": [
                dict(item) for item in normalized.get("recent_quarter_notes", [])
            ],
            "last_selection": dict(normalized.get("last_selection", {})),
            "last_human_decision": dict(normalized.get("last_human_decision", {})),
            "last_recommendation": dict(normalized.get("last_recommendation", {})),
            "last_review": dict(normalized.get("last_review", {})),
            "valuation_history_count": len(normalized.get("valuation_history", [])),
            "recent_rejected_signatures": [
                str(item.get("scenario_signature", "")).strip()
                for item in normalized.get("scenario_dispositions", [])
                if isinstance(item, dict)
                and str(item.get("decision", "")).strip().lower() == "rejected"
                and str(item.get("scenario_signature", "")).strip()
            ][:5],
        }

    def append_scenario_disposition(
        self,
        *,
        memory: dict[str, Any],
        disposition: dict[str, Any] | None,
    ) -> dict[str, Any]:
        normalized = self.load(memory, segment=memory.get("segment_id"))
        if not isinstance(disposition, dict) or not disposition:
            return normalized
        candidate = dict(disposition)
        signature = str(candidate.get("scenario_signature", "")).strip()
        scenario_id = str(candidate.get("scenario_id", "")).strip()
        if not signature and not scenario_id:
            return normalized
        existing = self._dict_list(normalized.get("scenario_dispositions"))
        filtered = []
        for item in existing:
            existing_signature = str(item.get("scenario_signature", "")).strip()
            existing_id = str(item.get("scenario_id", "")).strip()
            if signature and existing_signature == signature:
                continue
            if scenario_id and existing_id == scenario_id:
                continue
            filtered.append(item)
        filtered.insert(0, candidate)
        normalized["scenario_dispositions"] = filtered[
            : self._SCENARIO_DISPOSITION_LIMIT
        ]
        return normalized

    @classmethod
    def scenario_signature(cls, params: dict[str, Any] | None) -> str:
        import hashlib
        import json

        canonical = json.dumps(params or {}, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _base_memory(*, segment: str | None) -> dict[str, Any]:
        return {
            "schema_version": SegmentMemoryService.SCHEMA_VERSION,
            "segment_id": str(segment or ""),
            "segment_overview": "",
            "house_preferences": [],
            "known_issues": [],
            "recent_quarter_notes": [],
            "open_items": [],
            "last_selection": {},
            "last_human_decision": {},
            "last_recommendation": {},
            "last_analysis_basis": {},
            "last_review": {},
            "scenario_ledger": [],
            "scenario_dispositions": [],
            "valuation_history": [],
            "memory_change_log": [],
        }

    def append_memory_change(
        self,
        *,
        memory: dict[str, Any],
        entry: dict[str, Any] | None,
    ) -> dict[str, Any]:
        normalized = self.load(memory, segment=memory.get("segment_id"))
        if not isinstance(entry, dict) or not entry:
            return normalized
        candidate = self._normalize_memory_change_log_entry(entry)
        if not candidate:
            return normalized
        existing = self._dict_list(normalized.get("memory_change_log"))
        normalized["memory_change_log"] = [candidate] + existing[
            : self._MEMORY_CHANGE_LOG_LIMIT - 1
        ]
        return normalized

    def _migrate_scenario_dispositions(
        self,
        raw_memory: dict[str, Any],
    ) -> list[dict[str, Any]]:
        existing = self._dict_list(raw_memory.get("scenario_dispositions"))
        rejected = self._dict_list(raw_memory.get("rejected_scenarios"))
        migrated: list[dict[str, Any]] = [dict(item) for item in existing]
        for item in rejected:
            candidate = {
                "scenario_signature": item.get("scenario_signature")
                or item.get("scenario_hash")
                or item.get("scenario_id"),
                "scenario_id": item.get("scenario_id"),
                "decision": "rejected",
                "reason": item.get("reason", ""),
                "valuation_date": item.get("valuation_date"),
                "data_fingerprint": item.get("data_fingerprint"),
            }
            migrated.append(candidate)
        return self._trim_dict_list(migrated, limit=self._SCENARIO_DISPOSITION_LIMIT)

    @staticmethod
    def _house_preferences_list(value: object) -> list[dict[str, Any] | str]:
        if not isinstance(value, list):
            return []
        normalized: list[dict[str, Any] | str] = []
        for item in value:
            if isinstance(item, dict):
                normalized.append(dict(item))
                continue
            text = str(item).strip()
            if text:
                normalized.append(text)
        return normalized

    def _recent_quarter_notes_list(self, value: object) -> list[dict[str, Any]]:
        items = self._dict_list(value)
        normalized: list[dict[str, Any]] = []
        for item in items:
            period = str(item.get("period", "")).strip()
            note = str(item.get("note", "")).strip()
            if not period and not note:
                continue
            normalized.append(
                {
                    "period": period,
                    "note": note,
                    "source": str(item.get("source", "")).strip() or None,
                    "updated_at": self._normalize_timestamp(item.get("updated_at")),
                }
            )
        return normalized

    def _memory_change_log_list(self, value: object) -> list[dict[str, Any]]:
        items = self._dict_list(value)
        normalized: list[dict[str, Any]] = []
        for item in items:
            candidate = self._normalize_memory_change_log_entry(item)
            if candidate:
                normalized.append(candidate)
        return normalized

    def _normalize_memory_change_log_entry(
        self,
        entry: dict[str, Any],
    ) -> dict[str, Any]:
        field = str(entry.get("field", "")).strip()
        action = str(entry.get("action", "")).strip()
        if not field or not action:
            return {}
        return {
            "field": field,
            "action": action,
            "source": str(entry.get("source", "")).strip() or None,
            "updated_by": str(entry.get("updated_by", "")).strip() or None,
            "approved_by": str(entry.get("approved_by", "")).strip() or None,
            "summary": str(entry.get("summary", "")).strip() or None,
            "updated_at": self._normalize_timestamp(entry.get("updated_at")),
        }

    @staticmethod
    def _string_list(value: object) -> list[str]:
        if not isinstance(value, list):
            return []
        return [str(item) for item in value if str(item).strip()]

    @staticmethod
    def _dict(value: object) -> dict[str, Any]:
        return dict(value) if isinstance(value, dict) else {}

    @staticmethod
    def _dict_list(value: object) -> list[dict[str, Any]]:
        if not isinstance(value, list):
            return []
        return [dict(item) for item in value if isinstance(item, dict)]

    @classmethod
    def _trim_dict_list(cls, value: object, *, limit: int) -> list[dict[str, Any]]:
        items = cls._dict_list(value)
        return items[:limit]

    @staticmethod
    def _normalize_timestamp(value: object) -> str:
        text = str(value or "").strip()
        if text:
            return text
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
