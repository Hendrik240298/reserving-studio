from __future__ import annotations

from typing import Any

from ai.control_plane_types import (
    AcceptedAnalysisBasis,
    normalize_accepted_analysis_basis,
)
from ai.tool_payloads import build_baseline_analysis_basis


class BasisManager:
    @staticmethod
    def accepted_basis(
        *,
        accepted_analysis_basis: dict[str, Any] | None = None,
        legacy_working_memory: dict[str, Any] | None = None,
    ) -> AcceptedAnalysisBasis:
        normalized = normalize_accepted_analysis_basis(accepted_analysis_basis)
        if normalized:
            return normalized
        if not isinstance(legacy_working_memory, dict):
            return {}
        normalized = normalize_accepted_analysis_basis(
            legacy_working_memory.get("accepted_analysis_basis")
        )
        if normalized:
            return normalized
        return normalize_accepted_analysis_basis(legacy_working_memory.get("analysis_basis"))

    @staticmethod
    def baseline_basis_from_memory(
        *,
        memory_state: dict[str, Any],
        session_context: dict[str, Any] | None,
    ) -> AcceptedAnalysisBasis:
        session_summary = (
            memory_state.get("session_summary")
            if isinstance(memory_state.get("session_summary"), dict)
            else {}
        )
        baseline_basis = build_baseline_analysis_basis(session_summary)
        normalized = normalize_accepted_analysis_basis(baseline_basis)
        if normalized:
            return normalized
        session_id = None
        if isinstance(session_context, dict):
            raw_session_id = session_context.get("session_id")
            if isinstance(raw_session_id, str) and raw_session_id.strip():
                session_id = raw_session_id.strip()
        return normalize_accepted_analysis_basis(
            {
                "basis_type": "baseline",
                "session_id": session_id or "",
                "scenario_id": "baseline",
                "is_active_session": True,
                "parameters": {},
            }
        )

    @staticmethod
    def label(basis: dict[str, Any] | None) -> str:
        normalized = normalize_accepted_analysis_basis(basis)
        if not normalized:
            return "Basis used: current baseline session."
        scenario_label = str(normalized.get("scenario_label") or "").strip()
        scenario_id = str(normalized.get("scenario_id") or "").strip()
        basis_type = str(normalized.get("basis_type") or "").strip().lower()
        if scenario_label and scenario_id != "baseline":
            return f"Basis used: scenario {scenario_label}."
        if basis_type == "bespoke":
            return "Basis used: custom conversation basis."
        return "Basis used: current baseline session."

    @staticmethod
    def identifier_aliases(basis: dict[str, Any] | None) -> list[str]:
        normalized = normalize_accepted_analysis_basis(basis)
        if not normalized:
            return []
        aliases: list[str] = []
        for value in (
            normalized.get("basis_key"),
            normalized.get("scenario_id"),
            normalized.get("candidate_id"),
            normalized.get("scenario_label"),
        ):
            text = str(value or "").strip()
            if text and text not in aliases:
                aliases.append(text)
        return aliases

    @staticmethod
    def canonical_basis_key(basis: dict[str, Any] | None) -> str:
        normalized = normalize_accepted_analysis_basis(basis)
        if not normalized:
            return ""
        for value in (
            normalized.get("basis_key"),
            normalized.get("scenario_signature"),
            normalized.get("scenario_id"),
        ):
            text = str(value or "").strip()
            if text:
                return text
        return ""

    @staticmethod
    def canonicalize_basis_cache(
        basis_cache: dict[str, Any] | None,
    ) -> dict[str, Any]:
        canonical: dict[str, Any] = {}
        if not isinstance(basis_cache, dict):
            return canonical
        for value in basis_cache.values():
            normalized = normalize_accepted_analysis_basis(value)
            key = BasisManager.canonical_basis_key(normalized)
            if key and normalized:
                canonical[key] = normalized
        return canonical

    @staticmethod
    def lookup_basis_by_requested_id(
        *,
        requested_id: str,
        current_basis: dict[str, Any],
        basis_cache: dict[str, Any],
    ) -> AcceptedAnalysisBasis:
        target = str(requested_id or "").strip()
        if not target or target == "baseline":
            return {}
        normalized_current = normalize_accepted_analysis_basis(current_basis)
        if target == BasisManager.canonical_basis_key(normalized_current):
            return normalized_current
        if target in BasisManager.identifier_aliases(normalized_current):
            return normalized_current
        canonical_cache = BasisManager.canonicalize_basis_cache(basis_cache)
        cached = canonical_cache.get(target)
        normalized_cached = normalize_accepted_analysis_basis(cached)
        if normalized_cached:
            return normalized_cached
        for cache_key, item in canonical_cache.items():
            normalized_item = normalize_accepted_analysis_basis(item)
            aliases = BasisManager.identifier_aliases(normalized_item)
            if target == str(cache_key).strip() or target in aliases:
                return normalized_item
        return {}

    @staticmethod
    def basis_is_mentioned_in_prompt(*, prompt: str, basis: dict[str, Any]) -> bool:
        prompt_text = str(prompt or "").strip().lower()
        if not prompt_text:
            return False
        return any(alias.lower() in prompt_text for alias in BasisManager.identifier_aliases(basis))

    @staticmethod
    def scenario_id_mentioned_in_prompt(
        *,
        prompt: str,
        basis_cache: dict[str, Any],
    ) -> str | None:
        prompt_text = str(prompt or "").strip().lower()
        if not prompt_text:
            return None
        stable_aliases: list[tuple[str, str]] = []
        candidate_aliases: dict[str, set[str]] = {}
        scenario_label_aliases: dict[str, set[str]] = {}
        canonical_cache = BasisManager.canonicalize_basis_cache(basis_cache)
        for cache_key, cached in canonical_cache.items():
            normalized = normalize_accepted_analysis_basis(cached)
            key = str(cache_key).strip()
            if not key or key == "baseline":
                continue
            aliases = {key}
            scenario_id = str(normalized.get("scenario_id") or "").strip()
            basis_key = str(normalized.get("basis_key") or "").strip()
            if scenario_id and scenario_id != "baseline":
                aliases.add(scenario_id)
            if basis_key and basis_key != "baseline":
                aliases.add(basis_key)
            candidate_id = str(normalized.get("candidate_id") or "").strip().lower()
            if candidate_id and candidate_id != "baseline":
                candidate_aliases.setdefault(candidate_id, set()).add(key)
            scenario_label = str(normalized.get("scenario_label") or "").strip().lower()
            if scenario_label and scenario_label != "baseline":
                scenario_label_aliases.setdefault(scenario_label, set()).add(key)
            for alias in aliases:
                stable_aliases.append((alias.lower(), key))
        for alias, key in sorted(stable_aliases, key=lambda item: len(item[0]), reverse=True):
            if alias and alias in prompt_text:
                return key
        for alias, keys in sorted(
            scenario_label_aliases.items(), key=lambda item: len(item[0]), reverse=True
        ):
            if len(keys) == 1 and alias in prompt_text:
                return next(iter(keys))
        for alias, keys in sorted(candidate_aliases.items(), key=lambda item: len(item[0]), reverse=True):
            if len(keys) == 1 and alias in prompt_text:
                return next(iter(keys))
        return None

    @staticmethod
    def tool_basis_args(basis: dict[str, Any] | None) -> dict[str, Any]:
        normalized = normalize_accepted_analysis_basis(basis)
        if not normalized:
            return {}
        args: dict[str, Any] = {}
        basis_type = str(normalized.get("basis_type") or "").strip()
        scenario_id = str(normalized.get("scenario_id") or "").strip()
        parameters = normalized.get("parameters")
        if basis_type:
            args["basis_type"] = basis_type
        if scenario_id:
            args["scenario_id"] = scenario_id
        if isinstance(parameters, dict) and parameters:
            args["parameters"] = dict(parameters)
        return args
