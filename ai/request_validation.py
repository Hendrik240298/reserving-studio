from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RequestValidationResult:
    requested_inputs: dict[str, Any]
    effective_inputs: dict[str, Any]
    execution_status: str
    warnings: list[str] = field(default_factory=list)
    material_adjustments: list[str] = field(default_factory=list)
    rejection_reason: str | None = None

    @property
    def input_adjustments(self) -> list[str]:
        return list(self.warnings)


def build_passthrough_request_validation(
    arguments: dict[str, Any],
) -> RequestValidationResult:
    requested = dict(arguments) if isinstance(arguments, dict) else {}
    return RequestValidationResult(
        requested_inputs=requested,
        effective_inputs=dict(requested),
        execution_status="executed_exactly",
    )


def build_rejected_request_validation(
    arguments: dict[str, Any],
    *,
    reason: str,
) -> RequestValidationResult:
    requested = dict(arguments) if isinstance(arguments, dict) else {}
    message = str(reason).strip() or "Request could not be executed."
    return RequestValidationResult(
        requested_inputs=requested,
        effective_inputs={},
        execution_status="rejected",
        warnings=[message],
        material_adjustments=[message],
        rejection_reason=message,
    )


def validate_recalculate_like_arguments(
    arguments: dict[str, Any],
) -> RequestValidationResult:
    if not isinstance(arguments, dict):
        return build_rejected_request_validation(
            {},
            reason="Request payload must be an object.",
        )

    requested = dict(arguments)
    sanitized = dict(arguments)
    non_material_adjustments: list[str] = []
    material_adjustments: list[str] = []
    partial_execution = False

    average = sanitized.get("average")
    if average is not None:
        normalized_average = _normalize_average_or_volume(average)
        if normalized_average is None:
            return build_rejected_request_validation(
                requested,
                reason=f"Unsupported average assumption: {average}.",
            )
        if normalized_average != str(average).strip().lower():
            non_material_adjustments.append(
                f"Normalized average '{average}' to '{normalized_average}'."
            )
        sanitized["average"] = normalized_average

    tail = sanitized.get("tail")
    if isinstance(tail, dict):
        tail_copy = dict(tail)
        curve = tail_copy.get("curve")
        if curve is not None:
            normalized_curve = _normalize_tail_curve_or_default(curve)
            if normalized_curve is None:
                return build_rejected_request_validation(
                    requested,
                    reason=f"Unsupported tail curve assumption: {curve}.",
                )
            if normalized_curve != str(curve).strip().lower():
                non_material_adjustments.append(
                    f"Normalized tail curve '{curve}' to '{normalized_curve}'."
                )
            tail_copy["curve"] = normalized_curve
        fit_period = tail_copy.get("fit_period")
        if isinstance(fit_period, list) and len(fit_period) > 2:
            try:
                normalized = sorted({int(value) for value in fit_period})
            except (TypeError, ValueError):
                return build_rejected_request_validation(
                    requested,
                    reason="Tail fit period contained unsupported values.",
                )
            tail_copy["fit_period"] = [normalized[0], normalized[-1]]
            material_adjustments.append(
                f"Collapsed tail.fit_period to [{normalized[0]}, {normalized[-1]}]."
            )
        sanitized["tail"] = tail_copy

    selected = sanitized.get("selected_ultimate_by_uwy")
    if isinstance(selected, dict):
        valid_selected: dict[str, str] = {}
        dropped = 0
        for key, value in selected.items():
            normalized_method = _normalize_selected_method(value)
            if normalized_method is None:
                dropped += 1
                continue
            valid_selected[str(key)] = normalized_method
        if dropped:
            material_adjustments.append(
                f"Dropped {dropped} invalid selected_ultimate_by_uwy override(s) and kept only method values."
            )
            partial_execution = True
        sanitized["selected_ultimate_by_uwy"] = valid_selected

    for field_name in ("drop", "drop_valuation"):
        field_value = sanitized.get(field_name)
        if isinstance(field_value, list):
            sanitized_pairs, dropped = _sanitize_drop_like_pairs(field_value)
            if dropped:
                material_adjustments.append(
                    f"Dropped {dropped} invalid {field_name} entr{'y' if dropped == 1 else 'ies'}."
                )
                partial_execution = True
            sanitized[field_name] = sanitized_pairs

    execution_status = _classify_execution_status(
        non_material_adjustments=non_material_adjustments,
        material_adjustments=material_adjustments,
        partial_execution=partial_execution,
    )
    return RequestValidationResult(
        requested_inputs=requested,
        effective_inputs=sanitized,
        execution_status=execution_status,
        warnings=[*non_material_adjustments, *material_adjustments],
        material_adjustments=material_adjustments,
    )


def _classify_execution_status(
    *,
    non_material_adjustments: list[str],
    material_adjustments: list[str],
    partial_execution: bool,
) -> str:
    if material_adjustments:
        if partial_execution:
            return "partially_executed"
        return "executed_with_material_adjustment"
    if non_material_adjustments:
        return "executed_with_non_material_normalization"
    return "executed_exactly"


def _normalize_average_or_volume(value: object) -> str | None:
    normalized = str(value).strip().lower()
    aliases = {
        "volume": "volume",
        "weighted": "volume",
        "weighted_average": "volume",
        "weighted_average_all": "volume",
        "volume_weighted": "volume",
        "volume_weighted_average": "volume",
        "volume_weighted_all": "volume",
        "weighted_average_3_year": "volume",
        "simple": "simple",
        "simple_average": "simple",
        "arithmetic": "simple",
    }
    return aliases.get(normalized)


def _normalize_tail_curve_or_default(value: object) -> str | None:
    normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "exponential": "exponential",
        "exp": "exponential",
        "inverse_power": "inverse_power",
        "inversepower": "inverse_power",
        "power": "inverse_power",
        "power_curve": "inverse_power",
        "powercurve": "inverse_power",
        "inverse_power_curve": "inverse_power",
        "weibull": "weibull",
    }
    return aliases.get(normalized)


def _normalize_selected_method(value: object) -> str | None:
    normalized = str(value).strip().lower()
    if normalized in {"chainladder", "bornhuetter_ferguson"}:
        return normalized
    return None


def _sanitize_drop_like_pairs(value: list[Any]) -> tuple[list[list[str | int]], int]:
    valid_pairs: list[list[str | int]] = []
    dropped = 0
    for pair in value:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            dropped += 1
            continue
        origin, development = pair
        if origin is None or development is None:
            dropped += 1
            continue
        if not isinstance(development, int) or isinstance(development, bool):
            dropped += 1
            continue
        valid_pairs.append([str(origin), development])
    return valid_pairs, dropped
