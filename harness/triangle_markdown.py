from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from source.claims_collection import ClaimsCollection
from source.config_manager import ConfigManager
from source.input_loader import load_inputs_from_config
from source.premium_repository import PremiumRepository
from source.triangle import Triangle


REPO_ROOT = Path(__file__).resolve().parents[1]

SUPPORTED_TRIANGLE_ALIASES = {
    "a2a": "a2a",
    "incurred": "incurred",
    "outstanding": "outstanding",
    "paid": "paid",
    "premium": "Premium_selected",
    "premium_selected": "Premium_selected",
}


@dataclass(frozen=True)
class TriangleMarkdownResult:
    markdown: str
    triangle_type: str
    triangle_view: str
    bolted_drops: list[tuple[str, int]]
    output_path: Path | None = None


def render_triangle_markdown(
    *,
    config_path: Path,
    triangle_type: str,
    triangle_view: str = "cumulative",
    bolted_drops: list[tuple[str, int]] | None = None,
    output_path: Path | None = None,
) -> TriangleMarkdownResult:
    resolved_config_path = _resolve_repo_path(config_path)
    config = ConfigManager.from_yaml(resolved_config_path)
    claims_df, premium_df = load_inputs_from_config(config, repo_root=REPO_ROOT)

    normalized_triangle_type = _normalize_triangle_type(triangle_type)
    normalized_triangle_view = _normalize_triangle_view(triangle_view)
    triangle = _build_triangle(claims_df, premium_df, config=config)
    bolted_drop_set = {
        (str(origin), int(development_age))
        for origin, development_age in (bolted_drops or [])
    }

    if normalized_triangle_type == "a2a":
        markdown = _render_a2a_triangle(
            triangle,
            triangle_view=normalized_triangle_view,
            bolted_drop_set=bolted_drop_set,
        )
    else:
        markdown = _render_value_triangle(
            triangle,
            triangle_type=normalized_triangle_type,
            triangle_view=normalized_triangle_view,
            bolted_drop_set=bolted_drop_set,
        )

    resolved_output_path = _resolve_repo_path(output_path) if output_path else None
    if resolved_output_path is not None:
        resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
        resolved_output_path.write_text(markdown, encoding="utf-8")

    return TriangleMarkdownResult(
        markdown=markdown,
        triangle_type=normalized_triangle_type,
        triangle_view=normalized_triangle_view,
        bolted_drops=sorted(bolted_drop_set),
        output_path=resolved_output_path,
    )


def _build_triangle(
    claims_df: pd.DataFrame,
    premium_df: pd.DataFrame,
    *,
    config: ConfigManager | None,
) -> Triangle:
    claims = ClaimsCollection(
        claims_df,
        values_are_cumulative=bool(claims_df.attrs.get("values_are_cumulative", False)),
    )
    premium = PremiumRepository.from_dataframe(
        config_manager=config,
        dataframe=premium_df,
    )
    return Triangle.from_claims(claims, premium)


def _render_value_triangle(
    triangle: Triangle,
    *,
    triangle_type: str,
    triangle_view: str,
    bolted_drop_set: set[tuple[str, int]],
) -> str:
    tri_obj = triangle.get_triangle(triangle_type)
    if triangle_view == "incremental":
        tri_obj = tri_obj.cum_to_incr()
    values = tri_obj[triangle_type].values[0, 0, :, :]
    origins = [_format_origin_label(origin) for origin in tri_obj.origin.tolist()]
    development_values = [_normalize_development_key(dev) for dev in tri_obj.development.tolist()]
    development_labels = [
        _format_development_label(dev) for dev in development_values
    ]
    return _render_table(
        origins,
        development_labels,
        development_values,
        values,
        bolted_drop_set,
    )


def _render_a2a_triangle(
    triangle: Triangle,
    *,
    triangle_view: str,
    bolted_drop_set: set[tuple[str, int]],
) -> str:
    if triangle_view != "cumulative":
        raise ValueError(
            "triangle_view 'incremental' is not supported for triangle type 'a2a'"
        )
    tri_obj = triangle.get_triangle("incurred")
    model = triangle._get_model(tri_obj)
    dev = model.transform(tri_obj)
    link_ratios = dev.link_ratio.values[0, 0, :, :]
    origins = [_format_origin_label(origin) for origin in tri_obj.origin.tolist()]
    development_values = [
        _normalize_development_key(development_age)
        for development_age in tri_obj.development.tolist()[: link_ratios.shape[1]]
    ]
    development_labels = [
        _format_development_label(development_age)
        for development_age in development_values
    ]
    return _render_table(
        origins,
        development_labels,
        development_values,
        link_ratios,
        bolted_drop_set,
    )


def _render_table(
    origins: list[str],
    development_labels: list[str],
    development_values: list[int],
    values: np.ndarray,
    bolted_drop_set: set[tuple[str, int]],
) -> str:
    n_origins, n_devs = values.shape
    origins = origins[:n_origins]
    development_labels = development_labels[:n_devs]
    development_values = development_values[:n_devs]
    header = ["Origin", *development_labels]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for row_index, origin in enumerate(origins):
        row = [origin]
        for col_index, development_value in enumerate(development_values):
            value = values[row_index, col_index]
            if pd.isna(value):
                row.append("")
                continue
            cell = f"{value:.3f}"
            if (origin, development_value) in bolted_drop_set:
                cell = f"~~{cell}~~"
            row.append(cell)
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _format_origin_label(origin: Any) -> str:
    if hasattr(origin, "year"):
        return str(origin.year)
    return str(origin)


def _format_development_label(development_value: Any) -> str:
    return str(_normalize_development_key(development_value))


def _normalize_triangle_type(triangle_type: str) -> str:
    normalized = triangle_type.strip().lower()
    if normalized not in SUPPORTED_TRIANGLE_ALIASES:
        supported = ", ".join(sorted(SUPPORTED_TRIANGLE_ALIASES))
        raise ValueError(
            f"Unsupported triangle type '{triangle_type}'. Supported types: {supported}"
        )
    return SUPPORTED_TRIANGLE_ALIASES[normalized]


def _normalize_triangle_view(triangle_view: str) -> str:
    normalized = triangle_view.strip().lower()
    if normalized not in {"cumulative", "incremental"}:
        raise ValueError(
            f"Unsupported triangle view '{triangle_view}'. Supported views: cumulative, incremental"
        )
    return normalized


def _normalize_development_key(value: Any) -> int:
    if isinstance(value, (int, np.integer)):
        return int(value)
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Development value '{value}' is not an integer") from exc


def _resolve_repo_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    candidate = Path.cwd() / path
    if candidate.exists() or candidate.parent.exists():
        return candidate
    return REPO_ROOT / path
