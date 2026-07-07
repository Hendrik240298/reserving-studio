from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any
import warnings

import pandas as pd
import yaml

from source.claims_collection import ClaimsCollection
from source.config_manager import ConfigManager
from source.input_loader import load_inputs_from_config
from source.premium_repository import PremiumRepository
from source.triangle import Triangle
from source.reserving import Reserving


REPO_ROOT = Path(__file__).parents[1]
DEFAULT_CONFIG_PATH = Path("examples/config_quarterly.yml")
DEFAULT_ARTIFACT_DIR = Path("harness/artifacts")


@dataclass(frozen=True)
class ScenarioRun:
    name: str
    settings: dict[str, Any]
    effective_settings: dict[str, Any]
    ldf: pd.Series
    warnings: list[str]


def run_ldf_compare(
    *,
    config_path: Path = DEFAULT_CONFIG_PATH,
    scenarios_file: Path | None = None,
    scenario_json: list[str] | None = None,
    delta_threshold: float = 0.01,
    output_path: Path | None = None,
    command: str = "not recorded",
) -> Path:
    scenarios = load_scenarios(
        scenarios_file=scenarios_file,
        scenario_json=scenario_json or [],
    )
    if not scenarios:
        raise ValueError("Provide at least one scenario via --scenarios-file or --scenario-json")

    triangle = load_triangle(config_path)
    baseline = run_scenario(triangle, name="default", settings={})
    scenario_runs = [
        run_scenario(triangle, name=name, settings=settings)
        for name, settings in scenarios.items()
    ]
    ldf_table = build_ldf_table(baseline, scenario_runs)

    output_path = output_path or default_output_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_path = output_path.with_name(f"{output_path.stem}_lines.png")
    write_line_plot(
        ldf_table,
        scenario_names=[run.name for run in scenario_runs],
        delta_threshold=delta_threshold,
        output_path=plot_path,
    )
    output_path.write_text(
        render_markdown_packet(
            config_path=config_path,
            command=command,
            baseline=baseline,
            scenario_runs=scenario_runs,
            ldf_table=ldf_table,
            plot_path=plot_path,
            delta_threshold=delta_threshold,
        ),
        encoding="utf-8",
    )
    return output_path


def load_triangle(config_path: Path) -> Triangle:
    config = ConfigManager.from_yaml(file_name=config_path)
    claims_df, premium_df = load_inputs_from_config(config, repo_root=REPO_ROOT)
    claims = ClaimsCollection(
        claims_df,
        values_are_cumulative=bool(claims_df.attrs.get("values_are_cumulative", False)),
    )
    premium = PremiumRepository.from_dataframe(
        config_manager=config,
        dataframe=premium_df,
    )
    return Triangle.from_claims(claims=claims, premium=premium)


def load_scenarios(
    *,
    scenarios_file: Path | None,
    scenario_json: list[str],
) -> dict[str, dict[str, Any]]:
    scenarios: dict[str, dict[str, Any]] = {}

    if scenarios_file is not None:
        payload = yaml.safe_load(scenarios_file.read_text(encoding="utf-8")) or {}
        if not isinstance(payload, dict):
            raise ValueError("scenarios file must contain a mapping")
        file_scenarios = payload.get("scenarios", payload)
        if not isinstance(file_scenarios, dict):
            raise ValueError("scenarios file must contain a 'scenarios' mapping")
        for name, settings in file_scenarios.items():
            add_scenario(scenarios, str(name), settings)

    for value in scenario_json:
        if ":" not in value:
            raise ValueError(f"Invalid --scenario-json value '{value}'. Expected NAME:JSON")
        name, raw_json = value.split(":", 1)
        try:
            settings = json.loads(raw_json)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON for scenario '{name}': {exc}") from exc
        add_scenario(scenarios, name, settings)

    return scenarios


def add_scenario(
    scenarios: dict[str, dict[str, Any]], name: str, settings: Any
) -> None:
    name = name.strip()
    if not name:
        raise ValueError("scenario name must not be empty")
    if name == "default":
        raise ValueError("scenario name 'default' is reserved for the baseline")
    if name in scenarios:
        raise ValueError(f"duplicate scenario name '{name}'")
    if settings is None:
        settings = {}
    if not isinstance(settings, dict):
        raise ValueError(f"scenario '{name}' must be a mapping")
    scenarios[name] = normalize_settings(settings)


def normalize_settings(settings: dict[str, Any]) -> dict[str, Any]:
    allowed_sections = {"development", "tail", "bornhuetter", "reserve"}
    unknown_sections = sorted(set(settings) - allowed_sections)
    if unknown_sections:
        raise ValueError(f"Unknown scenario section(s): {', '.join(unknown_sections)}")

    normalized = {section: dict(settings.get(section) or {}) for section in allowed_sections}
    development = normalized["development"]
    tail = normalized["tail"]

    if "drop" in development:
        development["drop"] = normalize_drop(development["drop"])
    if "fit_period" in tail:
        tail["fit_period"] = normalize_fit_period(tail["fit_period"])
    return normalized


def normalize_drop(value: Any) -> list[tuple[str, int]] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError("development.drop must be a list of [origin, age] pairs")

    drops: list[tuple[str, int]] = []
    for index, item in enumerate(value):
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(
                f"development.drop[{index}] must be a two-item [origin, age] pair"
            )
        origin, age = item
        drops.append((str(origin), int(age)))
    return drops


def normalize_fit_period(value: Any) -> tuple[int, int | None] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("tail.fit_period must be a two-item [start, end] pair")
    start, end = value
    return int(start), None if end is None else int(end)


def run_scenario(triangle: Triangle, *, name: str, settings: dict[str, Any]) -> ScenarioRun:
    development = settings.get("development", {})
    tail = settings.get("tail", {})
    bornhuetter = settings.get("bornhuetter", {})
    reserve = settings.get("reserve", {})

    effective_settings = {
        "development": {
            "average": development.get("average", "volume"),
            "drop": [list(item) for item in development.get("drop") or []],
        },
        "tail": {"curve": tail.get("curve", "weibull")},
        "bornhuetter": {"apriori": bornhuetter.get("apriori", 0.6)},
        "reserve": {
            "final_ultimate": reserve.get("final_ultimate", "chainladder"),
            "enforce_monotone_tail": reserve.get("enforce_monotone_tail", True),
        },
    }
    for key in ["attachment_age", "extrap_periods", "projection_period", "fit_period"]:
        if key in tail:
            effective_settings["tail"][key] = tail[key]
    if "drop_valuation" in development:
        effective_settings["development"]["drop_valuation"] = development["drop_valuation"]
    if "selected_ultimate_by_uwy" in reserve:
        effective_settings["reserve"]["selected_ultimate_by_uwy"] = reserve[
            "selected_ultimate_by_uwy"
        ]

    reserving = Reserving(triangle=triangle)
    with warnings.catch_warnings(record=True) as captured_warnings:
        warnings.simplefilter("always")
        reserving.set_development(
            average=development.get("average", "volume"),
            drop=development.get("drop"),
            drop_valuation=development.get("drop_valuation"),
        )
        reserving.set_tail(
            curve=tail.get("curve", "weibull"),
            attachment_age=tail.get("attachment_age"),
            extrap_periods=tail.get("extrap_periods"),
            projection_period=tail.get("projection_period"),
            fit_period=tail.get("fit_period"),
        )
        reserving.set_bornhuetter_ferguson(apriori=bornhuetter.get("apriori", 0.6))
        reserving.reserve(
            final_ultimate=reserve.get("final_ultimate", "chainladder"),
            selected_ultimate_by_uwy=reserve.get("selected_ultimate_by_uwy"),
            enforce_monotone_tail=reserve.get("enforce_monotone_tail", True),
        )

    return ScenarioRun(
        name=name,
        settings=settings,
        effective_settings=effective_settings,
        ldf=reserving.get_ldf(),
        warnings=[str(item.message) for item in captured_warnings],
    )


def build_ldf_table(baseline: ScenarioRun, scenario_runs: list[ScenarioRun]) -> pd.DataFrame:
    labels = list(baseline.ldf.index)
    for run in scenario_runs:
        for label in run.ldf.index:
            if label not in labels:
                labels.append(label)

    rows = []
    for label in labels:
        baseline_value = _series_get_float(baseline.ldf, label)
        row: dict[str, Any] = {
            "age_label": str(label),
            "start_month": Reserving._parse_cdf_label_to_age(label),
            "default_ldf": baseline_value,
        }
        for run in scenario_runs:
            scenario_value = _series_get_float(run.ldf, label)
            delta = scenario_value - baseline_value
            row[f"{run.name}_ldf"] = scenario_value
            row[f"{run.name}_delta"] = delta
            row[f"{run.name}_delta_pct"] = (
                delta / baseline_value
                if pd.notna(baseline_value) and baseline_value != 0
                else pd.NA
            )
        rows.append(row)
    return pd.DataFrame(rows)


def write_line_plot(
    ldf_table: pd.DataFrame,
    *,
    scenario_names: list[str],
    delta_threshold: float,
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_table = ldf_table.copy()
    labels = plot_table["age_label"].astype(str).tolist()
    x_values = list(range(len(labels)))
    width = max(7.0, min(18.0, 0.18 * max(len(labels), 1) + 5.0))
    height = max(3.5, 0.55 * (len(scenario_names) + 1) + 2.5)

    fig, ax = plt.subplots(figsize=(width, height))
    ax.plot(x_values, plot_table["default_ldf"], marker="o", label="default")
    for name in scenario_names:
        ax.plot(x_values, plot_table[f"{name}_ldf"], marker="o", label=name)
        significant = plot_table[f"{name}_delta"].abs().ge(delta_threshold).tolist()
        if any(significant):
            marker_x = [position for position, is_significant in enumerate(significant) if is_significant]
            marker_y = plot_table.loc[plot_table[f"{name}_delta"].abs().ge(delta_threshold), f"{name}_ldf"]
            ax.scatter(marker_x, marker_y, color="red", s=24, zorder=5)

    tick_step = max(1, len(labels) // 20)
    tick_positions = x_values[::tick_step]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([labels[index] for index in tick_positions], rotation=45, ha="right")
    ax.set_xlabel("Development age")
    ax.set_ylabel("LDF")
    ax.set_title("LDF comparison vs default baseline")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def render_markdown_packet(
    *,
    config_path: Path,
    command: str,
    baseline: ScenarioRun,
    scenario_runs: list[ScenarioRun],
    ldf_table: pd.DataFrame,
    plot_path: Path,
    delta_threshold: float,
) -> str:
    lines = [
        "# LDF Compare Packet",
        "",
        "Analysis only - not for booking.",
        "",
        "## Summary",
        "",
        f"Compared {len(scenario_runs)} scenario(s) against the implicit `default` baseline.",
        "",
        "## Scenario Settings",
        "",
        "### default",
        "",
        "```json",
        json.dumps(baseline.effective_settings, indent=2),
        "```",
    ]
    for run in scenario_runs:
        lines.extend(
            [
                "",
                f"### {run.name}",
                "",
                "```json",
                json.dumps(run.effective_settings, indent=2),
                "```",
            ]
        )

    lines.extend(
        [
            "",
            "## Plot",
            "",
            f"![LDF comparison]({plot_path.name})",
            "",
            "## LDF Table",
            "",
            ldf_table.round(8).to_markdown(index=False),
            "",
            "## Warnings",
            "",
        ]
    )
    warning_lines = []
    warning_lines.extend([f"default: {item}" for item in baseline.warnings])
    for run in scenario_runs:
        warning_lines.extend([f"{run.name}: {item}" for item in run.warnings])
    if warning_lines:
        lines.extend([f"- {warning}" for warning in warning_lines])
    else:
        lines.append("- No warnings captured.")

    lines.extend(
        [
            "",
            "## Execution Details",
            "",
            f"- Config path: `{config_path}`",
            f"- Command: `{command}`",
            f"- Plot path: `{plot_path}`",
            f"- Delta threshold for plot markers: {delta_threshold}",
            f"- Timestamp UTC: {datetime.now(timezone.utc).isoformat()}",
            "- Tool calls: `ConfigManager.from_yaml`, `load_inputs_from_config`, `ClaimsCollection`, `PremiumRepository`, `Triangle.from_claims`, `Reserving.set_development`, `Reserving.set_tail`, `Reserving.set_bornhuetter_ferguson`, `Reserving.reserve`, `Reserving.get_ldf`.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def default_output_path() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return DEFAULT_ARTIFACT_DIR / f"ldf_compare_{stamp}.md"


def _series_get_float(series: pd.Series, label: object) -> float:
    value = series.get(label, pd.NA)
    return float(value) if pd.notna(value) else float("nan")
