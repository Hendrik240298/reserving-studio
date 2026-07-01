from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from source.claims_collection import ClaimsCollection
from source.config_manager import ConfigManager
from source.input_loader import load_inputs_from_config
from source.premium_repository import PremiumRepository
from source.triangle import Triangle
from source.reserving import Reserving
import logging

import pandas as pd
from statistics import median
REPO_ROOT = Path(__file__).parents[1]
DEFAULT_CONFIG_PATH = Path("examples/config_quarterly.yml")
DEFAULT_ARTIFACT_DIR = Path("harness/artifacts")


def analyze_candidates(
    reserving: Reserving,
    candidate_limit: int = 10,
):
    ldf = reserving.get_ldf()
    link_ratios = reserving.get_link_ratios()

    signals = []

    for period, ldf_period in ldf.items(): # loop over periods; easier with loop over ldf since only one per period
        if pd.isna(ldf_period) or ldf_period <= 0:
            continue
        if period not in link_ratios.columns:
            continue
        ratios_period = link_ratios[period].dropna()
        logging.info(f"ldf: {ldf_period}, ratios: {ratios_period}")

        values = [float(value) for value in ratios_period.tolist()] # getting just the value of the Serien

        if len(values) <= 3:
            continue
        # auxilliary for z value
        center = median(values)
        deviation = [abs(value - center) for value in values]
        mad = median(deviation)
        scale = max(1.4826 * mad, 1e-9)

        for origin, ratio in ratios_period.items():
            z_value = abs((ratio-center) / scale)
            relative_gap = abs(ratio - ldf_period) / abs(ldf_period)
            if z_value < 3.0 and relative_gap < 0.15:
                continue
            signal_score = round(z_value + relative_gap*4., 6)
            priority = 1 if z_value >= 4.5 or relative_gap >= 0.25 else 0 # 1 for high priority and 0 for medium
            signals.append({
                'origin': origin,
                'period': period,
                'link_ratio': ratio,
                'ldf': ldf_period,
                'z_value': z_value,
                'relative_gap': relative_gap,
                'signal_score': signal_score,
                'priority': priority
            })
    if not signals:
        return pd.DataFrame()

    df = pd.DataFrame(signals)
    df.sort_values(by=['priority','signal_score'],ascending=False, inplace=True)

    count_high = df[df['priority']==1].shape[0]

    return df.iloc[:max([count_high, candidate_limit]),:]


def drops_from_signals(signals: pd.DataFrame) -> list[tuple[str, int]]:
    drops: list[tuple[str, int]] = []

    for _, row in signals.iterrows():
        age = Reserving._parse_cdf_label_to_age(row["period"])
        if age is None:
            continue
        drops.append((str(row["origin"]), age))

    return drops


def origin_year_label(origin: object) -> str:
    if hasattr(origin, "year"):
        return str(origin.year)
    text = str(origin)
    return text[:4] if len(text) >= 4 and text[:4].isdigit() else text


def analyze_drop_effects(
    triangle: Triangle,
    candidate_limit: int = 5,
):
    baseline = Reserving(triangle=triangle)
    baseline.set_development()
    baseline.set_tail()
    baseline.set_bornhuetter_ferguson()
    baseline.reserve()

    signals = analyze_candidates(
        reserving=baseline,
        candidate_limit=candidate_limit,
    )

    if signals.empty:
        return {
            "signals": signals,
            "drops": [],
            "baseline_results": baseline.get_results(),
            "drop_results": None,
        }

    drops = drops_from_signals(signals)

    if not drops:
        return {
            "signals": signals,
            "drops": [],
            "baseline_results": baseline.get_results(),
            "drop_results": None,
        }

    reserving = Reserving(triangle=triangle)
    reserving.set_development(drop=drops)
    reserving.set_tail()
    reserving.set_bornhuetter_ferguson()
    reserving.reserve()

    return {
        "signals": signals,
        "drops": drops,
        "baseline_results": baseline.get_results(),
        "drop_results": reserving.get_results(),
    }


def ultimates_impact_from_analysis(analysis: dict) -> pd.DataFrame | None:
    baseline_results = analysis["baseline_results"]
    drop_results = analysis["drop_results"]

    if drop_results is None:
        return None

    ultimates_impact = baseline_results[["ultimate"]].join(
        drop_results[["ultimate"]],
        lsuffix="_baseline",
        rsuffix="_drop",
    )
    ultimates_impact["ultimate_delta"] = (
        ultimates_impact["ultimate_drop"] - ultimates_impact["ultimate_baseline"]
    )
    ultimates_impact.index = [origin_year_label(origin) for origin in ultimates_impact.index]
    ultimates_impact.index.name = "origin"
    return ultimates_impact


def drop_review_summary(analysis: dict) -> str:
    signals = analysis["signals"]
    drops = analysis["drops"]
    return f"Drop review found {len(signals)} candidate signals and applied {len(drops)} drops in the combined scenario."


def render_drop_review_packet(analysis: dict) -> str:
    signals = analysis["signals"]
    drops = analysis["drops"]
    ultimates_impact = ultimates_impact_from_analysis(analysis)

    lines = [
        "# Drop Review Packet",
        "",
        "## Summary",
        "",
        drop_review_summary(analysis),
        "",
        "## Run Status",
        "",
        f"- Status: {'ok' if drops else 'warning'}",
        f"- Candidate signals: {len(signals)}",
        f"- Applied drops: {len(drops)}",
        "",
        "## Ultimates Impact",
        "",
    ]

    if ultimates_impact is None or ultimates_impact.empty:
        lines.append("No ultimates impact available.")
    else:
        lines.append(ultimates_impact.round(2).to_markdown())

    lines.extend(["", "## Candidate Signals", ""])
    if signals.empty:
        lines.append("No candidate signals found.")
    else:
        display_signals = signals.copy()
        display_signals["origin"] = display_signals["origin"].map(origin_year_label)
        lines.append(display_signals.round(4).to_markdown(index=False))

    lines.extend(["", "## Warnings", ""])
    if drops:
        lines.append("- None")
    else:
        lines.append("- No valid drop candidates were applied.")

    return "\n".join(lines).rstrip() + "\n"


def default_output_path() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return DEFAULT_ARTIFACT_DIR / f"drop_review_quarterly_{stamp}.md"


def run_drop_review_packet(
        config_path: Path,
        candidate_limit: int = 5,
        output_path: Path | None = None,
) -> Path:
    # 1. Load data
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
    triangle = Triangle.from_claims(
        claims=claims,
        premium=premium,
    )

    # 2. Analysis
    analysis = analyze_drop_effects(
        triangle=triangle,
        candidate_limit=candidate_limit,
    )

    # 3. Render and write packet
    output_path = output_path or default_output_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_drop_review_packet(analysis), encoding="utf-8")

    return output_path
