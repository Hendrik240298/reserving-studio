from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from harness.markdown import drop_review_summary, render_drop_review_run
from source.claims_collection import ClaimsCollection
from source.config_manager import ConfigManager
from source.drop_analysis import DEFAULT_METHOD, analyze_drop_effects
from source.input_loader import load_inputs_from_config
from source.premium_repository import PremiumRepository
from source.triangle import Triangle


REPO_ROOT = Path(__file__).parents[1]
DEFAULT_CONFIG_PATH = Path("examples/config_quarterly.yml")
DEFAULT_ARTIFACT_DIR = Path("harness/artifacts")


@dataclass(frozen=True)
class DropReviewPacketResult:
    output_path: Path
    review_type: str
    recommendation_class: str | None
    candidate_id: str | None
    candidate_count: int
    warnings: list[str] = field(default_factory=list)


def run_drop_review_packet(
    *,
    config_path: Path = DEFAULT_CONFIG_PATH,
    output_path: Path | None = None,
    candidate_limit: int = 5,
    method: Literal["chainladder", "bornhuetter_ferguson"] = DEFAULT_METHOD,
    use_tail: bool = False,
    enforce_monotone_tail: bool = False,
) -> DropReviewPacketResult:
    if output_path is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_path = DEFAULT_ARTIFACT_DIR / f"drop_review_quarterly_{stamp}.md"
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path
    if not output_path.is_absolute():
        output_path = Path.cwd() / output_path

    config = ConfigManager.from_yaml(config_path)
    claims_df, premium_df = load_inputs_from_config(config, repo_root=REPO_ROOT)
    session = config.load_session()

    claims = ClaimsCollection(
        claims_df,
        values_are_cumulative=bool(claims_df.attrs.get("values_are_cumulative", False)),
    )
    premium = PremiumRepository.from_dataframe(config_manager=config, dataframe=premium_df)
    triangle = Triangle.from_claims(claims=claims, premium=premium)

    base_drops = [(str(origin), int(age)) for origin, age in session.get("drops", [])]
    tail_fit_period = [int(age) for age in session.get("tail_fit_period", [])]
    tail_attachment_age = session.get("tail_attachment_age")
    tail_attachment_age = None if tail_attachment_age is None else int(tail_attachment_age)
    bf_apriori_by_uwy = {
        str(origin): float(value)
        for origin, value in (session.get("bf_apriori_by_uwy") or {}).items()
    }

    analysis = analyze_drop_effects(
        triangle,
        base_drops=base_drops,
        candidate_limit=candidate_limit,
        average=str(session.get("average", "volume")),
        method=method,
        use_tail=use_tail,
        enforce_monotone_tail=enforce_monotone_tail,
        tail_curve=str(session.get("tail_curve", "weibull")),
        tail_attachment_age=tail_attachment_age,
        tail_projection_months=max(int(session.get("tail_projection_months", 0) or 0), 0),
        tail_fit_period=tail_fit_period,
        bf_apriori_by_uwy=bf_apriori_by_uwy,
    )
    config_label = str(config_path.relative_to(REPO_ROOT))
    output_label = str(output_path.relative_to(REPO_ROOT))
    packet = render_drop_review_run(
        analysis=analysis,
        config=config,
        config_path=config_label,
        output_path=output_label,
        candidate_limit=candidate_limit,
        method=method,
        use_tail=use_tail,
        enforce_monotone_tail=enforce_monotone_tail,
        average=str(session.get("average", "volume")),
        base_drop_count=len(base_drops),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(packet, encoding="utf-8")

    summary = drop_review_summary(analysis)
    recommendation = summary["recommendation"]
    return DropReviewPacketResult(
        output_path=output_path,
        review_type=str(summary["review_type"] or "native_drop_analysis"),
        recommendation_class=recommendation.get("recommendation_class"),
        candidate_id=recommendation.get("candidate_id"),
        candidate_count=int(summary["candidate_count"]),
        warnings=[str(item) for item in recommendation.get("caveats", []) if item],
    )
