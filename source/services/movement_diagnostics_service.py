from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from source.reserving import Reserving
from source.services.data_view_service import (
    DataViewQuery,
    DataViewService,
    summarize_distribution,
)


@dataclass(frozen=True)
class MovementFinding:
    code: str
    severity: str
    message: str
    evidence: dict[str, Any]


class MovementDiagnosticsService:
    def __init__(self, reserving: Reserving) -> None:
        self._reserving = reserving
        self._views = DataViewService(reserving)

    def run(self) -> dict[str, Any]:
        findings: list[MovementFinding] = []
        findings.extend(self._premium_findings())
        findings.extend(self._incurred_findings())
        findings.extend(self._outstanding_findings())
        findings.extend(self._large_loss_proxy_findings())
        findings_payload = [
            {
                "code": item.code,
                "severity": item.severity,
                "message": item.message,
                "evidence": item.evidence,
            }
            for item in findings
        ]
        summary = {
            "finding_count": len(findings_payload),
            "top_findings": findings_payload[:5],
        }
        return {"findings": findings_payload, "summary": summary}

    def run_ldf_consistency(self) -> dict[str, Any]:
        heatmap_data = self._reserving.get_triangle_heatmap_data()
        link_ratios_raw = heatmap_data.get("link_ratios")
        incurred_raw = heatmap_data.get("incurred")
        if not isinstance(link_ratios_raw, pd.DataFrame) or not isinstance(
            incurred_raw, pd.DataFrame
        ):
            return {"findings": [], "summary": {"finding_count": 0}}
        link_ratios = link_ratios_raw.apply(pd.to_numeric, errors="coerce")
        incurred = incurred_raw.apply(pd.to_numeric, errors="coerce")
        if "LDF" not in link_ratios.index:
            return {"findings": [], "summary": {"finding_count": 0}}
        ldf_row = link_ratios.loc["LDF"]
        triangle_only = link_ratios.loc[
            ~link_ratios.index.astype(str).isin(["LDF", "Tail"])
        ]
        findings: list[dict[str, Any]] = []
        for origin, row in triangle_only.iterrows():
            origin_label = self._origin_label(origin)
            for col in triangle_only.columns:
                observed = self._as_float(row.get(col))
                selected = self._as_float(ldf_row.get(col))
                if observed is None or selected is None or selected == 0:
                    continue
                rel_gap = abs(observed - selected) / abs(selected)
                if rel_gap < 0.15:
                    continue
                base_value = self._preceding_cumulative(incurred, origin, col)
                impact_estimate = None
                if base_value is not None:
                    impact_estimate = round(base_value * (observed - selected), 6)
                findings.append(
                    {
                        "origin": origin_label,
                        "age": str(col),
                        "observed_a2a": round(observed, 6),
                        "selected_ldf": round(selected, 6),
                        "relative_gap": round(rel_gap, 6),
                        "impact_estimate": impact_estimate,
                        "message": (
                            f"AY {origin_label} observed age-to-age {observed:.3f} differs "
                            f"from selected LDF {selected:.3f} at age {col}."
                        ),
                    }
                )
        findings.sort(
            key=lambda item: abs(float(item.get("impact_estimate") or 0.0)),
            reverse=True,
        )
        return {
            "findings": findings[:10],
            "summary": {
                "finding_count": len(findings),
                "top_findings": findings[:5],
            },
        }

    def run_late_emergence_benchmark(self, *, uwy: str | None = None) -> dict[str, Any]:
        incurred = self._views.get_data_view(
            DataViewQuery(metric="incurred", view="cumulative")
        )
        results_df = self._reserving.get_results()
        numeric = incurred.apply(pd.to_numeric, errors="coerce")
        rows: list[dict[str, Any]] = []
        for origin in numeric.index:
            origin_label = self._origin_label(origin)
            if uwy and origin_label != str(uwy):
                continue
            series = numeric.loc[origin].dropna()
            if len(series) < 1:
                continue
            current_age = series.index[-1]
            current_cumulative = float(series.iloc[-1])
            comparables: list[float] = []
            for other_origin in numeric.index:
                if other_origin == origin:
                    continue
                other = numeric.loc[other_origin].dropna()
                if current_age not in other.index:
                    continue
                if len(other.index) <= list(other.index).index(current_age) + 1:
                    continue
                start_value = self._as_float(other.loc[current_age])
                end_value = self._as_float(other.iloc[-1])
                if start_value is None or end_value is None or start_value <= 0:
                    continue
                comparables.append((end_value - start_value) / start_value)

            selected_ratio = None
            if origin in results_df.index:
                incurred_value = float(
                    results_df.loc[origin].get("incurred", 0.0) or 0.0
                )
                ultimate_value = float(
                    results_df.loc[origin].get("ultimate", 0.0) or 0.0
                )
                if incurred_value > 0:
                    selected_ratio = (ultimate_value - incurred_value) / incurred_value

            benchmark = summarize_distribution(comparables)
            rows.append(
                {
                    "origin": origin_label,
                    "current_age": str(current_age),
                    "current_cumulative": round(current_cumulative, 6),
                    "selected_future_ratio": round(float(selected_ratio), 6)
                    if selected_ratio is not None
                    else None,
                    **benchmark,
                }
            )
        return {
            "rows": rows,
            "summary": {
                "row_count": len(rows),
                "top_rows": rows[:5],
            },
        }

    def _premium_findings(self) -> list[MovementFinding]:
        premium = self._views.get_data_view(
            DataViewQuery(metric="premium", view="incremental")
        )
        numeric = premium.apply(pd.to_numeric, errors="coerce")
        findings: list[MovementFinding] = []
        for origin, row in numeric.iterrows():
            series = row.dropna()
            if len(series) < 3:
                continue
            tail_age = series.index[-1]
            tail_value = float(series.iloc[-1])
            peak = float(series.abs().max() or 0.0)
            if peak <= 0:
                continue
            ratio = abs(tail_value) / peak
            if ratio < 0.1:
                continue
            origin_label = self._origin_label(origin)
            findings.append(
                MovementFinding(
                    code=f"PREMIUM_LATE_MOVEMENT_{origin_label}_{tail_age}",
                    severity="medium" if ratio < 0.2 else "high",
                    message=(
                        f"AY {origin_label} shows a non-trivial premium movement at age {tail_age}, "
                        f"which is unusual for an older development period."
                    ),
                    evidence={
                        "metric_id": f"premium_late_movement_{origin_label}_{tail_age}",
                        "value": round(tail_value, 6),
                        "relative_to_peak": round(ratio, 6),
                    },
                )
            )
        return findings[:8]

    def _incurred_findings(self) -> list[MovementFinding]:
        incurred = self._views.get_data_view(
            DataViewQuery(metric="incurred", view="incremental")
        )
        numeric = incurred.apply(pd.to_numeric, errors="coerce")
        findings: list[MovementFinding] = []
        for col in numeric.columns:
            column = numeric[col].dropna()
            if len(column) < 4:
                continue
            center = float(column.median())
            mad = float((column - center).abs().median())
            if mad <= 0:
                continue
            scale = max(1.4826 * mad, 1e-9)
            for origin, value in column.items():
                robust_z = abs((float(value) - center) / scale)
                if robust_z < 3.0:
                    continue
                origin_label = self._origin_label(origin)
                findings.append(
                    MovementFinding(
                        code=f"INCURRED_SPIKE_{origin_label}_{col}",
                        severity="high" if robust_z >= 4.5 else "medium",
                        message=(
                            f"AY {origin_label} incurred movement at age {col} is unusually large "
                            f"versus peer years."
                        ),
                        evidence={
                            "metric_id": f"incurred_incremental_robust_z_{origin_label}_{col}",
                            "value": round(float(robust_z), 6),
                            "threshold": 3.0,
                        },
                    )
                )
        return findings[:10]

    def _outstanding_findings(self) -> list[MovementFinding]:
        outstanding = self._views.get_data_view(
            DataViewQuery(metric="outstanding", view="cumulative")
        )
        incurred = self._views.get_data_view(
            DataViewQuery(metric="incurred", view="cumulative")
        )
        outstanding_numeric = outstanding.apply(pd.to_numeric, errors="coerce")
        incurred_numeric = incurred.apply(pd.to_numeric, errors="coerce")
        findings: list[MovementFinding] = []
        for origin in outstanding_numeric.index:
            out_series = outstanding_numeric.loc[origin].dropna()
            inc_series = incurred_numeric.loc[origin].dropna()
            if len(out_series) == 0 or len(inc_series) == 0:
                continue
            out_value = float(out_series.iloc[-1])
            inc_value = float(inc_series.iloc[-1])
            if inc_value <= 0:
                continue
            ratio = out_value / inc_value
            if ratio < 0.45:
                continue
            origin_label = self._origin_label(origin)
            findings.append(
                MovementFinding(
                    code=f"OUTSTANDING_CONCENTRATION_{origin_label}",
                    severity="medium" if ratio < 0.65 else "high",
                    message=(
                        f"AY {origin_label} has a relatively large outstanding share versus incurred, "
                        "which may indicate higher reserve uncertainty."
                    ),
                    evidence={
                        "metric_id": f"outstanding_to_incurred_{origin_label}",
                        "value": round(ratio, 6),
                        "threshold": 0.45,
                    },
                )
            )
        return findings[:8]

    def _large_loss_proxy_findings(self) -> list[MovementFinding]:
        heatmap_data = self._reserving.get_triangle_heatmap_data()
        link_ratios_raw = heatmap_data.get("link_ratios")
        if not isinstance(link_ratios_raw, pd.DataFrame):
            return []
        link_ratios = link_ratios_raw.apply(pd.to_numeric, errors="coerce")
        triangle_only = link_ratios.loc[
            ~link_ratios.index.astype(str).isin(["LDF", "Tail"])
        ]
        findings: list[MovementFinding] = []
        columns = list(triangle_only.columns)
        for origin, row in triangle_only.iterrows():
            origin_label = self._origin_label(origin)
            for idx, col in enumerate(columns[:-1]):
                spike = self._as_float(row.get(col))
                if spike is None:
                    continue
                peer = triangle_only[col].dropna()
                if len(peer) < 4:
                    continue
                center = float(peer.median())
                mad = float((peer - center).abs().median())
                threshold = center + max(1.4826 * mad * 2.5, 0.25)
                if spike < threshold:
                    continue
                reversal_cols = columns[idx + 1 : idx + 3]
                reversal = None
                reversal_col = None
                for next_col in reversal_cols:
                    next_value = self._as_float(row.get(next_col))
                    if next_value is not None and next_value < 1.0:
                        reversal = next_value
                        reversal_col = next_col
                        break
                if reversal is None or reversal_col is None:
                    continue
                confidence = "medium"
                if spike >= threshold * 1.15 and reversal <= 0.95:
                    confidence = "high"
                findings.append(
                    MovementFinding(
                        code=f"LARGE_LOSS_PROXY_{origin_label}_{col}",
                        severity=confidence,
                        message=(
                            f"AY {origin_label} shows a large spike age-to-age factor at age {col} "
                            f"followed by a sub-1.0 reversal at age {reversal_col}, which is consistent "
                            "with an aggregated large-loss proxy pattern."
                        ),
                        evidence={
                            "metric_id": f"large_loss_proxy_{origin_label}_{col}",
                            "value": round(spike, 6),
                            "reversal_value": round(reversal, 6),
                            "threshold": round(threshold, 6),
                            "basis": "high age-to-age spike followed by sub-1.0 reversal within two quarters",
                        },
                    )
                )
        return findings[:8]

    @staticmethod
    def _as_float(value: object) -> float | None:
        numeric = pd.to_numeric(value, errors="coerce")
        if pd.isna(numeric):
            return None
        return float(numeric)

    @staticmethod
    def _origin_label(origin: object) -> str:
        if hasattr(origin, "year"):
            return str(origin.year)
        text = str(origin)
        return text[:4] if len(text) >= 4 and text[:4].isdigit() else text

    @staticmethod
    def _preceding_cumulative(
        incurred: pd.DataFrame,
        origin: object,
        column: object,
    ) -> float | None:
        columns = list(incurred.columns)
        if column not in columns:
            return None
        index = columns.index(column)
        if index == 0:
            value = pd.to_numeric(incurred.loc[origin, column], errors="coerce")
            return None if pd.isna(value) else float(value)
        previous = columns[index - 1]
        value = pd.to_numeric(incurred.loc[origin, previous], errors="coerce")
        return None if pd.isna(value) else float(value)
