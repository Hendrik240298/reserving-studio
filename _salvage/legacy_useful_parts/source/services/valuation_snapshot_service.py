from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from typing import Any

import pandas as pd

from source.config_manager import ConfigManager
from source.reserving import Reserving
from source.services.scenario_evaluation_service import ScenarioEvaluationService


class ValuationSnapshotService:
    VERSION = "v1"

    def __init__(
        self,
        *,
        evaluation_service: ScenarioEvaluationService | None = None,
    ) -> None:
        self._evaluation_service = evaluation_service or ScenarioEvaluationService()

    def build_current_snapshot(
        self,
        *,
        reserving: Reserving,
        claims_df: pd.DataFrame,
        premium_df: pd.DataFrame,
        comparison_basis: str = "current",
    ) -> dict[str, Any]:
        return {
            "version": self.VERSION,
            "generated_at": self._utc_now(),
            "comparison_basis": comparison_basis,
            "valuation_date": self._valuation_date(
                claims_df=claims_df, premium_df=premium_df
            ),
            "data_fingerprint": self._dataframe_fingerprint(
                claims_df=claims_df,
                premium_df=premium_df,
            ),
            "summary": self._summary_from_reserving(reserving),
        }

    def build_prior_proxy_snapshot(
        self,
        *,
        claims_df: pd.DataFrame,
        premium_df: pd.DataFrame,
        params: dict[str, Any],
        config: ConfigManager | None,
        comparison_basis: str = "latest_diagonal_excluded_proxy",
    ) -> dict[str, Any]:
        from source.app import build_workflow_from_dataframes

        trimmed_claims, trimmed_premium = self.exclude_latest_diagonal(
            claims_df=claims_df,
            premium_df=premium_df,
        )
        proxy_reserving = build_workflow_from_dataframes(
            trimmed_claims,
            trimmed_premium,
            config=config,
        )
        self._evaluation_service.apply_params_to_reserving(
            reserving=proxy_reserving,
            params=params,
        )
        return {
            "version": self.VERSION,
            "generated_at": self._utc_now(),
            "comparison_basis": comparison_basis,
            "valuation_date": self._valuation_date(
                claims_df=trimmed_claims,
                premium_df=trimmed_premium,
            ),
            "data_fingerprint": self._dataframe_fingerprint(
                claims_df=trimmed_claims,
                premium_df=trimmed_premium,
            ),
            "summary": self._summary_from_reserving(proxy_reserving),
            "source_row_counts": {
                "claims": len(trimmed_claims),
                "premium": len(trimmed_premium),
            },
        }

    def build_delta_summary(
        self,
        *,
        current_snapshot: dict[str, Any] | None,
        prior_snapshot: dict[str, Any] | None,
    ) -> dict[str, Any]:
        current = current_snapshot if isinstance(current_snapshot, dict) else {}
        prior = prior_snapshot if isinstance(prior_snapshot, dict) else {}
        current_summary = (
            current.get("summary") if isinstance(current.get("summary"), dict) else {}
        )
        prior_summary = (
            prior.get("summary") if isinstance(prior.get("summary"), dict) else {}
        )
        current_methods = (
            current_summary.get("selected_method_counts")
            if isinstance(current_summary.get("selected_method_counts"), dict)
            else {}
        )
        prior_methods = (
            prior_summary.get("selected_method_counts")
            if isinstance(prior_summary.get("selected_method_counts"), dict)
            else {}
        )
        return {
            "comparison_basis": str(
                prior.get("comparison_basis") or current.get("comparison_basis") or ""
            ),
            "current_valuation_date": current.get("valuation_date"),
            "prior_valuation_date": prior.get("valuation_date"),
            "metrics": {
                "uwy_count_delta": self._rounded_delta(
                    current_summary.get("uwy_count"),
                    prior_summary.get("uwy_count"),
                ),
                "total_ultimate_delta": self._rounded_delta(
                    current_summary.get("total_ultimate"),
                    prior_summary.get("total_ultimate"),
                ),
                "total_incurred_delta": self._rounded_delta(
                    current_summary.get("total_incurred"),
                    prior_summary.get("total_incurred"),
                ),
                "total_ibnr_delta": self._rounded_delta(
                    current_summary.get("total_ibnr"),
                    prior_summary.get("total_ibnr"),
                ),
            },
            "selected_method_count_delta": {
                key: self._rounded_delta(
                    current_methods.get(key, 0), prior_methods.get(key, 0)
                )
                for key in sorted(set(current_methods) | set(prior_methods))
            },
            "fingerprints": {
                "current": current.get("data_fingerprint"),
                "prior_proxy": prior.get("data_fingerprint"),
            },
        }

    def exclude_latest_diagonal(
        self,
        *,
        claims_df: pd.DataFrame,
        premium_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        trimmed_claims = self._drop_latest_origin_period_rows(claims_df)
        trimmed_premium = self._drop_latest_origin_period_rows(premium_df)
        if trimmed_claims.empty:
            raise ValueError("Latest-diagonal proxy would remove all claim rows")
        if trimmed_premium.empty:
            raise ValueError("Latest-diagonal proxy would remove all premium rows")
        return trimmed_claims, trimmed_premium

    @staticmethod
    def _drop_latest_origin_period_rows(dataframe: pd.DataFrame) -> pd.DataFrame:
        if dataframe.empty:
            return dataframe.copy()
        if "uw_year" not in dataframe.columns or "period" not in dataframe.columns:
            raise ValueError(
                "Valuation snapshot source data must contain uw_year and period columns"
            )
        working = dataframe.copy()
        working["uw_year"] = working["uw_year"].astype(str)
        period_series = pd.to_datetime(working["period"], errors="coerce")
        if period_series.notna().sum() == 0:
            period_series = pd.Series(
                pd.to_numeric(working["period"], errors="coerce"),
                index=working.index,
            )
        max_period_by_origin = period_series.groupby(working["uw_year"]).transform(
            "max"
        )
        keep_mask = period_series < max_period_by_origin
        if keep_mask.sum() == 0:
            keep_mask = ~working.index.duplicated(keep="first")
            keep_mask.iloc[-1] = False
        trimmed = working.loc[keep_mask].copy()
        return trimmed

    @staticmethod
    def _summary_from_reserving(reserving: Reserving) -> dict[str, Any]:
        results_df = reserving.get_results()
        total_ultimate = float(results_df["ultimate"].sum()) if len(results_df) else 0.0
        total_incurred = float(results_df["incurred"].sum()) if len(results_df) else 0.0
        selected_counts = (
            results_df["selected_method"].astype(str).value_counts().to_dict()
            if len(results_df) and "selected_method" in results_df.columns
            else {}
        )
        return {
            "uwy_count": int(len(results_df.index)),
            "total_ultimate": round(total_ultimate, 6),
            "total_incurred": round(total_incurred, 6),
            "total_ibnr": round(total_ultimate - total_incurred, 6),
            "selected_method_counts": {
                str(key): int(value) for key, value in selected_counts.items()
            },
        }

    @staticmethod
    def _valuation_date(
        *,
        claims_df: pd.DataFrame,
        premium_df: pd.DataFrame,
    ) -> str | None:
        candidates: list[pd.Timestamp] = []
        for frame in [claims_df, premium_df]:
            if frame.empty or "period" not in frame.columns:
                continue
            series = pd.to_datetime(frame["period"], errors="coerce")
            non_null = series.dropna()
            if non_null.empty:
                continue
            candidates.append(non_null.max())
        if not candidates:
            return None
        return max(candidates).isoformat()

    @staticmethod
    def _dataframe_fingerprint(
        *,
        claims_df: pd.DataFrame,
        premium_df: pd.DataFrame,
    ) -> str:
        payload = {
            "claims": ValuationSnapshotService._records(claims_df),
            "premium": ValuationSnapshotService._records(premium_df),
        }
        canonical = json.dumps(
            payload, sort_keys=True, default=str, separators=(",", ":")
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @staticmethod
    def _records(dataframe: pd.DataFrame) -> list[dict[str, Any]]:
        records = dataframe.to_dict(orient="records")
        return [
            {
                str(key): ScenarioEvaluationService.json_safe_value(value)
                for key, value in row.items()
            }
            for row in records
            if isinstance(row, dict)
        ]

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    @staticmethod
    def _rounded_delta(current_value: object, prior_value: object) -> float | None:
        current = pd.to_numeric(current_value, errors="coerce")
        prior = pd.to_numeric(prior_value, errors="coerce")
        if pd.isna(current) or pd.isna(prior):
            return None
        return round(float(current) - float(prior), 6)
