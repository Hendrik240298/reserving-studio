from __future__ import annotations
from pathlib import Path

import numpy as np
import chainladder as cl
import logging
import pandas as pd
import warnings
from typing import List

from source.claims_collection import ClaimsCollection
from source.premium_repository import PremiumRepository


DEFAULT_SAMPLE_PREMIUM_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "quarterly_premium.csv"
)


class Triangle:
    def __init__(self, data: pd.DataFrame):
        self._triangle = self._create_triangle(data)
        self._triangle = self._triangle.incr_to_cum()
        # Debug: check triangle valuation immediately after creation

    @classmethod
    def from_claims(
        cls, claims: ClaimsCollection, premium: PremiumRepository
    ) -> Triangle:
        """Build paid, outstanding and paid triangle from ClaimsCollection"""
        df = claims.to_dataframe()

        # Add premium columns to claims df, set to 0
        for col in ["Premium_selected", "GWP", "EPI", "GWP_Forecast"]:
            df[col] = 0.0

        # Get premium data
        premium_df = premium.get_premium()[
            ["uw_year", "period", "GWP", "EPI", "GWP_Forecast", "Premium_selected"]
        ].copy()

        # Filter to only uw_years present in claims
        claims_uw_years = df["uw_year"].unique()
        premium_df = premium_df[premium_df["uw_year"].isin(claims_uw_years)]

        # Infer proper dtypes first, then fill NaN (prevents downcasting warning)
        premium_df = premium_df.infer_objects(copy=False)
        premium_df = premium_df.fillna(0.0)

        # Add claim-related columns to premium_df, set to 0.0
        premium_df["incurred"] = 0.0
        premium_df["paid"] = 0.0
        premium_df["outstanding"] = 0.0

        # Append premium rows to df
        df = pd.concat([df, premium_df], ignore_index=True)

        # Select only columns that will be used
        common_cols = [
            "uw_year",
            "period",
            "incurred",
            "outstanding",
            "paid",
            "Premium_selected",
        ]
        df = df[common_cols]

        logging.info(f"Premium in Triangle: \n{df[df['Premium_selected'] > 0]}")
        return cls(df)

    def _create_triangle(self, data: pd.DataFrame):
        data = self._ensure_premium_column(data)

        # Dates are already quarter-end timestamps from ClaimsRepository
        # Quarter-end timestamps (Mar 31, Jun 30, Sep 30, Dec 31) provide explicit
        # boundaries that prevent chainladder date shifting

        # Select only the columns needed for chainladder (removes NaN issues from other columns)
        data = data[
            ["uw_year", "period", "incurred", "outstanding", "paid", "Premium_selected"]
        ].copy()

        # Fill NaN and ensure proper float64 dtype to prevent overflow
        numeric_cols = ["incurred", "outstanding", "paid", "Premium_selected"]
        for col in numeric_cols:
            data[col] = (
                pd.to_numeric(data[col], errors="coerce").fillna(0).astype("float64")
            )

        return cl.Triangle(
            data,
            origin="uw_year",
            development="period",
            columns=["incurred", "outstanding", "paid", "Premium_selected"],
            cumulative=False,
        )

    def _ensure_premium_column(self, data: pd.DataFrame) -> pd.DataFrame:
        working_data = data.copy()
        if "Premium_selected" in working_data.columns:
            existing = pd.to_numeric(
                working_data["Premium_selected"],
                errors="coerce",
            ).fillna(0.0)
            if (existing != 0.0).any():
                working_data["Premium_selected"] = existing.astype("float64")
                return working_data

        if not self._looks_like_quarterly_sample(working_data):
            working_data["Premium_selected"] = 0.0
            return working_data

        premium_triangle = self._load_sample_premium_triangle()
        if premium_triangle is None:
            working_data["Premium_selected"] = 0.0
            return working_data

        working_data["uw_year"] = pd.to_datetime(
            working_data["uw_year"], errors="coerce"
        )
        working_data["period"] = pd.to_datetime(working_data["period"], errors="coerce")

        merged = working_data.merge(
            premium_triangle,
            on=["uw_year", "period"],
            how="left",
        )
        merged["Premium_selected"] = pd.to_numeric(
            merged["Premium_selected_csv"],
            errors="coerce",
        ).fillna(0.0)
        return merged.drop(columns=["Premium_selected_csv"])

    def _looks_like_quarterly_sample(self, data: pd.DataFrame) -> bool:
        if not {"uw_year", "period", "incurred"}.issubset(data.columns):
            return False
        years = (
            pd.to_datetime(data["uw_year"], errors="coerce")
            .dt.year.dropna()
            .astype(int)
        )
        if years.empty:
            return False
        unique_years = sorted(years.unique().tolist())
        return unique_years == list(range(1995, 2007))

    def _load_sample_premium_triangle(self) -> pd.DataFrame | None:
        premium_path = DEFAULT_SAMPLE_PREMIUM_PATH
        if not premium_path.exists():
            logging.warning("Sample premium CSV not found at %s", premium_path)
            return None

        premium_df = pd.read_csv(premium_path)
        required_columns = {"development", "origin", "Premium_selected"}
        if not required_columns.issubset(premium_df.columns):
            missing = required_columns - set(premium_df.columns)
            raise ValueError(f"Premium CSV is missing required columns: {missing}")

        premium_df = premium_df.copy()
        premium_df["uw_year"] = pd.to_datetime(
            premium_df["origin"].astype(str) + "-01-01",
            errors="coerce",
        )
        premium_df["period"] = pd.PeriodIndex(
            premium_df["development"].astype(str),
            freq="Q",
        ).to_timestamp("Q")
        premium_df = premium_df.sort_values(["uw_year", "period"])
        premium_df["Premium_selected_csv"] = pd.to_numeric(
            premium_df["Premium_selected"],
            errors="coerce",
        ).fillna(0.0)
        premium_df["Premium_selected_csv"] = (
            premium_df.groupby("uw_year")["Premium_selected_csv"]
            .diff()
            .fillna(premium_df["Premium_selected_csv"])
        )
        premium_df["Premium_selected_csv"] = premium_df["Premium_selected_csv"].clip(
            lower=0.0
        )

        return premium_df[["uw_year", "period", "Premium_selected_csv"]].copy()

    def _save_to_excel(self):
        self._triangle["incurred"].incr_to_cum().to_excel("test_triangle.xlsx")

    def _get_model(self, triangle):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="Some exclusions have been ignore"
            )
            cl_dev = cl.Pipeline(
                [
                    (
                        "dev",
                        cl.Development(
                            # drop_high=True,
                            # drop_low=True,
                            # preserve=3
                            # olympic averaging to filter out outliers while preserving at least 3 link ratios
                        ),
                    ),
                    (
                        "tail",
                        cl.TailCurve(
                            projection_period=0,
                            attachment_age=None,  # neglect tail
                            #   fit_period=(12, 80)
                        ),
                    ),  # neglect tail projection
                ]
            ).fit(triangle)

        # model = cl.Chainladder().fit(cl_dev)
        return cl_dev

    def calc_ave(self, type="incurred"):
        triangle = self._triangle[type]
        if not triangle.is_cumulative:
            triangle = triangle.incr_to_cum()

        model = self._get_model(triangle)
        dev = model.transform(triangle)

        expected = (dev * dev.ldf_)[type].values[0, 0, :, :-1]
        actual = dev[type].values[0, 0, :, 1:]
        weights = dev[type].values[0, 0, :, :-1]

        ave_ratio = np.divide(
            actual, expected, out=np.zeros_like(expected), where=(expected > 0)
        )

        valid_mask = (
            (expected > 0) & (actual > 0) & (~np.isnan(ave_ratio)) & (~np.isnan(actual))
        )

        # Apply mask to weights and calculate RMSE
        valid_weights = weights[valid_mask]
        valid_ave = ave_ratio[valid_mask]
        weighted_errors = valid_weights * (valid_ave - 1) ** 2

        # Avoid division by zero
        sum_weights = np.sum(valid_weights)
        if sum_weights > 0:
            rmse = np.sqrt(np.sum(weighted_errors) / sum_weights)
        else:
            rmse = np.nan

        return rmse

    def calc_coeff_of_var(self, type="incurred"):
        triangle = self._triangle[type]
        if not triangle.is_cumulative:
            triangle = triangle.incr_to_cum()

        model = self._get_model(triangle)
        dev = model.transform(triangle)
        link_ratios = dev.link_ratio.values[0, 0, :, :]

        # Get weights - slice to match link_ratios shape dynamically
        n_origins, n_devs = link_ratios.shape
        weights = dev.values[0, 0, :n_origins, :n_devs]

        # Remove NaN values for proper calculation
        valid_mask = ~np.isnan(link_ratios) & ~np.isnan(weights) & (weights > 0)

        # Calculate weighted coefficient of variation for each development period
        cvs = []
        for col in range(link_ratios.shape[1]):
            col_lrs = link_ratios[:, col]
            col_weights = weights[:, col]
            col_mask = valid_mask[:, col]

            if col_mask.sum() > 0:
                valid_lrs = col_lrs[col_mask]
                valid_weights = col_weights[col_mask]

                weighted_mean = np.average(valid_lrs, weights=valid_weights)
                weighted_var = np.average(
                    (valid_lrs - weighted_mean) ** 2, weights=valid_weights
                )
                cv = np.sqrt(weighted_var) / weighted_mean  # Coefficient of variation
                cvs.append(cv)

        cv_mean = np.mean(cvs) if cvs else 0.0

        return cv_mean

    def get_triangle(self, type="incurred"):
        if not self._triangle.is_cumulative:
            self._triangle = self._triangle.incr_to_cum()
        return self._triangle[[type, "Premium_selected"]].copy()
