"""
follow the workflow in ``https://chainladder-python.readthedocs.io/en/latest/user_guide/workflow.html``
"""

from source.triangle import Triangle
import chainladder as cl
import pandas as pd
import logging
from typing import Any, Optional, Tuple, Literal


class Reserving:
    def __init__(self, triangle: Triangle):
        self._triangle = triangle
        self.development = None
        self.tail = None
        self.bf = None
        self._chainladder_result = None
        self._bornhuetter_result = None
        self._bf_apriori_by_uwy: Optional[dict[str, float]] = None
        self._selected_ultimate_by_uwy: dict[str, str] = {}

    @staticmethod
    def _origin_to_uwy_label(origin: object) -> str:
        if hasattr(origin, "year"):
            return str(origin.year)
        origin_text = str(origin)
        if len(origin_text) >= 4 and origin_text[:4].isdigit():
            return origin_text[:4]
        return origin_text

    @staticmethod
    def _normalize_selected_method(method: object) -> str | None:
        value = str(method).strip().lower()
        if value in {"chainladder", "bornhuetter_ferguson"}:
            return value
        return None

    def set_development(
        self,
        average: str = "volume",
        drop: Optional[list] = None,
        drop_valuation: Optional[list] = None,
    ):
        if average not in ("volume", "simple"):
            raise ValueError(f"average must be 'volume' or 'simple', got '{average}'")

        # validate drop format
        if drop is not None:
            for i, item in enumerate(drop):
                if not isinstance(item, tuple):
                    raise ValueError(
                        f"drop[{i}] must be a tuple, got {type(item).__name__}"
                    )
                if len(item) != 2:
                    raise ValueError(
                        f"drop[{i}] must have exactly 2 elements, got {len(item)}"
                    )
                if not isinstance(item[0], str):
                    raise ValueError(
                        f"drop[{i}][0] must be a string (origin), got {type(item[0]).__name__}"
                    )
                if not isinstance(item[1], int):
                    raise ValueError(
                        f"drop[{i}][1] must be an integer (development period), got {type(item[1]).__name__}"
                    )

        # Validate drop_valuation format
        if isinstance(drop_valuation, str):
            drop_valuation = [drop_valuation]
        if drop_valuation is not None:
            if not isinstance(drop_valuation, list):
                raise ValueError(
                    f"drop_valuation must be a list, got {type(drop_valuation).__name__}"
                )
            for i, year in enumerate(drop_valuation):
                if not isinstance(year, str):
                    raise ValueError(
                        f"drop_valuation[{i}] must be a string (year), got {type(year).__name__}"
                    )

        params: dict[str, Any] = {
            "average": average,
        }
        if drop is not None:
            params["drop"] = drop
        if drop_valuation is not None:
            params["drop_valuation"] = drop_valuation

        self.development = cl.Development(**params)  # type: ignore[call-arg]

    def set_tail(
        self,
        curve: str = "weibull",
        attachment_age: Optional[int] = None,
        projection_period: Optional[int] = None,
        fit_period: Optional[Tuple[int, Optional[int]]] = None,
    ):
        if curve not in ("exponential", "inverse_power", "weibull"):
            raise ValueError(
                f"curve is {curve}, but has to be: 'exponential', 'inverse_power' or 'weibull'"
            )
        params: dict[str, Any] = {
            "curve": curve,
        }

        # Only add optional parameters if they're not None
        if attachment_age is not None:
            params["attachment_age"] = attachment_age
        if fit_period is not None:
            params["fit_period"] = fit_period
        if projection_period is not None:
            params["projection_period"] = projection_period

        self.tail = cl.TailCurve(**params)  # type: ignore[call-arg]

    def set_bornhuetter_ferguson(self, apriori: float | dict[str, float] = 0.6):
        if isinstance(apriori, dict):
            if len(apriori) == 0:
                raise ValueError("apriori mapping must not be empty")

            normalized: dict[str, float] = {}
            for origin, factor in apriori.items():
                key = str(origin)
                try:
                    factor_value = float(factor)
                except (TypeError, ValueError):
                    raise ValueError(
                        f"apriori factor for origin '{key}' must be numeric, got {factor}"
                    )
                if pd.isna(factor_value):
                    raise ValueError(
                        f"apriori factor for origin '{key}' must not be NaN"
                    )
                if factor_value < 0:
                    raise ValueError(
                        f"apriori factor for origin '{key}' must be >= 0, got {factor_value}"
                    )
                normalized[key] = factor_value

            self._bf_apriori_by_uwy = normalized
            self.bf = cl.BornhuetterFerguson(apriori=1.0)
            return

        try:
            apriori_value = float(apriori)
        except (TypeError, ValueError):
            raise ValueError(f"apriori must be numeric, got {apriori}")

        if pd.isna(apriori_value):
            raise ValueError("apriori must not be NaN")
        if apriori_value < 0:
            raise ValueError(f"apriori must be >= 0, got {apriori_value}")

        self._bf_apriori_by_uwy = None
        self.bf = cl.BornhuetterFerguson(apriori=apriori_value)

    def reserve(
        self,
        final_ultimate: Literal["chainladder", "bornhuetter_ferguson"] = "chainladder",
        selected_ultimate_by_uwy: Optional[dict[str, str]] = None,
    ):
        if self.development is None:
            raise ValueError(
                "Development estimator not set. Call set_development() first."
            )
        if self.tail is None:
            raise ValueError("Tail estimator not set. Call set_tail() first.")
        if self.bf is None:
            raise ValueError(
                "Bornhuetter-Ferguson estimator not set. Call set_bornhuetter_ferguson() first."
            )

        incurred = self._triangle.get_triangle().latest_diagonal["incurred"].to_frame()
        premium = (
            self._triangle.get_triangle().latest_diagonal["Premium_selected"].to_frame()
        )

        chainladder = self.chainladder()
        bornhuetter = self.bornhuetter_ferguson()
        self._chainladder_result = chainladder
        self._bornhuetter_result = bornhuetter

        self._triangle_transformed = chainladder.named_steps.dev.fit_transform(
            self._triangle.get_triangle()
        )
        cl_model = chainladder.named_steps.model
        bf_model = bornhuetter.named_steps.model

        cl_ultimate = cl_model.ultimate_["incurred"].to_frame()
        bf_ultimate = bf_model.ultimate_["incurred"].to_frame()

        cl_ultimate.columns = premium.columns
        bf_ultimate.columns = premium.columns

        # logging.info(f"Premium: /n{premium}")
        # logging.info(f"Cl ultimate: /n{cl_ultimate}")

        cl_loss_ratio = cl_ultimate / premium
        bf_loss_ratio = bf_ultimate / premium

        # logging.info(f"CL ultimates: {cl_ultimate}")
        # logging.info(f"CL loss ratios: {cl_loss_ratio}")

        cl_ultimate.columns = ["cl_ultimate"]
        cl_loss_ratio.columns = ["cl_loss_ratio"]
        bf_ultimate.columns = ["bf_ultimate"]
        bf_loss_ratio.columns = ["bf_loss_ratio"]

        incurred.columns = ["incurred"]
        premium.columns = ["Premium"]

        selected_mapping_input: dict[str, str] = {}
        if selected_ultimate_by_uwy:
            for key, value in selected_ultimate_by_uwy.items():
                method = self._normalize_selected_method(value)
                if method is None:
                    continue
                selected_mapping_input[str(key)] = method

        per_uwy_selection: dict[str, str] = {}
        ultimate_values = []
        for origin in cl_ultimate.index:
            uwy_label = self._origin_to_uwy_label(origin)
            selected_method = selected_mapping_input.get(uwy_label)
            if selected_method is None:
                selected_method = final_ultimate
            if selected_method not in {"chainladder", "bornhuetter_ferguson"}:
                raise ValueError(f"Unknown final_ultimate method: {selected_method}")

            per_uwy_selection[uwy_label] = selected_method
            if selected_method == "bornhuetter_ferguson":
                ultimate_values.append(bf_ultimate.loc[origin, "bf_ultimate"])
            else:
                ultimate_values.append(cl_ultimate.loc[origin, "cl_ultimate"])

        ultimate = pd.DataFrame(
            {"ultimate": ultimate_values},
            index=cl_ultimate.index,
        )
        self._selected_ultimate_by_uwy = per_uwy_selection

        selected_methods = set(per_uwy_selection.values())
        if selected_methods == {"bornhuetter_ferguson"}:
            self.result = bornhuetter
        else:
            self.result = chainladder

        self.correct_tail()

        self.df_results = cl_ultimate.copy().merge(
            cl_loss_ratio, right_index=True, left_index=True
        )
        self.df_results = self.df_results.merge(
            bf_ultimate, right_index=True, left_index=True
        )
        self.df_results = self.df_results.merge(
            bf_loss_ratio, right_index=True, left_index=True
        )
        self.df_results = self.df_results.merge(
            incurred, right_index=True, left_index=True
        )
        self.df_results = self.df_results.merge(
            premium, right_index=True, left_index=True
        )
        self.df_results = self.df_results.merge(
            ultimate, right_index=True, left_index=True
        )
        self.df_results["selected_method"] = [
            per_uwy_selection[self._origin_to_uwy_label(idx)]
            for idx in self.df_results.index
        ]
        # logging.info(
        #     f"ldfs: {self.result.named_steps.tail.ldf_['incurred'].to_frame().iloc[0]}"
        # )

    class CorrectTail:
        def fit(self, X, y=None):
            return self

        def transform(self, X):
            # set the final ldf factor (tail) to 1.0 for the 'incurred' triangle to prevent over-projection
            X_tail_corrected = X.copy()
            # Find the index of 'incurred' in the vdims
            incurred_idx = list(X.ldf_.vdims).index("incurred")
            logger = logging.getLogger(__name__)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("incurred index in vdims: %s", incurred_idx)
                logger.debug(
                    "Original tail LDFs: %s",
                    X_tail_corrected.ldf_.values[:, incurred_idx, :, -1],
                )
            # Set the tail link ratio (last development period) to 1.0 for 'incurred'
            X_tail_corrected.ldf_.values[:, incurred_idx, :, -1] = 1.0
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "Corrected tail LDFs: %s",
                    X_tail_corrected.ldf_.values[:, incurred_idx, :, -1],
                )
            return X_tail_corrected

    def correct_tail(self):
        incurred_idx = list(self.result.named_steps.tail.ldf_.vdims).index("incurred")
        self.result.named_steps.tail.ldf_.values[:, incurred_idx, :, -1] = 1.0

    def chainladder(self):
        pipe = cl.Pipeline(
            steps=[
                ("dev", self.development),
                ("tail", self.tail),
                ("correct_tail", self.CorrectTail()),
                ("model", cl.Chainladder()),
            ]
        )

        return pipe.fit(self._triangle.get_triangle())

    def bornhuetter_ferguson(self):
        exposure = self._triangle.get_triangle()["Premium_selected"].latest_diagonal
        exposure = self._apply_bf_apriori_to_exposure(exposure)
        incurred = self._triangle.get_triangle()["incurred"]

        pipe = cl.Pipeline(
            steps=[
                ("dev", self.development),
                (
                    "tail",
                    self.tail,
                ),  # optional, include if you need a tail to reach ultimate
                ("correct_tail", self.CorrectTail()),
                ("model", self.bf),
            ]
        )

        # incurred: cumulative Triangle
        # exposure: Triangle or vector aligned to origins (e.g., Earned Premium)
        return pipe.fit(incurred, model__sample_weight=exposure)

    def _apply_bf_apriori_to_exposure(self, exposure: cl.Triangle) -> cl.Triangle:
        if not self._bf_apriori_by_uwy:
            return exposure

        factors_by_origin: list[float] = []
        missing_origins: list[str] = []
        for origin in exposure.origin:
            key_candidates: list[str] = [str(origin)]
            if hasattr(origin, "year"):
                key_candidates.append(str(origin.year))

            factor = None
            for key in key_candidates:
                if key in self._bf_apriori_by_uwy:
                    factor = self._bf_apriori_by_uwy[key]
                    break

            if factor is None:
                missing_origins.append(str(origin))
                continue

            factors_by_origin.append(float(factor))

        if missing_origins:
            raise ValueError(
                "Missing BF apriori factors for origins: " + ", ".join(missing_origins)
            )

        adjusted = exposure.copy()
        adjusted_values = adjusted.values.copy()
        for i, factor in enumerate(factors_by_origin):
            adjusted_values[0, 0, i, 0] = adjusted_values[0, 0, i, 0] * factor
        adjusted.values = adjusted_values
        return adjusted

    def get_results(self):
        return self.df_results.copy()

    def get_emergence_pattern(self):
        # Calculate emergence triangle: for each UWY, show incurred as % of ultimate over development
        triangle_df = self._triangle.get_triangle()["incurred"].to_frame()

        # Emergence should always be measured against Chainladder ultimates.
        ultimate = self.df_results["cl_ultimate"]

        # Calculate emergence by dividing each row by its ultimate
        emergence = triangle_df.copy()
        for uwy in triangle_df.index:
            emergence.loc[uwy] = triangle_df.loc[uwy] / ultimate.loc[uwy]

        # Use Chainladder tail CDF for expected emergence regardless of selected results method.
        if self._chainladder_result is None:
            raise ValueError("Chainladder results not available. Call reserve() first.")
        cdf_values = (
            self._chainladder_result.named_steps.tail.cdf_["incurred"]
            .to_frame()
            .iloc[0]
        )
        # Cut off the last 4 tail periods to match triangle length
        cdf_values = cdf_values.iloc[:-4]
        expected_pattern = 1 / cdf_values

        # Align expected pattern with emergence columns
        expected_series = expected_pattern.reindex(emergence.columns)
        if expected_series.isna().all():
            expected_series = pd.Series(
                expected_pattern.values[: len(emergence.columns)],
                index=emergence.columns[: len(expected_pattern)],
            )
        expected_full = expected_series.reindex(emergence.columns)

        # Create expected dataframe with same structure as emergence
        expected = pd.DataFrame(
            [expected_full.values] * len(emergence),
            index=emergence.index,
            columns=emergence.columns,
        )

        # Unify results with multi-level columns
        result = pd.concat([emergence, expected], axis=1, keys=["Actual", "Expected"])
        return result

    def get_triangle_heatmap_data(self):
        """
        Extract link ratios, LDF, Tail, cumulative incurred, and premium data for heatmap visualization.
        Returns dictionary with:
            - 'link_ratios': DataFrame with link ratios for each UWY, plus LDF and Tail rows
            - 'incurred': DataFrame with cumulative incurred values
            - 'premium': DataFrame with premium values
        """
        if self.result is None:
            raise ValueError("Results not available. Call reserve() first")

        # Extract link ratio data
        link_ratios = self._triangle_transformed.link_ratio["incurred"].to_frame()
        ldfs = self.result.named_steps.dev.ldf_["incurred"].to_frame()
        tail = self.result.named_steps.tail.ldf_["incurred"].to_frame()

        # Add LDF row at the bottom of link_ratios
        ldf_row = ldfs.iloc[0].to_frame().T
        ldf_row.index = ["LDF"]

        # Add Tail row
        tail_row = tail.iloc[0].to_frame().T
        tail_row.index = ["Tail"]

        # Combine all rows
        link_ratios_with_ldf = pd.concat([link_ratios, ldf_row, tail_row])

        # Extract cumulative incurred and premium from triangle
        triangle_data = self._triangle.get_triangle()
        incurred_df = triangle_data["incurred"].to_frame()
        premium_df = triangle_data["Premium_selected"].to_frame()

        return {
            "link_ratios": link_ratios_with_ldf,
            "incurred": incurred_df,
            "premium": premium_df,
        }
