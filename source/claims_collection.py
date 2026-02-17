from __future__ import annotations

import pandas as pd


class ClaimsCollection:
    """
    Collection of all claims in long format
    """

    def __init__(self, df: pd.DataFrame, values_are_cumulative: bool = False):
        self._df = df.copy()
        self._values_are_cumulative = bool(values_are_cumulative)
        self._validate_required_columns()
        paid = pd.Series(
            pd.to_numeric(self._df["paid"], errors="coerce"), index=self._df.index
        )
        outstanding = pd.Series(
            pd.to_numeric(
                self._df["outstanding"],
                errors="coerce",
            ),
            index=self._df.index,
        )
        self._df["paid"] = paid.fillna(0.0)
        self._df["outstanding"] = outstanding.fillna(0.0)
        self._df["incurred"] = self._df["paid"] + self._df["outstanding"]

    def _validate_required_columns(self) -> None:
        required = {"uw_year", "period", "paid", "outstanding"}
        missing = required - set(self._df.columns)
        if missing:
            raise ValueError(f"Claims dataframe is missing required columns: {missing}")

    def _require_id_column(self, method_name: str) -> None:
        if "id" not in self._df.columns:
            raise ValueError(
                f"ClaimsCollection.{method_name} requires an 'id' column in the claims dataframe."
            )

    def iter_claims(self):
        """Make possible to iterate over claim df rather than object"""
        self._require_id_column("iter_claims")
        for id, df in self._df.groupby("id"):
            yield id, df

    def get_claim_amounts(self) -> pd.Series:
        """Returns the total incurred amount for each claim."""
        self._require_id_column("get_claim_amounts")
        amounts = self._df.groupby("id")["incurred"].sum()
        return pd.Series(amounts)

    def to_dataframe(self) -> pd.DataFrame:
        return self._df.copy()

    @property
    def values_are_cumulative(self) -> bool:
        return self._values_are_cumulative
