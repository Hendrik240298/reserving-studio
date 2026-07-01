from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harness.drop_review import analyze_candidates


class ReservingStub:
    def __init__(self, *, ldf: pd.Series, link_ratios: pd.DataFrame) -> None:
        self._ldf = ldf
        self._link_ratios = link_ratios

    def get_ldf(self) -> pd.Series:
        return self._ldf

    def get_link_ratios(self) -> pd.DataFrame:
        return self._link_ratios


def test_candidate_signals_identify_outlier_drop_pair() -> None:
    link_ratios = pd.DataFrame(
        {
            "12-24": [1.50, 1.52, 1.49, 2.40],
            "24-36": [1.18, 1.17, 1.19, None],
        },
        index=pd.Index(["2019", "2020", "2021", "2022"]),
    )
    link_ratios.loc["LDF"] = [1.51, 1.18]
    link_ratios.loc["Tail"] = [1.51, 1.18]

    signals = analyze_candidates(
        ReservingStub(
            ldf=link_ratios.loc["LDF"],
            link_ratios=link_ratios.drop(index=["LDF", "Tail"]),
        ),
        candidate_limit=3,
    )

    assert not signals.empty
    assert signals.iloc[0]["origin"] == "2022"
    assert signals.iloc[0]["period"] == "12-24"
    assert signals.iloc[0]["priority"] == 1
