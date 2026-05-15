from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from source.drop_analysis import _candidate_signals


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

    signals = _candidate_signals(
        link_ratios,
        baseline_drops=[],
        candidate_limit=3,
    )

    assert signals
    assert signals[0]["drop_pair"] == ("2022", 12)
    assert signals[0]["priority"] == "high"
