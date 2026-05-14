from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from source.api.adapters.reserving_adapter import (
    InMemoryReservingBackend,
    SessionContext,
)
from source.api.schemas import ParamsStore, ResultsStoreMeta


class _FakeReserving:
    def get_triangle_heatmap_data(self):
        link_ratios = pd.DataFrame(
            {
                12: [1.5, 2.2, 1.8],
                24: [1.3, 1.1, 1.7],
            },
            index=["2020", "2021", "2022"],
        )
        link_ratios.loc["LDF"] = [1.6, 1.2]
        link_ratios.loc["Tail"] = [1.0, 1.0]
        return {"link_ratios": link_ratios}


class _FakeReservingStringLabels:
    def get_triangle_heatmap_data(self):
        link_ratios = pd.DataFrame(
            {
                "3-6": [1.5, 2.2, 1.8],
                "6-9": [1.3, 1.1, 1.7],
            },
            index=["2020", "2021", "2022"],
        )
        link_ratios.loc["LDF"] = [1.6, 1.2]
        link_ratios.loc["Tail"] = [1.0, 1.0]
        return {"link_ratios": link_ratios}


def _context() -> SessionContext:
    return SessionContext(
        session_id="s-1",
        segment="seg",
        reserving=_FakeReserving(),
        sync_version=0,
        params_store=ParamsStore(
            request_id=0,
            source="test",
            force_recalc=False,
            drop_store=[],
            tail_attachment_age=None,
            tail_projection_months=0,
            tail_fit_period_selection=[],
            average="volume",
            tail_curve="weibull",
            bf_apriori_by_uwy={},
            selected_ultimate_by_uwy={},
            sync_version=0,
        ),
        results_store_meta=ResultsStoreMeta(sync_version=0),
        last_results_payload={},
    )


def test_link_ratio_rule_picks_max_per_development_period() -> None:
    backend = InMemoryReservingBackend()
    rows = backend._selected_link_ratio_rows(
        _context(),
        selection_mode="max",
        scope="per_development_period",
        limit=10,
    )

    assert rows == [
        {"development_period": 12, "origin": "2021", "a2a": 2.2},
        {"development_period": 24, "origin": "2022", "a2a": 1.7},
    ]


def test_link_ratio_rule_supports_global_ranking() -> None:
    backend = InMemoryReservingBackend()
    rows = backend._selected_link_ratio_rows(
        _context(),
        selection_mode="max",
        scope="global",
        limit=3,
    )

    assert rows == [
        {"development_period": 12, "origin": "2021", "a2a": 2.2},
        {"development_period": 12, "origin": "2022", "a2a": 1.8},
        {"development_period": 24, "origin": "2022", "a2a": 1.7},
    ]


def test_link_ratio_rule_parses_string_interval_labels() -> None:
    backend = InMemoryReservingBackend()
    context = _context()
    context.reserving = _FakeReservingStringLabels()

    rows = backend._selected_link_ratio_rows(
        context,
        selection_mode="max",
        scope="per_development_period",
        limit=10,
    )

    assert rows == [
        {"development_period": 3, "origin": "2021", "a2a": 2.2},
        {"development_period": 6, "origin": "2022", "a2a": 1.7},
    ]


def test_link_ratio_rule_supports_threshold_filter() -> None:
    backend = InMemoryReservingBackend()
    rows = backend._selected_link_ratio_rows(
        _context(),
        selection_mode="min",
        scope="global",
        limit=10,
        threshold_operator="lt",
        threshold_value=1.6,
    )

    assert rows == [
        {"development_period": 24, "origin": "2021", "a2a": 1.1},
        {"development_period": 24, "origin": "2020", "a2a": 1.3},
        {"development_period": 12, "origin": "2020", "a2a": 1.5},
    ]


def test_invalid_drop_pairs_are_filtered_against_triangle() -> None:
    backend = InMemoryReservingBackend()
    filtered = backend._filter_valid_drop_pairs(
        _context(),
        [("9999", 12), ("2021", 999), ("2021", 12), ("2021", 12)],
    )

    assert filtered == [("2021", 12)]
