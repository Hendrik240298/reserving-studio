from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harness.triangle_markdown import _build_triangle, render_triangle_markdown


def test_render_triangle_markdown_marks_bolted_a2a_cells() -> None:
    result = render_triangle_markdown(
        config_path=Path("examples/config_quarterly.yml"),
        triangle_type="a2a",
        bolted_drops=[("2002", 21), ("2003", 9)],
    )

    assert result.triangle_type == "a2a"
    assert result.triangle_view == "cumulative"
    assert "| Origin | 3 | 6 | 9 | 12 |" in result.markdown
    assert "~~**1.464**~~" in result.markdown
    assert "~~**7.689**~~" in result.markdown


def test_render_triangle_markdown_supports_premium_alias() -> None:
    result = render_triangle_markdown(
        config_path=Path("examples/config_quarterly.yml"),
        triangle_type="premium",
    )

    assert result.triangle_type == "Premium_selected"
    assert result.triangle_view == "cumulative"
    assert "| Origin |" in result.markdown
    assert "1995" in result.markdown


def test_render_triangle_markdown_supports_yearly_headers() -> None:
    result = render_triangle_markdown(
        config_path=Path("examples/config_clrd.yml"),
        triangle_type="incurred",
    )

    assert result.triangle_view == "cumulative"
    assert "| Origin | 12 | 24 | 36 | 48 |" in result.markdown
    assert "154,058.00" in result.markdown


def test_render_triangle_markdown_supports_incremental_view() -> None:
    cumulative = render_triangle_markdown(
        config_path=Path("examples/config_quarterly.yml"),
        triangle_type="incurred",
        triangle_view="cumulative",
    )
    incremental = render_triangle_markdown(
        config_path=Path("examples/config_quarterly.yml"),
        triangle_type="incurred",
        triangle_view="incremental",
    )

    assert incremental.triangle_view == "incremental"
    assert cumulative.markdown != incremental.markdown


def test_render_triangle_markdown_rejects_incremental_a2a() -> None:
    with pytest.raises(ValueError, match="a2a"):
        render_triangle_markdown(
            config_path=Path("examples/config_quarterly.yml"),
            triangle_type="a2a",
            triangle_view="incremental",
        )


def test_build_triangle_matches_core_origin_filtering() -> None:
    claims_df = pd.DataFrame(
        {
            "id": ["c1", "c2"],
            "uw_year": pd.to_datetime(["2000-01-01", "2000-01-01"]),
            "period": pd.to_datetime(["2000-03-31", "2000-06-30"]),
            "paid": [10.0, 5.0],
            "outstanding": [0.0, 0.0],
        }
    )
    premium_df = pd.DataFrame(
        {
            "uw_year": pd.to_datetime(["2000-01-01", "2001-01-01"]),
            "period": pd.to_datetime(["2000-03-31", "2001-03-31"]),
            "Premium_selected": [100.0, 999.0],
        }
    )

    triangle = _build_triangle(claims_df, premium_df, config=None)
    premium_triangle = triangle.get_triangle("Premium_selected")["Premium_selected"]
    origins = [
        getattr(origin, "year", str(origin))
        for origin in premium_triangle.origin.tolist()
    ]

    assert origins == [2000]
