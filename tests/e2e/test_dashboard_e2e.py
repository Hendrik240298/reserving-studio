from __future__ import annotations

import pytest
from playwright.sync_api import Page

from tests.e2e.helpers import (
    edit_first_bf_apriori_value,
    graph_fingerprint,
    trigger_first_valid_drop_click,
    wait_for_dashboard_ready,
    wait_for_graph_fingerprint_change,
    wait_for_graph_ready,
)


@pytest.mark.e2e
def test_drop_updates_emergence_and_results(
    dash_base_url: str,
    page: Page,
) -> None:
    wait_for_dashboard_ready(page, dash_base_url)

    page.click("#nav-chainladder")
    wait_for_graph_ready(page, "triangle-heatmap")
    wait_for_graph_ready(page, "emergence-plot")
    emergence_before = graph_fingerprint(page, "emergence-plot")

    page.click("#nav-results")
    wait_for_graph_ready(page, "results-table")
    results_before = graph_fingerprint(page, "results-table")

    page.click("#nav-chainladder")
    trigger_first_valid_drop_click(page)

    emergence_after = wait_for_graph_fingerprint_change(
        page,
        "emergence-plot",
        emergence_before,
    )
    assert emergence_after != emergence_before

    page.click("#nav-results")
    results_after = wait_for_graph_fingerprint_change(
        page,
        "results-table",
        results_before,
    )
    assert results_after != results_before


@pytest.mark.e2e
def test_bf_apriori_updates_bf_result_entries(
    dash_base_url: str,
    page: Page,
) -> None:
    wait_for_dashboard_ready(page, dash_base_url)

    page.click("#nav-results")
    wait_for_graph_ready(page, "results-table")
    baseline_results = graph_fingerprint(page, "results-table")

    page.click("#nav-bf")
    edit_first_bf_apriori_value(page, "3.0")

    page.click("#nav-results")
    updated_results = wait_for_graph_fingerprint_change(
        page,
        "results-table",
        baseline_results,
    )

    assert updated_results != baseline_results
