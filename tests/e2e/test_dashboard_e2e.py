from __future__ import annotations

import pytest
from playwright.sync_api import Page

from tests.e2e.helpers import (
    click_results_method_cell,
    edit_first_bf_apriori_value,
    graph_fingerprint,
    read_results_cell_style,
    read_results_column_values,
    trigger_first_valid_drop_click,
    wait_for_dashboard_ready,
    wait_for_graph_fingerprint_change,
    wait_for_graph_ready,
    wait_for_results_column_change,
    wait_for_results_table_ready,
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
    wait_for_results_table_ready(page)
    results_before = read_results_column_values(
        page,
        ["Selected Ultimate (EUR)", "IBNR (EUR)"],
    )

    page.click("#nav-chainladder")
    trigger_first_valid_drop_click(page)

    emergence_after = wait_for_graph_fingerprint_change(
        page,
        "emergence-plot",
        emergence_before,
    )
    assert emergence_after != emergence_before

    page.click("#nav-results")
    results_after = wait_for_results_column_change(
        page,
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
    wait_for_results_table_ready(page)
    baseline_results = read_results_column_values(
        page,
        ["BF Ultimate (EUR)", "Selected Ultimate (EUR)"],
    )

    page.click("#nav-bf")
    edit_first_bf_apriori_value(page, "3.0")

    page.click("#nav-results")
    updated_results = wait_for_results_column_change(
        page,
        baseline_results,
    )

    assert updated_results != baseline_results


@pytest.mark.e2e
def test_results_selection_is_per_uwy_and_persists(
    dash_base_url: str,
    page: Page,
) -> None:
    wait_for_dashboard_ready(page, dash_base_url)
    page.click("#nav-results")
    wait_for_results_table_ready(page)

    baseline = read_results_column_values(
        page,
        ["UWY", "CL Ultimate (EUR)", "BF Ultimate (EUR)", "Selected Ultimate (EUR)"],
    )
    differing_indices = [
        idx
        for idx, (cl_value, bf_value) in enumerate(
            zip(baseline["CL Ultimate (EUR)"], baseline["BF Ultimate (EUR)"])
        )
        if cl_value != bf_value
    ]
    assert len(differing_indices) >= 2

    first_idx = differing_indices[0]
    second_idx = differing_indices[1]
    first_uwy = baseline["UWY"][first_idx]
    second_uwy = baseline["UWY"][second_idx]

    click_results_method_cell(page, first_uwy, "bornhuetter_ferguson")
    after_first_click = wait_for_results_column_change(
        page,
        baseline,
    )
    assert (
        after_first_click["Selected Ultimate (EUR)"][first_idx]
        == after_first_click["BF Ultimate (EUR)"][first_idx]
    )
    assert (
        after_first_click["Selected Ultimate (EUR)"][second_idx]
        == after_first_click["CL Ultimate (EUR)"][second_idx]
    )

    click_results_method_cell(page, second_uwy, "bornhuetter_ferguson")
    after_second_click = wait_for_results_column_change(
        page,
        after_first_click,
    )
    assert (
        after_second_click["Selected Ultimate (EUR)"][second_idx]
        == after_second_click["BF Ultimate (EUR)"][second_idx]
    )

    click_results_method_cell(page, first_uwy, "chainladder")
    after_switch_back = wait_for_results_column_change(
        page,
        after_second_click,
    )
    assert (
        after_switch_back["Selected Ultimate (EUR)"][first_idx]
        == after_switch_back["CL Ultimate (EUR)"][first_idx]
    )

    page.reload(wait_until="domcontentloaded")
    wait_for_dashboard_ready(page, dash_base_url)
    page.click("#nav-results")
    wait_for_results_table_ready(page)
    after_reload = read_results_column_values(
        page,
        ["CL Ultimate (EUR)", "BF Ultimate (EUR)", "Selected Ultimate (EUR)"],
    )
    assert (
        after_reload["Selected Ultimate (EUR)"][first_idx]
        == after_reload["CL Ultimate (EUR)"][first_idx]
    )
    assert (
        after_reload["Selected Ultimate (EUR)"][second_idx]
        == after_reload["BF Ultimate (EUR)"][second_idx]
    )


@pytest.mark.e2e
def test_results_selection_uses_border_only_visual_indicator(
    dash_base_url: str,
    page: Page,
) -> None:
    wait_for_dashboard_ready(page, dash_base_url)
    page.click("#nav-results")
    wait_for_results_table_ready(page)

    baseline = read_results_column_values(
        page,
        ["UWY", "CL Ultimate (EUR)", "BF Ultimate (EUR)", "Selected Ultimate (EUR)"],
    )
    differing_indices = [
        idx
        for idx, (cl_value, bf_value) in enumerate(
            zip(baseline["CL Ultimate (EUR)"], baseline["BF Ultimate (EUR)"])
        )
        if cl_value != bf_value
    ]
    assert differing_indices
    target_idx = differing_indices[0]
    target_uwy = baseline["UWY"][target_idx]

    bg_before = read_results_cell_style(
        page,
        target_uwy,
        "bf_ultimate_display",
    )["background_color"]

    click_results_method_cell(page, target_uwy, "bornhuetter_ferguson")
    wait_for_results_column_change(page, baseline)

    bf_ultimate_style = read_results_cell_style(
        page,
        target_uwy,
        "bf_ultimate_display",
    )
    bf_lr_style = read_results_cell_style(
        page,
        target_uwy,
        "bf_loss_ratio_display",
    )

    assert bf_ultimate_style["background_color"] == bg_before
    assert bf_lr_style["background_color"] == bg_before
