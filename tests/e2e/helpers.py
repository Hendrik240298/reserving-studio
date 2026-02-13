from __future__ import annotations

import hashlib
import json
import time

from playwright.sync_api import Page


def wait_for_dashboard_ready(page: Page, base_url: str) -> None:
    page.goto(base_url, wait_until="domcontentloaded")
    page.wait_for_selector("#nav-data", state="visible", timeout=20_000)
    wait_for_graph_ready(page, "triangle-heatmap")
    wait_for_graph_ready(page, "emergence-plot")


def wait_for_graph_ready(page: Page, graph_id: str, timeout_ms: int = 20_000) -> None:
    page.wait_for_function(
        """
        (id) => {
            const root = document.getElementById(id);
            if (!root) {
                return false;
            }
            const gd = root.classList.contains("js-plotly-plot")
                ? root
                : root.querySelector(".js-plotly-plot");
            return Boolean(gd && Array.isArray(gd.data) && gd.data.length > 0);
        }
        """,
        arg=graph_id,
        timeout=timeout_ms,
    )


def wait_for_results_table_ready(page: Page, timeout_ms: int = 20_000) -> None:
    page.wait_for_selector(
        '#results-table td[data-dash-column="uwy"]',
        state="visible",
        timeout=timeout_ms,
    )


def get_graph_data(page: Page, graph_id: str) -> list[dict]:
    payload = page.evaluate(
        """
        (id) => {
            const root = document.getElementById(id);
            if (!root) {
                return [];
            }
            const gd = root.classList.contains("js-plotly-plot")
                ? root
                : root.querySelector(".js-plotly-plot");
            if (!gd || !Array.isArray(gd.data)) {
                return [];
            }
            return JSON.parse(JSON.stringify(gd.data));
        }
        """,
        graph_id,
    )
    if not isinstance(payload, list):
        return []
    return payload


def graph_fingerprint(page: Page, graph_id: str) -> str:
    data = get_graph_data(page, graph_id)
    blob = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def wait_for_graph_fingerprint_change(
    page: Page,
    graph_id: str,
    old_fingerprint: str,
    timeout_ms: int = 20_000,
) -> str:
    deadline = time.time() + (timeout_ms / 1000)
    while time.time() < deadline:
        current = graph_fingerprint(page, graph_id)
        if current != old_fingerprint:
            return current
        page.wait_for_timeout(150)
    raise AssertionError(f"Graph '{graph_id}' did not update before timeout")


def trigger_first_valid_drop_click(page: Page) -> dict[str, str]:
    payload = page.evaluate(
        """
        () => {
            const root = document.getElementById("triangle-heatmap");
            if (!root) {
                return null;
            }
            const gd = root.classList.contains("js-plotly-plot")
                ? root
                : root.querySelector(".js-plotly-plot");
            if (!gd || !Array.isArray(gd.data) || gd.data.length === 0) {
                return null;
            }

            const heatmap = gd.data.find((trace) => trace && trace.type === "heatmap") || gd.data[0];
            const xValues = Array.isArray(heatmap.x) ? heatmap.x.map(String) : [];
            const yValues = Array.isArray(heatmap.y) ? heatmap.y.map(String) : [];

            const targetX = xValues.find((value) => value !== "UWY");
            const targetY = yValues.find(
                (value) => value !== "Dev" && value !== "LDF" && value !== "Tail"
            );

            if (!targetX || !targetY) {
                return null;
            }

            if (typeof gd.emit === "function") {
                gd.emit("plotly_click", {
                    points: [
                        {
                            x: targetX,
                            y: targetY,
                            curveNumber: 0,
                            pointNumber: 0,
                        },
                    ],
                });
                return { x: targetX, y: targetY };
            }

            return null;
        }
        """
    )
    if not isinstance(payload, dict):
        raise AssertionError("Could not trigger heatmap drop click")
    x = str(payload.get("x", "")).strip()
    y = str(payload.get("y", "")).strip()
    if not x or not y:
        raise AssertionError("Heatmap click payload is missing x/y")
    return {"x": x, "y": y}


def read_results_row_values(page: Page, headers: list[str]) -> dict[str, str]:
    values = page.evaluate(
        """
        (requestedHeaders) => {
            const headerToColumn = {
                "UWY": "uwy",
                "Incurred (EUR)": "incurred_display",
                "Premium (EUR)": "premium_display",
                "Incurred Loss Ratio": "incurred_loss_ratio_display",
                "CL Ultimate (EUR)": "cl_ultimate_display",
                "CL Loss Ratio": "cl_loss_ratio_display",
                "BF Ultimate (EUR)": "bf_ultimate_display",
                "BF Loss Ratio": "bf_loss_ratio_display",
                "Selected Ultimate (EUR)": "ultimate_display",
                "IBNR (EUR)": "ibnr_display",
            };

            const output = {};
            for (const header of requestedHeaders) {
                const column = headerToColumn[header];
                if (!column) {
                    output[header] = "";
                    continue;
                }
                const cell = document.querySelector(`#results-table td[data-dash-row="0"][data-dash-column="${column}"]`);
                output[header] = cell ? String(cell.textContent || "").trim() : "";
            }
            return output;
        }
        """,
        headers,
    )
    if not isinstance(values, dict):
        raise AssertionError("Could not read results table values")
    return {str(key): str(value) for key, value in values.items()}


def read_results_column_values(page: Page, headers: list[str]) -> dict[str, list[str]]:
    values = page.evaluate(
        """
        (requestedHeaders) => {
            const headerToColumn = {
                "UWY": "uwy",
                "Incurred (EUR)": "incurred_display",
                "Premium (EUR)": "premium_display",
                "Incurred Loss Ratio": "incurred_loss_ratio_display",
                "CL Ultimate (EUR)": "cl_ultimate_display",
                "CL Loss Ratio": "cl_loss_ratio_display",
                "BF Ultimate (EUR)": "bf_ultimate_display",
                "BF Loss Ratio": "bf_loss_ratio_display",
                "Selected Ultimate (EUR)": "ultimate_display",
                "IBNR (EUR)": "ibnr_display",
            };
            const output = {};

            for (const header of requestedHeaders) {
                const column = headerToColumn[header];
                if (!column) {
                    output[header] = [];
                    continue;
                }
                const cells = Array.from(document.querySelectorAll(`#results-table td[data-dash-column="${column}"]`));
                output[header] = cells.map((cell) => String(cell.textContent || "").trim());
            }
            return output;
        }
        """,
        headers,
    )
    if not isinstance(values, dict):
        raise AssertionError("Could not read results table columns")
    normalized: dict[str, list[str]] = {}
    for key, raw in values.items():
        if isinstance(raw, list):
            normalized[str(key)] = [str(item) for item in raw]
        else:
            normalized[str(key)] = []
    return normalized


def wait_for_results_row_change(
    page: Page,
    previous_values: dict[str, str],
    timeout_ms: int = 20_000,
) -> dict[str, str]:
    headers = list(previous_values.keys())
    deadline = time.time() + (timeout_ms / 1000)
    while time.time() < deadline:
        current_values = read_results_row_values(page, headers)
        if any(
            current_values.get(key, "") != previous_values.get(key, "")
            for key in headers
        ):
            return current_values
        page.wait_for_timeout(150)
    raise AssertionError("Results row did not update before timeout")


def wait_for_results_column_change(
    page: Page,
    previous_values: dict[str, list[str]],
    timeout_ms: int = 20_000,
) -> dict[str, list[str]]:
    headers = list(previous_values.keys())
    deadline = time.time() + (timeout_ms / 1000)
    while time.time() < deadline:
        current_values = read_results_column_values(page, headers)
        if any(
            current_values.get(key, []) != previous_values.get(key, [])
            for key in headers
        ):
            return current_values
        page.wait_for_timeout(150)
    raise AssertionError("Results columns did not update before timeout")


def edit_first_bf_apriori_value(page: Page, new_value: str) -> None:
    cell = page.locator('#bf-apriori-table td[data-dash-column="apriori"]').last
    cell.wait_for(state="visible", timeout=10_000)
    cell.click()

    input_box = page.locator(
        '#bf-apriori-table td[data-dash-column="apriori"] input'
    ).last
    input_box.wait_for(state="visible", timeout=10_000)
    input_box.fill(new_value)
    input_box.press("Enter")
    page.keyboard.press("Tab")


def click_results_method_cell(page: Page, uwy: str, method: str) -> None:
    normalized = method.strip().lower()
    if normalized == "chainladder":
        column = "cl_ultimate_display"
    elif normalized == "bornhuetter_ferguson":
        column = "bf_ultimate_display"
    else:
        raise AssertionError(f"Unknown method: {method}")

    row_index = page.evaluate(
        """
        (targetUwy) => {
            const cells = Array.from(document.querySelectorAll('#results-table td[data-dash-column="uwy"]'));
            for (const cell of cells) {
                const text = String(cell.textContent || '').trim();
                if (text === targetUwy) {
                    const row = cell.getAttribute('data-dash-row');
                    return row === null ? -1 : parseInt(row, 10);
                }
            }
            return -1;
        }
        """,
        uwy,
    )
    if not isinstance(row_index, int) or row_index < 0:
        raise AssertionError(f"Could not find UWY row {uwy}")

    target_cell = page.locator(
        f'#results-table td[data-dash-row="{row_index}"][data-dash-column="{column}"]'
    )
    target_cell.wait_for(state="visible", timeout=10_000)
    target_cell.click()


def read_results_cell_style(page: Page, uwy: str, column: str) -> dict[str, str]:
    style_payload = page.evaluate(
        """
        ({ targetUwy, targetColumn }) => {
            const uwyCells = Array.from(document.querySelectorAll('#results-table td[data-dash-column="uwy"]'));
            let rowIndex = -1;
            for (const cell of uwyCells) {
                const text = String(cell.textContent || '').trim();
                if (text === targetUwy) {
                    const row = cell.getAttribute('data-dash-row');
                    rowIndex = row === null ? -1 : parseInt(row, 10);
                    break;
                }
            }
            if (rowIndex < 0) {
                return null;
            }

            const selector = `#results-table td[data-dash-row="${rowIndex}"][data-dash-column="${targetColumn}"]`;
            const targetCell = document.querySelector(selector);
            if (!targetCell) {
                return null;
            }
            const style = window.getComputedStyle(targetCell);
            return {
                background_color: style.backgroundColor || '',
                border_top_color: style.borderTopColor || '',
                border_right_color: style.borderRightColor || '',
                border_bottom_color: style.borderBottomColor || '',
                border_left_color: style.borderLeftColor || '',
                border_top_width: style.borderTopWidth || '',
                border_right_width: style.borderRightWidth || '',
                border_bottom_width: style.borderBottomWidth || '',
                border_left_width: style.borderLeftWidth || '',
                outline_color: style.outlineColor || '',
                outline_style: style.outlineStyle || '',
                outline_width: style.outlineWidth || '',
                box_shadow: style.boxShadow || '',
            };
        }
        """,
        {"targetUwy": uwy, "targetColumn": column},
    )
    if not isinstance(style_payload, dict):
        raise AssertionError(f"Could not read style for UWY {uwy}, column {column}")
    return {str(key): str(value) for key, value in style_payload.items()}
