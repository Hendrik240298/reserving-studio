from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlopen

import pytest
import yaml
from playwright.sync_api import (
    Browser,
    BrowserContext,
    Page,
    Playwright,
    sync_playwright,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = REPO_ROOT / "tests" / "artifacts" / "e2e"


def _wait_for_http_ready(url: str, timeout_seconds: float = 60.0) -> None:
    deadline = time.time() + timeout_seconds
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            with urlopen(url, timeout=1.5) as response:
                if response.status == 200:
                    return
        except Exception as exc:  # pragma: no cover - startup race handling
            last_error = exc
            time.sleep(0.25)
    raise RuntimeError(f"Dashboard server did not start at {url}: {last_error}")


def _is_port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.3)
        return sock.connect_ex((host, port)) == 0


def _get_free_port(host: str = "127.0.0.1") -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


@pytest.fixture(scope="session")
def dash_base_url(tmp_path_factory: pytest.TempPathFactory) -> str:
    host = "127.0.0.1"
    port = _get_free_port(host)
    base_url = f"http://{host}:{port}"

    runtime_root = tmp_path_factory.mktemp("dash-e2e-runtime")
    (runtime_root / "results").mkdir(parents=True, exist_ok=True)
    (runtime_root / "plots").mkdir(parents=True, exist_ok=True)
    (runtime_root / "data").mkdir(parents=True, exist_ok=True)
    (runtime_root / "sessions").mkdir(parents=True, exist_ok=True)

    config_payload = {
        "paths": {
            "results": str(runtime_root / "results") + "/",
            "plots": str(runtime_root / "plots") + "/",
            "data": str(runtime_root / "data") + "/",
            "sessions": str(runtime_root / "sessions") + "/",
        },
        "first date": 1900,
        "last date": "December 2006",
        "segment": "quarterly",
        "session": {
            "path": str(runtime_root / "sessions" / "quarterly.yml"),
        },
    }

    config_path = runtime_root / "config.e2e.yml"
    with config_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config_payload, file, sort_keys=False)

    env = os.environ.copy()
    env["RESERVING_CONFIG"] = str(config_path)
    env["PYTHONUNBUFFERED"] = "1"

    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "from source.app import load_config, build_sample_triangle, build_reserving; "
                "from source.dashboard import Dashboard; "
                "config = load_config(); "
                "triangle = build_sample_triangle(); "
                "reserving = build_reserving(triangle, config=config); "
                f"dashboard = Dashboard(reserving, config=config); dashboard.show(port={port})"
            ),
        ],
        cwd=str(REPO_ROOT),
        env=env,
    )
    try:
        _wait_for_http_ready(base_url)
        yield base_url
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:  # pragma: no cover - safety net
            process.kill()
            process.wait(timeout=5)


@pytest.fixture(scope="session")
def playwright_instance() -> Playwright:
    with sync_playwright() as playwright:
        yield playwright


@pytest.fixture(scope="session")
def browser(playwright_instance: Playwright) -> Browser:
    browser = playwright_instance.chromium.launch(headless=True)
    yield browser
    browser.close()


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo[object]):
    outcome = yield
    report = outcome.get_result()
    setattr(item, f"rep_{report.when}", report)


@pytest.fixture
def context(browser: Browser, request: pytest.FixtureRequest) -> BrowserContext:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    context = browser.new_context(viewport={"width": 1600, "height": 1000})
    context.tracing.start(screenshots=True, snapshots=True, sources=True)
    yield context

    test_name = request.node.name
    failed = bool(
        getattr(request.node, "rep_call", None) and request.node.rep_call.failed
    )
    trace_path = ARTIFACTS_DIR / f"{test_name}.zip"
    if failed:
        context.tracing.stop(path=str(trace_path))
    else:
        context.tracing.stop()
    context.close()


@pytest.fixture
def page(context: BrowserContext, request: pytest.FixtureRequest) -> Page:
    page = context.new_page()
    yield page
    failed = bool(
        getattr(request.node, "rep_call", None) and request.node.rep_call.failed
    )
    if failed:
        screenshot_path = ARTIFACTS_DIR / f"{request.node.name}.png"
        page.screenshot(path=str(screenshot_path), full_page=True)
    page.close()
