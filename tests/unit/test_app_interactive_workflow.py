from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from source.app import wait_for_finalize
from source.interactive_session import (
    FinalizePayload,
    InteractiveSessionController,
    ParamsStoreSnapshot,
    ResultsStoreSnapshot,
    utc_now,
)


def _build_payload() -> FinalizePayload:
    return FinalizePayload(
        finalized_at_utc=utc_now(),
        segment="quarterly",
        params_store=ParamsStoreSnapshot.from_store_dict({}),
        results_store=ResultsStoreSnapshot.from_store_dict({}),
        results_df=pd.DataFrame({"ultimate": [100.0]}),
    )


def test_wait_for_finalize_returns_payload_when_finalized() -> None:
    controller = InteractiveSessionController()
    payload = _build_payload()
    controller.finalize(payload)

    finalized = wait_for_finalize(controller, timeout_seconds=0.2)

    assert finalized.segment == "quarterly"
    assert float(finalized.results_df.iloc[0]["ultimate"]) == 100.0


def test_wait_for_finalize_raises_on_timeout() -> None:
    controller = InteractiveSessionController()

    with pytest.raises(TimeoutError):
        wait_for_finalize(controller, timeout_seconds=0.01)


def test_wait_for_finalize_raises_on_failure() -> None:
    controller = InteractiveSessionController()
    controller.fail("boom")

    with pytest.raises(RuntimeError, match="Interactive session failed"):
        wait_for_finalize(controller, timeout_seconds=0.2)
