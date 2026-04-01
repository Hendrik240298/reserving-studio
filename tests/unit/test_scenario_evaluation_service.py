from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from source.api.schemas import RunMetadata
from source.services.diagnostics_service import DiagnosticFinding
from source.services.scenario_evaluation_service import ScenarioEvaluationService


def test_map_finding_generates_evidence_id_when_missing() -> None:
    run_metadata = RunMetadata(
        run_id="run-1",
        generated_at=datetime.now(timezone.utc),
        data_fingerprint="fp-1",
        diagnostics_version="v2.2",
        scenario_generator_version="v1.2",
    )
    finding = DiagnosticFinding(
        code="TEST_FINDING",
        severity="medium",
        message="Finding message",
        evidence={"metric_id": "test_metric", "value": 1.2},
        suggested_actions=[],
    )

    mapped = ScenarioEvaluationService.map_finding(finding, run_metadata=run_metadata)

    assert mapped.evidence.evidence_id
    assert mapped.evidence.diagnostic_version == "v2.2"


def test_data_fingerprint_is_stable_for_same_payload() -> None:
    results_df = pd.DataFrame(
        {"incurred": [100.0], "ultimate": [120.0]},
        index=pd.Index(["2022"]),
    )
    incurred = pd.DataFrame({12: [100.0]}, index=pd.Index(["2022"]))
    link_ratios = pd.DataFrame({12: [1.2]}, index=pd.Index(["2022"]))
    link_ratios.loc["LDF"] = [1.2]

    first = ScenarioEvaluationService.data_fingerprint(
        results_df=results_df,
        heatmap_data={"incurred": incurred, "link_ratios": link_ratios},
    )
    second = ScenarioEvaluationService.data_fingerprint(
        results_df=results_df.copy(),
        heatmap_data={"incurred": incurred.copy(), "link_ratios": link_ratios.copy()},
    )

    assert first == second
