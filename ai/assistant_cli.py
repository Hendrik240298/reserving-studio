from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import chainladder as cl
import pandas as pd

from ai.assistant_service import AssistantService
from ai.env_loader import load_dotenv


def _load_rows_from_csv(path: Path) -> list[dict]:
    dataframe = pd.read_csv(path)
    return dataframe.to_dict(orient="records")


def _load_chainladder_sample_claim_rows() -> list[dict]:
    sample = cl.load_sample("quarterly")["incurred"]
    dataframe = sample.to_frame(keepdims=True).reset_index()

    origin_col = next(
        (col for col in ["origin", "uw_year"] if col in dataframe.columns), None
    )
    development_col = next(
        (col for col in ["development", "period"] if col in dataframe.columns),
        None,
    )
    value_col = "incurred" if "incurred" in dataframe.columns else dataframe.columns[-1]

    if origin_col is None or development_col is None:
        raise ValueError(
            "Could not infer origin/development columns from sample triangle"
        )

    dataframe = dataframe.rename(
        columns={
            origin_col: "uw_year",
            development_col: "period",
            value_col: "incurred",
        }
    )
    dataframe["uw_year"] = pd.to_datetime(dataframe["uw_year"], errors="coerce")
    dataframe["period"] = pd.to_datetime(dataframe["period"], errors="coerce")
    paid_numeric = pd.to_numeric(dataframe["incurred"], errors="coerce")
    if not isinstance(paid_numeric, pd.Series):
        paid_numeric = pd.Series(paid_numeric, index=dataframe.index)
    dataframe["paid"] = paid_numeric.fillna(0.0)
    dataframe["outstanding"] = 0.0
    return dataframe[["uw_year", "period", "incurred", "paid", "outstanding"]].to_dict(
        orient="records"
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    load_dotenv(".env")
    load_dotenv(".env.local")

    parser = argparse.ArgumentParser(
        description="OpenRouter-powered reserving assistant prototype"
    )
    parser.add_argument("--api-base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--segment", default="motor")
    parser.add_argument("--claims-csv", type=Path)
    parser.add_argument(
        "--premium-csv", type=Path, default=Path("data/quarterly_premium.csv")
    )
    parser.add_argument("--use-chainladder-sample", action="store_true")
    parser.add_argument("--granularity", default="quarterly")
    parser.add_argument(
        "--prompt",
        default=(
            "Run diagnostics and provide concise reserving commentary with evidence references."
        ),
    )
    args = parser.parse_args()

    if args.use_chainladder_sample:
        claims_rows = _load_chainladder_sample_claim_rows()
    elif args.claims_csv is not None:
        claims_rows = _load_rows_from_csv(args.claims_csv)
    else:
        raise ValueError("Provide --claims-csv or use --use-chainladder-sample")

    premium_rows = _load_rows_from_csv(args.premium_csv)

    assistant = AssistantService(api_base_url=args.api_base_url)
    workflow = assistant.bootstrap_workflow(
        segment=args.segment,
        claims_rows=claims_rows,
        premium_rows=premium_rows,
        granularity=args.granularity,
    )

    session_id = workflow["session_id"]
    enriched_prompt = (
        f"Session initialized with session_id={session_id}, segment={args.segment}. "
        f"Use tool_get_session first, run diagnostics, run iterative diagnostics search, "
        f"then recommend drops, BF apriori, and tail fit choices with evidence. "
        f"Original request: {args.prompt}"
    )
    answer = assistant.answer(user_prompt=enriched_prompt)

    print(json.dumps(workflow, indent=2, default=str))
    print("\n---\n")
    print(answer)


if __name__ == "__main__":
    main()
