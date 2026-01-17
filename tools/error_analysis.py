#!/usr/bin/env python3
"""Error analysis comparing predictions to ground truth when available."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


def load_json(path: Path) -> List[Dict]:
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description="Error analysis for extraction outputs")
    parser.add_argument("--pred_json", type=Path, required=True, help="Predicted result.json")
    parser.add_argument("--gt_json", type=Path, required=True, help="Ground truth JSON")
    parser.add_argument("--output_dir", type=Path, required=True, help="Folder to write reports")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    preds = {p["doc_id"]: p for p in load_json(args.pred_json)}
    gts = {g["doc_id"]: g for g in load_json(args.gt_json)}

    rows = []
    for doc_id, gt in gts.items():
        pred = preds.get(doc_id)
        if pred is None:
            rows.append({"doc_id": doc_id, "error": "missing_prediction"})
            continue

        gt_fields = gt.get("fields", {})
        pr_fields = pred.get("fields", {})

        def mismatch(field: str) -> bool:
            return gt_fields.get(field) != pr_fields.get(field)

        rows.append(
            {
                "doc_id": doc_id,
                "dealer_mismatch": mismatch("dealer_name"),
                "model_mismatch": mismatch("model_name"),
                "hp_mismatch": mismatch("horse_power"),
                "cost_mismatch": mismatch("asset_cost"),
                "signature_mismatch": mismatch("signature"),
                "stamp_mismatch": mismatch("stamp"),
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(args.output_dir / "error_cases.csv", index=False)

    summary = {}
    for col in [
        "dealer_mismatch",
        "model_mismatch",
        "hp_mismatch",
        "cost_mismatch",
        "signature_mismatch",
        "stamp_mismatch",
    ]:
        if col in df:
            summary[col] = float(df[col].mean())

    (args.output_dir / "error_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
