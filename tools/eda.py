#!/usr/bin/env python3
"""Generate lightweight EDA plots from result.json and optional metadata."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def load_results(path: Path) -> List[Dict]:
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description="EDA for extraction outputs")
    parser.add_argument("--results_json", type=Path, required=True, help="Path to result.json")
    parser.add_argument("--output_dir", type=Path, required=True, help="Folder to write plots")
    parser.add_argument(
        "--metadata_csv",
        type=Path,
        default=None,
        help="Optional CSV with columns: doc_id,state,language",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = load_results(args.results_json)

    rows = []
    for item in results:
        fields = item.get("fields", {})
        rows.append(
            {
                "doc_id": item.get("doc_id"),
                "confidence": item.get("confidence"),
                "processing_time_sec": item.get("processing_time_sec"),
                "dealer_present": bool(fields.get("dealer_name")),
                "model_present": bool(fields.get("model_name")),
                "hp_present": fields.get("horse_power") is not None,
                "cost_present": fields.get("asset_cost") is not None,
                "signature_present": bool(fields.get("signature", {}).get("present")),
                "stamp_present": bool(fields.get("stamp", {}).get("present")),
            }
        )

    df = pd.DataFrame(rows)

    # Processing time distribution
    plt.figure(figsize=(6, 4))
    sns.histplot(df["processing_time_sec"].dropna(), bins=20)
    plt.title("Processing Time Distribution")
    plt.xlabel("Seconds")
    plt.tight_layout()
    plt.savefig(args.output_dir / "processing_time_hist.png", dpi=200)
    plt.close()

    # Confidence distribution
    plt.figure(figsize=(6, 4))
    sns.histplot(df["confidence"].dropna(), bins=20)
    plt.title("Confidence Distribution")
    plt.xlabel("Confidence")
    plt.tight_layout()
    plt.savefig(args.output_dir / "confidence_hist.png", dpi=200)
    plt.close()

    # Field presence bar chart
    presence_cols = [
        "dealer_present",
        "model_present",
        "hp_present",
        "cost_present",
        "signature_present",
        "stamp_present",
    ]
    presence = df[presence_cols].mean().sort_values(ascending=False) * 100
    plt.figure(figsize=(7, 4))
    sns.barplot(x=presence.values, y=presence.index)
    plt.title("Field Presence Rate (%)")
    plt.xlabel("% of docs")
    plt.tight_layout()
    plt.savefig(args.output_dir / "field_presence.png", dpi=200)
    plt.close()

    # Optional metadata plots
    if args.metadata_csv and args.metadata_csv.exists():
        meta = pd.read_csv(args.metadata_csv)
        merged = df.merge(meta, on="doc_id", how="left")
        if "state" in merged.columns:
            plt.figure(figsize=(7, 4))
            merged["state"].value_counts().plot(kind="bar")
            plt.title("State Distribution")
            plt.tight_layout()
            plt.savefig(args.output_dir / "state_distribution.png", dpi=200)
            plt.close()
        if "language" in merged.columns:
            plt.figure(figsize=(7, 4))
            merged["language"].value_counts().plot(kind="bar")
            plt.title("Language Distribution")
            plt.tight_layout()
            plt.savefig(args.output_dir / "language_distribution.png", dpi=200)
            plt.close()


if __name__ == "__main__":
    main()
