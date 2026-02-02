#!/usr/bin/env python3
"""
LLM Inference Entry Point

Run LLM-based causal relationship classification on gene-disease pairs.

Usage:
    python run_llm_inference.py --input_file data.tsv --output_dir results/
"""

import os
import argparse
import datetime
import pandas as pd

from src.config import BASE_DIR
from src.utils import group_evidence_data


def main():
    parser = argparse.ArgumentParser(
        description='LLM-based causal relationship inference'
    )
    parser.add_argument(
        '--input_file', type=str,
        required=True,
        help='Input TSV file with sentence-level data'
    )
    parser.add_argument(
        '--output_dir', type=str,
        default=os.path.join(BASE_DIR, 'data', 'results'),
        help='Output directory for results'
    )
    parser.add_argument(
        '--model_name', type=str,
        default='MMed-Llama-3-8B',
        help='Model name for output file naming'
    )

    args = parser.parse_args()

    # Check for LLM dependencies
    try:
        from src.inference import process_all_pairs
    except ImportError as e:
        print(f"Error: LLM dependencies not installed.")
        print(f"Install with: pip install openai instructor tiktoken pydantic")
        return

    # Load data
    if not os.path.exists(args.input_file):
        raise FileNotFoundError(f"Could not find {args.input_file}")

    print(f"Loading data from {args.input_file}...")
    data = pd.read_csv(args.input_file, sep='\t')

    # Group evidence by gene-disease pairs
    grouped_data = group_evidence_data(data)
    print(f"Processing {len(grouped_data)} unique gene-disease pairs...")

    # Setup output paths
    os.makedirs(args.output_dir, exist_ok=True)
    datestamp = datetime.datetime.now().strftime("%Y-%m-%d")
    data_name = os.path.splitext(os.path.basename(args.input_file))[0]

    summary_path = os.path.join(
        args.output_dir,
        f"{data_name}_{args.model_name}_llm_results_{datestamp}.csv"
    )
    log_path = os.path.join(
        args.output_dir,
        f"{data_name}_{args.model_name}_raw_responses_{datestamp}.jsonl"
    )

    # Clear existing output files
    for f in [summary_path, log_path]:
        if os.path.exists(f):
            os.remove(f)

    # Run inference
    start_time = datetime.datetime.now()
    results = process_all_pairs(grouped_data, log_path=log_path)

    # Save results
    df_results = pd.DataFrame(results)
    cols_order = [
        "id1", "id2", "original_label",
        "llm_relationship", "llm_score", "llm_reasoning", "error_msg"
    ]

    for c in cols_order:
        if c not in df_results.columns:
            df_results[c] = None

    df_results = df_results[cols_order]
    df_results.to_csv(summary_path, index=False)

    print(f"\nDone! Time: {datetime.datetime.now() - start_time}")
    print(f"CSV: {summary_path}")
    print(f"Logs: {log_path}")

    if not df_results.empty:
        causal_count = (df_results['llm_relationship'] == 'Causal').sum()
        print(f"Causal found: {causal_count}")


if __name__ == "__main__":
    main()
