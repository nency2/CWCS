#!/usr/bin/env python3
"""
CWCS Pipeline Entry Point

Run the complete Corpus-Wide Causal Scoring pipeline.

Usage:
    python run_pipeline.py --input_file data.tsv --output_dir results/
"""

from src.pipeline import main

if __name__ == "__main__":
    main()
