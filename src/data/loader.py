"""
Data loading utilities for CWCS pipeline.
"""

import os
import pandas as pd
import numpy as np
from ..config import BETA, LAM


def load_unified_data(file_path: str, min_papers: int = 1) -> pd.DataFrame:
    """
    Load and process bibliometric data, calculating VQ* scores.

    Args:
        file_path: Path to the TSV file containing bibliometric data
        min_papers: Minimum number of papers required for a gene-disease pair

    Returns:
        DataFrame with processed data including VQ* scores
    """
    print(f"  Loading Unified Text Data: {file_path}")
    df = pd.read_csv(file_path, sep='\t')
    print(f"  Columns: {list(df.columns)}")

    # Cleaning & Type Casting
    if 'id1' not in df.columns:
        raise ValueError("Missing id1 column")
    df['id1'] = df['id1'].astype(str).str.replace(r'\.0$', '', regex=True)

    # Fix H-Index (Extract numeric)
    if 'hindex' in df.columns:
        df['hindex'] = pd.to_numeric(
            df['hindex'].astype(str).str.split().str[0],
            errors='coerce'
        )

    # Fix Citations
    if 'citations' in df.columns:
        df['citations'] = pd.to_numeric(df['citations'], errors='coerce')

    # Fix Year
    if 'year' in df.columns:
        df['year'] = pd.to_numeric(df['year'], errors='coerce')
    elif 'Clean_Year' in df.columns:
        df['year'] = df['Clean_Year']
    elif 'year_diff' in df.columns:
        df['year'] = 2025 - df['year_diff']

    # Drop NaNs
    original_len = len(df)
    df.dropna(subset=['hindex', 'citations', 'year'], inplace=True)
    if len(df) < original_len:
        print(f"  Dropped {original_len - len(df)} rows due to missing values.")

    # Feature Normalization
    df['year_diff_norm'] = (2025 - df['year']) / (2025 - df['year'].min())

    # Min-Max Scaling
    df['hindex'] = df['hindex'].astype('float64')
    df['citations_scaled'] = (df['citations'] - df['citations'].min()) / (df['citations'].max() - df['citations'].min())
    df['hindex_scaled'] = (df['hindex'] - df['hindex'].min()) / (df['hindex'].max() - df['hindex'].min())

    # RS Calculation
    f1 = df['hindex_scaled'].values
    f2 = df['citations_scaled'].values
    f3 = df['year_diff_norm'].values
    fs = np.vstack((f1, f2, f3)).T
    df['rs'] = fs @ BETA

    # VQ* Calculation
    if 'pred_label' in df.columns:
        df['Prediction'] = df['pred_label']
    elif 'Prediction' not in df.columns:
        df['Prediction'] = 0

    gd_pair_dict = {}
    vq_star_dict = {}

    for _, row in df.iterrows():
        gd_pair = (row.get('disease', 'unknown'), str(row['id1']))

        if gd_pair not in gd_pair_dict:
            gd_pair_dict[gd_pair] = []
        gd_pair_dict[gd_pair].append((row['rs'], row['Prediction']))

    for gd_pair, src_list in gd_pair_dict.items():
        rs_values = np.array([x[0] for x in src_list])
        vqs_values = np.array([x[1] for x in src_list]).astype(float)

        numerator = (np.sum(rs_values * (vqs_values == 1).astype(float)) +
                     LAM * np.sum(rs_values * (vqs_values == 0).astype(float) * vqs_values))

        denominator = (np.sum(rs_values * (vqs_values == 1).astype(float)) +
                       LAM * np.sum(rs_values * (vqs_values == 0).astype(float)))

        vq_star = numerator / denominator if denominator != 0 else 0
        vq_star_dict[gd_pair] = vq_star

    df['vq*'] = df.apply(
        lambda row: vq_star_dict.get((row.get('disease', 'unknown'), str(row['id1'])), 0),
        axis=1
    )
    df['vq_star_mean'] = df['vq*']

    # Filter by paper count
    if min_papers > 1:
        df = df.groupby(['disease', 'id1']).filter(lambda x: len(x) >= min_papers)

    return df


def load_crispr_data(file_path: str) -> pd.DataFrame:
    """
    Load CRISPR/directed regulatory network data.

    Args:
        file_path: Path to the TSV file containing network data

    Returns:
        DataFrame with source_gene, target_gene, and score columns
    """
    if not os.path.exists(file_path):
        return None

    print(f"  Loading Directed Network: {file_path}")
    try:
        df = pd.read_csv(file_path, sep='\t')
        df['source_gene'] = df['source_gene'].astype(str).str.replace(r'\.0$', '', regex=True)
        df['target_gene'] = df['target_gene'].astype(str).str.replace(r'\.0$', '', regex=True)
        return df
    except Exception as e:
        print(f"  Error loading CRISPR data: {e}")
        return None
