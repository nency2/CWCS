"""
Score fusion methods: RRF and Geometric Mean.
"""

import numpy as np
import pandas as pd
from ..config import K_RRF


def calculate_fusion(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate fusion scores using RRF and Geometric Mean.

    Args:
        df: DataFrame with 'matrix_pagerank_score' and 'vq_star_mean' columns

    Returns:
        DataFrame with added 'rrf_score' and 'geo_mean_score' columns
    """
    df = df.copy()

    df['matrix_pagerank_score'] = df['matrix_pagerank_score'].fillna(0)
    df['vq_star_mean'] = df['vq_star_mean'].fillna(0)

    # Reciprocal Rank Fusion (RRF)
    df['rank_pr'] = df['matrix_pagerank_score'].rank(ascending=False)
    df['rank_vq'] = df['vq_star_mean'].rank(ascending=False)
    df['rrf_score'] = (1 / (K_RRF + df['rank_pr'])) + (1 / (K_RRF + df['rank_vq']))

    # Geometric Mean (Primary Metric)
    df['geo_mean_score'] = np.sqrt(df['matrix_pagerank_score'] * df['vq_star_mean'])

    return df
