"""
Shared pytest fixtures for the CWCS test suite.
"""

import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def sample_bibliometric_df():
    """
    A pandas DataFrame simulating bibliometric data with columns matching
    the unified TSV format consumed by load_unified_data.

    Contains 6 rows, 2 diseases (Asthma, COPD), and 3 genes (BRCA1, TP53, EGFR).
    """
    return pd.DataFrame({
        "id1": ["BRCA1", "BRCA1", "TP53", "TP53", "EGFR", "EGFR"],
        "id2": ["pmid_1", "pmid_2", "pmid_3", "pmid_4", "pmid_5", "pmid_6"],
        "disease": ["Asthma", "Asthma", "Asthma", "COPD", "COPD", "COPD"],
        "sentence": [
            "BRCA1 causes asthma symptoms.",
            "BRCA1 is linked to asthma.",
            "TP53 regulates asthma pathway.",
            "TP53 is associated with COPD.",
            "EGFR drives COPD progression.",
            "EGFR mutation leads to COPD.",
        ],
        "hindex": [45, 30, 60, 25, 50, 35],
        "citations": [1200, 800, 2500, 600, 1800, 900],
        "year": [2020, 2018, 2022, 2019, 2021, 2017],
        "pred_label": [1, 0, 1, 1, 0, 1],
        "pred_proba": [0.92, 0.35, 0.88, 0.78, 0.42, 0.91],
        "Symbol": ["BRCA1", "BRCA1", "TP53", "TP53", "EGFR", "EGFR"],
    })


@pytest.fixture
def sample_crispr_df():
    """
    A DataFrame with source_gene, target_gene, score columns
    simulating a CRISPR / directed regulatory network.
    """
    return pd.DataFrame({
        "source_gene": ["BRCA1", "TP53", "EGFR", "MYC"],
        "target_gene": ["TP53", "EGFR", "MYC", "BRCA1"],
        "score": [0.95, 0.80, 0.65, 0.70],
    })


@pytest.fixture
def sample_processed_df():
    """
    A DataFrame with matrix_pagerank_score and vq_star_mean columns,
    simulating the output of the scoring pipeline.
    """
    return pd.DataFrame({
        "matrix_pagerank_score": [0.12, 0.45, 0.78, 0.33, 0.91],
        "vq_star_mean": [0.55, 0.62, 0.88, 0.41, 0.73],
    })


@pytest.fixture
def sample_evidence_df():
    """
    A DataFrame suitable for testing group_evidence_data, containing
    sentence-level rows with prediction columns and a ground-truth label.
    """
    return pd.DataFrame({
        "id1": ["BRCA1", "BRCA1", "TP53", "TP53", "TP53"],
        "id2": ["Asthma", "Asthma", "COPD", "COPD", "COPD"],
        "sentence": [
            "BRCA1 causes asthma.",
            "BRCA1 linked to asthma symptoms.",
            "TP53 drives COPD.",
            "TP53 associated with COPD risk.",
            "TP53 regulates COPD pathway.",
        ],
        "pred_label": [1, 0, 1, 1, 0],
        "pred_proba": [0.91, 0.30, 0.85, 0.78, 0.40],
        "label": [1, 1, 1, 1, 0],
    })


@pytest.fixture
def tmp_output_dir(tmp_path):
    """
    A temporary directory (provided by pytest's tmp_path) for writing
    test output files that are automatically cleaned up.
    """
    output_dir = tmp_path / "test_output"
    output_dir.mkdir()
    return output_dir
