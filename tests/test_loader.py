"""
Tests for src.data.loader module.
"""

import os
import pytest
import pandas as pd
import numpy as np

from src.data.loader import load_unified_data, load_crispr_data
from src.config import BETA


# ------------------------------------------------------------------ #
# load_unified_data
# ------------------------------------------------------------------ #

class TestLoadUnifiedData:
    """Tests for the load_unified_data function."""

    def _write_tsv(self, path, df):
        """Helper to write a DataFrame as a TSV file."""
        df.to_csv(path, sep="\t", index=False)

    def test_load_unified_data_valid(self, tmp_path):
        """Load a well-formed TSV and verify RS and VQ* scores are computed."""
        tsv_path = str(tmp_path / "valid_data.tsv")

        df_input = pd.DataFrame({
            "id1": ["BRCA1", "BRCA1", "TP53"],
            "id2": ["pmid_1", "pmid_2", "pmid_3"],
            "disease": ["Asthma", "Asthma", "Asthma"],
            "sentence": [
                "BRCA1 causes asthma.",
                "BRCA1 linked to asthma.",
                "TP53 regulates asthma.",
            ],
            "hindex": [50, 30, 60],
            "citations": [2000, 800, 3000],
            "year": [2022, 2018, 2023],
            "pred_label": [1, 0, 1],
            "pred_proba": [0.9, 0.3, 0.85],
            "Symbol": ["BRCA1", "BRCA1", "TP53"],
        })
        self._write_tsv(tsv_path, df_input)

        result = load_unified_data(tsv_path)

        # The returned DataFrame must contain the computed columns.
        assert "rs" in result.columns, "RS score column missing"
        assert "vq*" in result.columns, "VQ* column missing"
        assert "vq_star_mean" in result.columns, "vq_star_mean column missing"

        # All rows should survive (no NaN in required cols).
        assert len(result) == 3

        # RS must be a finite number for every row.
        assert result["rs"].notna().all()
        assert np.isfinite(result["rs"].values).all()

        # VQ* must lie in [0, 1].
        assert (result["vq*"] >= 0).all()
        assert (result["vq*"] <= 1).all()

        # Verify RS calculation uses the correct BETA vector.
        # Pick first row and recompute manually.
        row = result.iloc[0]
        h_scaled = row["hindex_scaled"]
        c_scaled = row["citations_scaled"]
        y_norm = row["year_diff_norm"]
        expected_rs = BETA[0] * h_scaled + BETA[1] * c_scaled + BETA[2] * y_norm
        assert np.isclose(row["rs"], expected_rs, atol=1e-6), (
            f"RS mismatch: got {row['rs']}, expected {expected_rs}"
        )

    def test_load_unified_data_missing_column(self, tmp_path):
        """A TSV without the 'id1' column must raise ValueError."""
        tsv_path = str(tmp_path / "bad_data.tsv")

        df_input = pd.DataFrame({
            "gene": ["BRCA1"],  # 'id1' is missing
            "id2": ["pmid_1"],
            "disease": ["Asthma"],
            "sentence": ["text"],
            "hindex": [50],
            "citations": [1000],
            "year": [2020],
        })
        self._write_tsv(tsv_path, df_input)

        with pytest.raises(ValueError, match="Missing id1 column"):
            load_unified_data(tsv_path)

    def test_load_unified_data_min_papers_filter(self, tmp_path):
        """Pairs with fewer rows than min_papers must be filtered out."""
        tsv_path = str(tmp_path / "filter_data.tsv")

        # BRCA1/Asthma has 3 rows, TP53/Asthma has 1 row.
        df_input = pd.DataFrame({
            "id1": ["BRCA1", "BRCA1", "BRCA1", "TP53"],
            "id2": ["pmid_1", "pmid_2", "pmid_3", "pmid_4"],
            "disease": ["Asthma", "Asthma", "Asthma", "Asthma"],
            "sentence": ["s1", "s2", "s3", "s4"],
            "hindex": [40, 30, 50, 20],
            "citations": [1000, 800, 1500, 500],
            "year": [2020, 2019, 2021, 2018],
            "pred_label": [1, 0, 1, 1],
            "pred_proba": [0.9, 0.3, 0.8, 0.7],
            "Symbol": ["BRCA1", "BRCA1", "BRCA1", "TP53"],
        })
        self._write_tsv(tsv_path, df_input)

        # With min_papers=2 the single-row TP53/Asthma pair should be removed.
        result = load_unified_data(tsv_path, min_papers=2)
        assert len(result) == 3
        assert set(result["id1"].unique()) == {"BRCA1"}

        # With min_papers=1 (default), all rows stay.
        result_all = load_unified_data(tsv_path, min_papers=1)
        assert len(result_all) == 4


# ------------------------------------------------------------------ #
# load_crispr_data
# ------------------------------------------------------------------ #

class TestLoadCrisprData:
    """Tests for the load_crispr_data function."""

    def test_load_crispr_data_valid(self, tmp_path):
        """Loading a valid CRISPR TSV returns a proper DataFrame."""
        tsv_path = str(tmp_path / "crispr.tsv")
        df_input = pd.DataFrame({
            "source_gene": ["BRCA1", "TP53"],
            "target_gene": ["TP53", "EGFR"],
            "score": [0.9, 0.7],
        })
        df_input.to_csv(tsv_path, sep="\t", index=False)

        result = load_crispr_data(tsv_path)

        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert list(result.columns) == ["source_gene", "target_gene", "score"]
        # Values should be strings after the .astype(str) processing.
        assert result["source_gene"].dtype == object
        assert result["target_gene"].dtype == object

    def test_load_crispr_data_missing_file(self, tmp_path):
        """A non-existent file must return None, not raise."""
        result = load_crispr_data(str(tmp_path / "does_not_exist.tsv"))
        assert result is None
