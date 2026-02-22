"""Tests for src.algorithms.fusion.calculate_fusion."""

import math

import pytest
import numpy as np
import pandas as pd

from src.algorithms.fusion import calculate_fusion
from src.config import K_RRF


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _make_df(pr_scores, vq_scores):
    """Build a minimal DataFrame with the two required score columns."""
    return pd.DataFrame({
        "matrix_pagerank_score": pr_scores,
        "vq_star_mean": vq_scores,
    })


# ------------------------------------------------------------------ #
# Tests
# ------------------------------------------------------------------ #

class TestFusionBasic:
    """test_fusion_basic: verify rrf_score and geo_mean_score columns are added."""

    def test_rrf_column_added(self):
        df = _make_df([0.9, 0.5, 0.1], [0.3, 0.7, 0.6])
        result = calculate_fusion(df)
        assert "rrf_score" in result.columns

    def test_geo_mean_column_added(self):
        df = _make_df([0.9, 0.5, 0.1], [0.3, 0.7, 0.6])
        result = calculate_fusion(df)
        assert "geo_mean_score" in result.columns

    def test_row_count_preserved(self):
        df = _make_df([0.9, 0.5, 0.1], [0.3, 0.7, 0.6])
        result = calculate_fusion(df)
        assert len(result) == 3

    def test_no_nans_in_output(self):
        df = _make_df([0.9, 0.5, 0.1], [0.3, 0.7, 0.6])
        result = calculate_fusion(df)
        assert result["rrf_score"].isna().sum() == 0
        assert result["geo_mean_score"].isna().sum() == 0


class TestFusionWithNans:
    """test_fusion_with_nans: NaN values are filled with 0 before computation."""

    def test_nan_pagerank_filled(self):
        df = _make_df([float("nan"), 0.5], [0.3, 0.7])
        result = calculate_fusion(df)
        assert not result["rrf_score"].isna().any()
        assert not result["geo_mean_score"].isna().any()

    def test_nan_vq_filled(self):
        df = _make_df([0.9, 0.5], [float("nan"), 0.7])
        result = calculate_fusion(df)
        assert not result["rrf_score"].isna().any()
        assert not result["geo_mean_score"].isna().any()

    def test_all_nan_handled(self):
        df = _make_df([float("nan"), float("nan")], [float("nan"), float("nan")])
        result = calculate_fusion(df)
        # All zeros -> geo_mean should be 0, rrf should still be computable
        assert (result["geo_mean_score"] == 0.0).all()
        assert not result["rrf_score"].isna().any()


class TestFusionGeometricMean:
    """test_fusion_geometric_mean: geo_mean = sqrt(pr * vq) for known values."""

    def test_known_values(self):
        df = _make_df([0.64, 0.25], [0.36, 0.16])
        result = calculate_fusion(df)
        expected_0 = math.sqrt(0.64 * 0.36)  # 0.48
        expected_1 = math.sqrt(0.25 * 0.16)  # 0.2
        assert result["geo_mean_score"].iloc[0] == pytest.approx(expected_0)
        assert result["geo_mean_score"].iloc[1] == pytest.approx(expected_1)

    def test_zero_input_gives_zero(self):
        df = _make_df([0.0], [0.5])
        result = calculate_fusion(df)
        assert result["geo_mean_score"].iloc[0] == pytest.approx(0.0)

    def test_identical_scores(self):
        """sqrt(x * x) == x for non-negative x."""
        df = _make_df([0.49], [0.49])
        result = calculate_fusion(df)
        assert result["geo_mean_score"].iloc[0] == pytest.approx(0.49)


class TestFusionRRFFormula:
    """test_fusion_rrf_formula: RRF = 1/(K+rank_pr) + 1/(K+rank_vq)."""

    def test_known_values(self):
        # Three rows; ranks are assigned descending (highest value = rank 1).
        df = _make_df([0.9, 0.5, 0.1], [0.1, 0.5, 0.9])
        result = calculate_fusion(df)

        # Row 0: pr rank=1, vq rank=3
        expected_0 = 1 / (K_RRF + 1) + 1 / (K_RRF + 3)
        assert result["rrf_score"].iloc[0] == pytest.approx(expected_0)

        # Row 1: pr rank=2, vq rank=2
        expected_1 = 1 / (K_RRF + 2) + 1 / (K_RRF + 2)
        assert result["rrf_score"].iloc[1] == pytest.approx(expected_1)

        # Row 2: pr rank=3, vq rank=1
        expected_2 = 1 / (K_RRF + 3) + 1 / (K_RRF + 1)
        assert result["rrf_score"].iloc[2] == pytest.approx(expected_2)

    def test_rrf_symmetry(self):
        """Swapping pr and vq values should produce the same RRF score."""
        df1 = _make_df([0.9, 0.1], [0.1, 0.9])
        df2 = _make_df([0.1, 0.9], [0.9, 0.1])
        r1 = calculate_fusion(df1)
        r2 = calculate_fusion(df2)
        # Row 0 of df1 should match Row 1 of df2 (both have rank 1 in one, rank 2 in other)
        assert r1["rrf_score"].iloc[0] == pytest.approx(r2["rrf_score"].iloc[1])

    def test_k_rrf_is_60(self):
        """Sanity check that the constant used is 60."""
        assert K_RRF == 60


class TestFusionDoesNotModifyInput:
    """test_fusion_does_not_modify_input: input DataFrame is untouched."""

    def test_original_df_unchanged(self):
        df = _make_df([0.9, 0.5, 0.1], [0.3, 0.7, 0.6])
        original_columns = list(df.columns)
        original_values = df.copy()

        _ = calculate_fusion(df)

        # Columns should not have changed
        assert list(df.columns) == original_columns
        # Values should not have changed
        pd.testing.assert_frame_equal(df, original_values)

    def test_nan_not_filled_in_original(self):
        df = _make_df([float("nan"), 0.5], [0.3, float("nan")])
        _ = calculate_fusion(df)
        # Original NaNs should still be NaN
        assert pd.isna(df["matrix_pagerank_score"].iloc[0])
        assert pd.isna(df["vq_star_mean"].iloc[1])
