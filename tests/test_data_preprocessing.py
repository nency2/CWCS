"""
Tests for src.utils.data_preprocessing module.
"""

import pytest
import pandas as pd

from src.utils.data_preprocessing import group_evidence_data


class TestGroupEvidenceData:
    """Tests for the group_evidence_data function."""

    def test_group_evidence_data_basic(self, sample_evidence_df):
        """
        Basic grouping: two (id1, id2) pairs should produce two result dicts,
        each with a combined evidence_context string.
        """
        result = group_evidence_data(sample_evidence_df, label_col="label")

        assert isinstance(result, list)
        assert len(result) == 2  # BRCA1/Asthma, TP53/COPD

        # Build a lookup by (id1, id2) for easier assertions.
        lookup = {(d["id1"], d["id2"]): d for d in result}

        # --- BRCA1 / Asthma pair ---
        brca_entry = lookup[("BRCA1", "Asthma")]
        assert "BRCA1 causes asthma." in brca_entry["evidence_context"]
        assert "BRCA1 linked to asthma symptoms." in brca_entry["evidence_context"]
        # Both rows have label=1, so original_label should be 1.
        assert brca_entry["original_label"] == 1

        # --- TP53 / COPD pair ---
        tp53_entry = lookup[("TP53", "COPD")]
        assert "TP53 drives COPD." in tp53_entry["evidence_context"]
        assert "TP53 associated with COPD risk." in tp53_entry["evidence_context"]
        assert "TP53 regulates COPD pathway." in tp53_entry["evidence_context"]
        # Evidence context should include prediction info.
        assert "Prior Model Prediction:" in tp53_entry["evidence_context"]
        assert "Confidence:" in tp53_entry["evidence_context"]

    def test_group_evidence_data_auto_label_detection(self):
        """
        When label_col is not provided, the function should auto-detect
        common label column names (e.g. 'label', 'Label', 'ground_truth').
        """
        df = pd.DataFrame({
            "id1": ["GENEA", "GENEA"],
            "id2": ["DiseaseX", "DiseaseX"],
            "sentence": ["sent1", "sent2"],
            "pred_label": [1, 1],
            "pred_proba": [0.8, 0.7],
            "label": [1, 1],  # should be auto-detected
        })

        result = group_evidence_data(df)  # label_col=None

        assert len(result) == 1
        assert result[0]["original_label"] == 1

        # Also test with a differently named column: 'ground_truth'.
        df2 = df.rename(columns={"label": "ground_truth"})
        result2 = group_evidence_data(df2)
        assert result2[0]["original_label"] == 1

    def test_group_evidence_data_no_predictions(self):
        """
        When pred_label and pred_proba columns are absent, the function
        should fill them with 'N/A' and still produce valid output.
        """
        df = pd.DataFrame({
            "id1": ["GENEB", "GENEB"],
            "id2": ["DiseaseY", "DiseaseY"],
            "sentence": ["Evidence sentence one.", "Evidence sentence two."],
        })

        result = group_evidence_data(df)

        assert len(result) == 1
        entry = result[0]
        assert entry["id1"] == "GENEB"
        assert entry["id2"] == "DiseaseY"
        # With no label column detected, original_label should be None.
        assert entry["original_label"] is None
        # Evidence context should still be built with N/A markers.
        assert "N/A" in entry["evidence_context"]
        assert "Evidence sentence one." in entry["evidence_context"]
        assert "Evidence sentence two." in entry["evidence_context"]
