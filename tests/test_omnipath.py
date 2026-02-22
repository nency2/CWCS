"""
Tests for src.data.omnipath module.
"""

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock

from src.data.omnipath import fetch_omnipath_network


# A minimal TSV response that the OmniPath API would return.
MOCK_OMNIPATH_TSV = (
    "source_genesymbol\ttarget_genesymbol\tsources\n"
    "BRCA1\tTP53\tSignor;KEGG;Reactome\n"
    "TP53\tEGFR\tSignor;KEGG\n"
    "EGFR\tMYC\tSignor\n"
    "NRAS\tKRAS\tKEGG\n"  # genes NOT in our set
)


class TestFetchOmnipathNetwork:
    """Tests for fetch_omnipath_network."""

    @patch("src.data.omnipath.requests.get")
    def test_fetch_omnipath_network_success(self, mock_get, tmp_path):
        """
        Mock a successful API response and verify:
        - Filtering keeps only interactions where BOTH genes are in our set.
        - The 'score' column is computed from n_sources / max_sources.
        - Output file is written.
        """
        mock_response = MagicMock()
        mock_response.text = MOCK_OMNIPATH_TSV
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        unique_genes = ["BRCA1", "TP53", "EGFR"]
        output_file = str(tmp_path / "omnipath_network.tsv")

        result = fetch_omnipath_network(unique_genes, output_file)

        # Should have called requests.get exactly once.
        mock_get.assert_called_once()

        # The NRAS->KRAS edge should be filtered out (not in unique_genes).
        # EGFR->MYC should also be filtered out (MYC not in our set).
        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2  # BRCA1->TP53, TP53->EGFR

        # Verify column names.
        assert set(result.columns) == {"source_gene", "target_gene", "score"}

        # Verify score calculation: max n_sources = 3 (BRCA1->TP53).
        brca1_row = result[result["source_gene"] == "BRCA1"].iloc[0]
        assert brca1_row["score"] == pytest.approx(1.0)  # 3/3

        tp53_row = result[result["source_gene"] == "TP53"].iloc[0]
        assert tp53_row["score"] == pytest.approx(2 / 3)  # 2/3

        # Verify the output file was written.
        saved = pd.read_csv(output_file, sep="\t")
        assert len(saved) == 2

    @patch("src.data.omnipath.requests.get")
    def test_fetch_omnipath_network_api_failure(self, mock_get, tmp_path):
        """
        When requests.get raises an exception the function must return None
        and not propagate the error.
        """
        mock_get.side_effect = Exception("Connection timed out")

        unique_genes = ["BRCA1", "TP53"]
        output_file = str(tmp_path / "omnipath_fail.tsv")

        result = fetch_omnipath_network(unique_genes, output_file)
        assert result is None

    @patch("src.data.omnipath.requests.get")
    def test_fetch_omnipath_network_no_overlaps(self, mock_get, tmp_path):
        """
        When the API returns data but none of the interactions involve our
        genes, the function must return None.
        """
        # The TSV contains only NRAS->KRAS which won't match our gene set.
        mock_response = MagicMock()
        mock_response.text = (
            "source_genesymbol\ttarget_genesymbol\tsources\n"
            "NRAS\tKRAS\tKEGG\n"
        )
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        unique_genes = ["BRCA1", "TP53", "EGFR"]
        output_file = str(tmp_path / "omnipath_empty.tsv")

        result = fetch_omnipath_network(unique_genes, output_file)
        assert result is None
