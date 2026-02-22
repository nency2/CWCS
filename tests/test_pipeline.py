"""
Tests for src.pipeline module.
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock, call

import pandas as pd
import networkx as nx


class TestRunPipelineIntegration(unittest.TestCase):
    """Integration test for run_pipeline orchestration."""

    @patch("src.pipeline.calculate_fusion")
    @patch("src.pipeline.calculate_pagerank")
    @patch("src.pipeline.build_graph")
    @patch("src.pipeline.fetch_omnipath_network")
    @patch("src.pipeline.load_crispr_data")
    @patch("src.pipeline.load_unified_data")
    @patch("src.pipeline.os.path.exists")
    @patch("src.pipeline.os.makedirs")
    @patch("src.pipeline.pd.read_csv")
    def test_run_pipeline_integration(
        self,
        mock_read_csv,
        mock_makedirs,
        mock_exists,
        mock_load_unified,
        mock_load_crispr,
        mock_fetch_omnipath,
        mock_build_graph,
        mock_pagerank,
        mock_fusion,
    ):
        """Mock the data loading, build_graph, pagerank, fusion. Verify pipeline orchestrates calls correctly."""
        from src.pipeline import run_pipeline

        # Setup: load_unified_data returns a DataFrame with expected columns
        mock_df = pd.DataFrame({
            "id1": ["BRCA1", "TP53"],
            "id2": ["100", "200"],
            "disease": ["Breast Cancer", "Breast Cancer"],
            "Symbol": ["BRCA1", "TP53"],
        })
        mock_load_unified.return_value = mock_df

        # Setup: crispr file exists
        mock_exists_map = {
            "crispr_data.tsv": True,       # crispr_file exists
        }

        def exists_side_effect(path):
            basename = os.path.basename(path)
            # The crispr_file path
            if path == "crispr_data.tsv":
                return True
            # The output directory
            if path == "/tmp/test_output":
                return False
            # Split files (train, val, test)
            if basename in ("train_aggregated.tsv", "val_aggregated.tsv", "test_aggregated.tsv"):
                return False
            # vq_scores
            if path == "vq_scores.tsv":
                return False
            return False

        mock_exists.side_effect = exists_side_effect

        # Setup: load_crispr_data returns a DataFrame
        mock_crispr_df = pd.DataFrame({
            "source_gene": ["BRCA1"],
            "target_gene": ["TP53"],
            "weight": [0.5],
        })
        mock_load_crispr.return_value = mock_crispr_df

        # Setup: build_graph returns a graph, target_id, and seed_genes
        mock_graph = nx.DiGraph()
        mock_graph.add_node("BRCA1")
        mock_graph.add_node("Breast Cancer")
        mock_build_graph.return_value = (mock_graph, "Breast Cancer", ["BRCA1", "TP53"])

        # Setup: calculate_pagerank returns scores
        mock_pagerank.return_value = {"BRCA1": 0.8, "TP53": 0.6}

        # Run the pipeline
        run_pipeline(
            input_file="test_input.tsv",
            output_dir="/tmp/test_output",
            vq_scores=None,
            crispr_file="crispr_data.tsv",
            min_papers=1,
            export_neo4j=False,
        )

        # Verify orchestration: load_unified_data was called with the input file
        mock_load_unified.assert_called_once_with("test_input.tsv", min_papers=1)

        # Verify crispr data was loaded (file existed)
        mock_load_crispr.assert_called_once_with("crispr_data.tsv")

        # Verify build_graph was called
        mock_build_graph.assert_called_once()

        # Verify pagerank was calculated
        mock_pagerank.assert_called_once_with(mock_graph, "Breast Cancer", ["BRCA1", "TP53"])

        # Verify output dir creation was attempted
        mock_makedirs.assert_called_once_with("/tmp/test_output")

    @patch("src.pipeline.calculate_fusion")
    @patch("src.pipeline.calculate_pagerank")
    @patch("src.pipeline.build_graph")
    @patch("src.pipeline.fetch_omnipath_network")
    @patch("src.pipeline.load_crispr_data")
    @patch("src.pipeline.load_unified_data")
    @patch("src.pipeline.export_to_neo4j")
    @patch("src.pipeline.os.path.exists")
    @patch("src.pipeline.os.makedirs")
    def test_run_pipeline_with_neo4j_export(
        self,
        mock_makedirs,
        mock_exists,
        mock_export_neo4j,
        mock_load_unified,
        mock_load_crispr,
        mock_fetch_omnipath,
        mock_build_graph,
        mock_pagerank,
        mock_fusion,
    ):
        """Verify export_to_neo4j is called when export_neo4j=True."""
        from src.pipeline import run_pipeline

        mock_df = pd.DataFrame({
            "id1": ["BRCA1"],
            "id2": ["100"],
            "disease": ["Breast Cancer"],
            "Symbol": ["BRCA1"],
        })
        mock_load_unified.return_value = mock_df

        def exists_side_effect(path):
            if path == "crispr_data.tsv":
                return True
            return False

        mock_exists.side_effect = exists_side_effect

        mock_load_crispr.return_value = pd.DataFrame({
            "source_gene": ["BRCA1"],
            "target_gene": ["TP53"],
            "weight": [0.5],
        })

        mock_graph = nx.DiGraph()
        mock_build_graph.return_value = (mock_graph, "Breast Cancer", ["BRCA1"])
        mock_pagerank.return_value = {"BRCA1": 0.8}

        run_pipeline(
            input_file="test_input.tsv",
            output_dir="/tmp/test_output",
            crispr_file="crispr_data.tsv",
            export_neo4j=True,
        )

        # Verify export_to_neo4j was called
        mock_export_neo4j.assert_called_once_with(mock_graph, "Breast Cancer")


class TestMainCliArgs(unittest.TestCase):
    """Test that main() parses command-line arguments correctly."""

    @patch("src.pipeline.run_pipeline")
    def test_main_cli_args(self, mock_run_pipeline):
        """Test main() parses command-line arguments and passes them to run_pipeline."""
        from src.pipeline import main

        test_args = [
            "prog",
            "--input_file", "my_input.tsv",
            "--output_dir", "/tmp/my_output",
            "--vq_scores", "my_vq.tsv",
            "--crispr_file", "my_crispr.tsv",
            "--min_papers", "3",
            "--export_neo4j",
        ]

        with patch.object(sys, "argv", test_args):
            main()

        mock_run_pipeline.assert_called_once_with(
            input_file="my_input.tsv",
            output_dir="/tmp/my_output",
            vq_scores="my_vq.tsv",
            crispr_file="my_crispr.tsv",
            min_papers=3,
            export_neo4j=True,
        )

    @patch("src.pipeline.run_pipeline")
    def test_main_cli_defaults(self, mock_run_pipeline):
        """Test main() uses default arguments when none are provided."""
        from src.pipeline import main
        from src.config import OUTPUT_DIR

        test_args = ["prog"]

        with patch.object(sys, "argv", test_args):
            main()

        mock_run_pipeline.assert_called_once()
        call_kwargs = mock_run_pipeline.call_args[1]
        self.assertEqual(
            call_kwargs["input_file"],
            "complete_data_bibliometrics_with_all_diseases_biobert_svm_prediction.tsv",
        )
        self.assertEqual(call_kwargs["output_dir"], OUTPUT_DIR)
        self.assertEqual(call_kwargs["min_papers"], 1)
        self.assertFalse(call_kwargs["export_neo4j"])


if __name__ == "__main__":
    unittest.main()
