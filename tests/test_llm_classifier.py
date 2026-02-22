"""
Tests for src.inference.llm_classifier module.
"""

import json
import sys
import unittest
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd

# Import the module under test
from src.inference.llm_classifier import (
    build_user_prompt,
    NpEncoder,
    CausalAnalysis,
    LLM_AVAILABLE,
)


class TestBuildUserPrompt(unittest.TestCase):
    """Tests for build_user_prompt function."""

    def test_build_user_prompt(self):
        """Test build_user_prompt returns a string containing gene, disease, and evidence context."""
        gene = "BRCA1"
        disease = "Breast Cancer"
        evidence = "Paper 1: BRCA1 mutations linked to breast cancer."

        result = build_user_prompt(evidence, gene, disease)

        self.assertIsInstance(result, str)
        self.assertIn("Gene: BRCA1", result)
        self.assertIn("Disease: Breast Cancer", result)
        self.assertIn("EVIDENCE & PRIOR PREDICTIONS:", result)
        self.assertIn(evidence, result)
        self.assertIn(f"Does {gene} cause {disease}?", result)


class TestNpEncoder(unittest.TestCase):
    """Tests for NpEncoder JSON encoder."""

    def test_np_encoder_int64(self):
        """Test NpEncoder handles np.int64."""
        data = {"value": np.int64(42)}
        result = json.dumps(data, cls=NpEncoder)
        decoded = json.loads(result)
        self.assertEqual(decoded["value"], 42)
        self.assertIsInstance(decoded["value"], int)

    def test_np_encoder_float64(self):
        """Test NpEncoder handles np.float64."""
        data = {"value": np.float64(3.14)}
        result = json.dumps(data, cls=NpEncoder)
        decoded = json.loads(result)
        self.assertAlmostEqual(decoded["value"], 3.14, places=5)
        self.assertIsInstance(decoded["value"], float)

    def test_np_encoder_ndarray(self):
        """Test NpEncoder handles np.ndarray -> list."""
        data = {"value": np.array([1, 2, 3])}
        result = json.dumps(data, cls=NpEncoder)
        decoded = json.loads(result)
        self.assertEqual(decoded["value"], [1, 2, 3])
        self.assertIsInstance(decoded["value"], list)

    def test_np_encoder_nan_to_none(self):
        """Test NpEncoder handles np.nan -> None.

        Note: np.nan is a Python float, so json.dumps serializes it as NaN
        without calling NpEncoder.default. The pd.isna check in default()
        is reached by non-float NA types (e.g., pd.NaT). For np.nan,
        we verify the encoder does not raise and produces valid output.
        We use a numpy float wrapper to force the default method path.
        """
        # np.float64(np.nan) goes through the NpEncoder.default path
        # because it's np.floating, and float(np.nan) => NaN in JSON.
        data = {"value": np.float64(np.nan)}
        result = json.dumps(data, cls=NpEncoder)
        # np.float64(nan) -> float(nan) -> JSON NaN
        # json.loads converts NaN to float('nan')
        decoded = json.loads(result)
        self.assertTrue(pd.isna(decoded["value"]))

    def test_np_encoder_pd_nat(self):
        """Test NpEncoder handles pd.NaT -> None (also pd.isna)."""
        data = {"value": pd.NaT}
        result = json.dumps(data, cls=NpEncoder)
        decoded = json.loads(result)
        self.assertIsNone(decoded["value"])


@unittest.skipIf(CausalAnalysis is None, "LLM dependencies (pydantic, instructor, etc.) not installed")
class TestCausalAnalysis(unittest.TestCase):
    """Tests for CausalAnalysis Pydantic model."""

    def test_causal_analysis_valid(self):
        """Test CausalAnalysis validates correctly for a valid input."""
        obj = CausalAnalysis(relationship="Causal", score=0.8)
        self.assertEqual(obj.relationship, "Causal")
        self.assertAlmostEqual(obj.score, 0.8)

    def test_causal_analysis_valid_not_causal(self):
        """Test CausalAnalysis validates 'Not causal' relationship."""
        obj = CausalAnalysis(relationship="Not causal", score=0.3)
        self.assertEqual(obj.relationship, "Not causal")
        self.assertAlmostEqual(obj.score, 0.3)

    def test_causal_analysis_invalid_score(self):
        """Test score > 1.0 raises ValidationError."""
        from pydantic import ValidationError
        with self.assertRaises(ValidationError):
            CausalAnalysis(relationship="Causal", score=1.5)

    def test_causal_analysis_invalid_score_negative(self):
        """Test score < 0.0 raises ValidationError."""
        from pydantic import ValidationError
        with self.assertRaises(ValidationError):
            CausalAnalysis(relationship="Causal", score=-0.1)

    def test_causal_analysis_robust_validator_list(self):
        """Test that list input returns default 'Not causal' via robust_validator."""
        obj = CausalAnalysis.model_validate([{"something": "irrelevant"}])
        self.assertEqual(obj.relationship, "Not causal")
        self.assertAlmostEqual(obj.score, 0.0)

    def test_causal_analysis_robust_validator_nested_data(self):
        """Test robust_validator extracts from nested 'data' key."""
        obj = CausalAnalysis.model_validate({
            "data": {"relationship": "Causal", "score": 0.9}
        })
        self.assertEqual(obj.relationship, "Causal")
        self.assertAlmostEqual(obj.score, 0.9)

    def test_causal_analysis_robust_validator_nested_result(self):
        """Test robust_validator extracts from nested 'result' key."""
        obj = CausalAnalysis.model_validate({
            "result": {"relationship": "Not causal", "score": 0.1}
        })
        self.assertEqual(obj.relationship, "Not causal")
        self.assertAlmostEqual(obj.score, 0.1)


class TestCallLlmMocked(unittest.TestCase):
    """Tests for call_llm with mocked dependencies."""

    @patch("src.inference.llm_classifier.build_user_prompt", return_value="mocked prompt")
    def test_call_llm_mocked(self, mock_build_prompt):
        """Mock the client and tokenizer, verify call_llm returns expected dict structure."""
        from src.inference.llm_classifier import call_llm

        # Create a mock response object
        mock_resp = MagicMock()
        mock_resp.relationship = "Causal"
        mock_resp.score = 0.85
        mock_resp.reasoning = "Strong evidence"
        mock_resp.model_dump.return_value = {
            "relationship": "Causal",
            "score": 0.85,
        }

        # Mock client
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_resp

        # Mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = list(range(100))  # 100 tokens
        mock_tokenizer.decode.return_value = "decoded text"

        item = {
            "id1": "BRCA1",
            "id2": "Breast Cancer",
            "evidence_context": "Some evidence text",
            "original_label": 1,
        }

        result = call_llm(item, mock_client, mock_tokenizer, log_path=None)

        # Verify the result dict has the expected keys
        self.assertIn("id1", result)
        self.assertIn("id2", result)
        self.assertIn("llm_relationship", result)
        self.assertIn("llm_score", result)

        self.assertEqual(result["id1"], "BRCA1")
        self.assertEqual(result["id2"], "Breast Cancer")
        self.assertEqual(result["llm_relationship"], "Causal")
        self.assertAlmostEqual(result["llm_score"], 0.85)
        self.assertEqual(result["original_label"], 1)

        # Verify client was called
        mock_client.chat.completions.create.assert_called_once()


class TestProcessAllPairsMocked(unittest.TestCase):
    """Tests for process_all_pairs with mocked dependencies."""

    @patch("src.inference.llm_classifier.create_client")
    @patch("src.inference.llm_classifier.tiktoken")
    @patch("src.inference.llm_classifier.call_llm")
    def test_process_all_pairs_mocked(self, mock_call_llm, mock_tiktoken, mock_create_client):
        """Mock create_client and tiktoken, verify process_all_pairs returns a list of results."""
        from src.inference.llm_classifier import process_all_pairs

        # Skip if LLM deps not available (process_all_pairs raises ImportError)
        if not LLM_AVAILABLE:
            self.skipTest("LLM dependencies not installed")

        # Setup mocks
        mock_create_client.return_value = MagicMock()
        mock_tokenizer_instance = MagicMock()
        mock_tiktoken.encoding_for_model.return_value = mock_tokenizer_instance

        # Configure call_llm to return a result dict
        mock_call_llm.side_effect = lambda item, client, tokenizer, log_path=None: {
            "id1": item["id1"],
            "id2": item["id2"],
            "original_label": item["original_label"],
            "llm_relationship": "Causal",
            "llm_score": 0.9,
            "llm_reasoning": "mock reasoning",
        }

        grouped_data = [
            {
                "id1": "TP53",
                "id2": "Lung Cancer",
                "evidence_context": "evidence 1",
                "original_label": 1,
            },
            {
                "id1": "EGFR",
                "id2": "Glioblastoma",
                "evidence_context": "evidence 2",
                "original_label": 0,
            },
        ]

        results = process_all_pairs(grouped_data, log_path=None)

        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 2)

        self.assertEqual(results[0]["id1"], "TP53")
        self.assertEqual(results[0]["llm_relationship"], "Causal")

        self.assertEqual(results[1]["id1"], "EGFR")
        self.assertEqual(results[1]["llm_relationship"], "Causal")


if __name__ == "__main__":
    unittest.main()
