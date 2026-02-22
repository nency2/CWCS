"""Tests for src.graph.builder.build_graph."""

import pytest
import pandas as pd
import networkx as nx

from src.graph.builder import build_graph


def _make_df(rows):
    """Helper to build a DataFrame from a list of (id1, id2, disease, pred_proba, pred_label, Symbol) tuples."""
    return pd.DataFrame(rows, columns=["id1", "id2", "disease", "pred_proba", "pred_label", "Symbol"])


# ------------------------------------------------------------------ #
# Fixtures
# ------------------------------------------------------------------ #

@pytest.fixture
def simple_df():
    """Three genes associated with one disease; two causal, one non-causal."""
    return _make_df([
        ("GENE_A", "DISEASE_1", "Asthma", 0.9, 1, "BRCA1"),
        ("GENE_B", "DISEASE_1", "Asthma", 0.7, 1, "TP53"),
        ("GENE_C", "DISEASE_1", "Asthma", 0.3, 0, "EGFR"),
    ])


@pytest.fixture
def crispr_df():
    """Regulatory edges between genes that exist in simple_df."""
    return pd.DataFrame([
        {"source_gene": "GENE_A", "target_gene": "GENE_B", "score": 0.8},
        {"source_gene": "GENE_B", "target_gene": "GENE_C", "score": 0.6},
    ])


# ------------------------------------------------------------------ #
# Tests
# ------------------------------------------------------------------ #

class TestBuildGraphBasic:
    """test_build_graph_basic: verify basic graph construction."""

    def test_returns_tuple_of_three(self, simple_df):
        result = build_graph(simple_df)
        assert isinstance(result, tuple) and len(result) == 3

    def test_graph_is_digraph(self, simple_df):
        G, _, _ = build_graph(simple_df)
        assert isinstance(G, nx.DiGraph)

    def test_disease_node_exists(self, simple_df):
        G, target_id, _ = build_graph(simple_df)
        assert target_id in G.nodes
        assert G.nodes[target_id]["type"] == "DISEASE"

    def test_gene_nodes_exist(self, simple_df):
        G, _, _ = build_graph(simple_df)
        for gene_id in ("GENE_A", "GENE_B", "GENE_C"):
            assert gene_id in G.nodes
            assert G.nodes[gene_id]["type"] == "GENE"

    def test_edges_have_direct_association_type(self, simple_df):
        G, target_id, _ = build_graph(simple_df)
        for gene_id in ("GENE_A", "GENE_B", "GENE_C"):
            assert G.has_edge(target_id, gene_id)
            assert G[target_id][gene_id]["type"] == "DIRECT_ASSOCIATION"


class TestBuildGraphWithCrispr:
    """test_build_graph_with_crispr: verify CAUSAL_REGULATION edges."""

    def test_causal_edges_added(self, simple_df, crispr_df):
        G, _, _ = build_graph(simple_df, crispr_df=crispr_df)
        assert G.has_edge("GENE_A", "GENE_B")
        assert G.has_edge("GENE_B", "GENE_C")

    def test_causal_edge_type(self, simple_df, crispr_df):
        G, _, _ = build_graph(simple_df, crispr_df=crispr_df)
        assert G["GENE_A"]["GENE_B"]["type"] == "CAUSAL_REGULATION"
        assert G["GENE_B"]["GENE_C"]["type"] == "CAUSAL_REGULATION"

    def test_causal_edge_weight(self, simple_df, crispr_df):
        G, _, _ = build_graph(simple_df, crispr_df=crispr_df)
        assert G["GENE_A"]["GENE_B"]["weight"] == pytest.approx(0.8)
        assert G["GENE_B"]["GENE_C"]["weight"] == pytest.approx(0.6)

    def test_crispr_ignores_unknown_genes(self, simple_df):
        """Genes not in df should be skipped."""
        crispr = pd.DataFrame([
            {"source_gene": "GENE_A", "target_gene": "UNKNOWN_GENE", "score": 0.5},
        ])
        G, _, _ = build_graph(simple_df, crispr_df=crispr)
        assert "UNKNOWN_GENE" not in G.nodes


class TestBuildGraphEmpty:
    """test_build_graph_empty: empty DataFrame should return (None, None, [])."""

    def test_empty_dataframe(self):
        empty_df = pd.DataFrame(columns=["id1", "id2", "disease", "pred_proba", "pred_label", "Symbol"])
        G, target_id, seed_genes = build_graph(empty_df)
        assert G is None
        assert target_id is None
        assert seed_genes == []

    def test_none_input(self):
        G, target_id, seed_genes = build_graph(None)
        assert G is None
        assert target_id is None
        assert seed_genes == []


class TestBuildGraphSeedGenes:
    """test_build_graph_seed_genes: genes with pred_label==1 appear in seed_genes."""

    def test_causal_genes_in_seeds(self, simple_df):
        _, _, seed_genes = build_graph(simple_df)
        assert "GENE_A" in seed_genes
        assert "GENE_B" in seed_genes

    def test_noncausal_genes_not_in_seeds(self, simple_df):
        _, _, seed_genes = build_graph(simple_df)
        assert "GENE_C" not in seed_genes

    def test_seed_genes_count(self, simple_df):
        _, _, seed_genes = build_graph(simple_df)
        assert len(seed_genes) == 2


class TestBuildGraphEdgeWeights:
    """test_build_graph_edge_weights: edge weight is average of causal pred_proba."""

    def test_single_causal_prediction(self, simple_df):
        """GENE_A has one causal prediction with proba=0.9."""
        G, target_id, _ = build_graph(simple_df)
        assert G[target_id]["GENE_A"]["weight"] == pytest.approx(0.9)

    def test_average_of_multiple_causal(self):
        """Two causal predictions for the same gene should be averaged."""
        df = _make_df([
            ("GENE_X", "DIS", "Flu", 0.8, 1, "SYM1"),
            ("GENE_X", "DIS", "Flu", 0.6, 1, "SYM1"),
        ])
        G, target_id, _ = build_graph(df)
        assert G[target_id]["GENE_X"]["weight"] == pytest.approx(0.7)

    def test_noncausal_weight_used_when_no_causal(self, simple_df):
        """GENE_C is non-causal (pred_label=0) so weight is nc_sum/nc_count."""
        G, target_id, _ = build_graph(simple_df)
        assert G[target_id]["GENE_C"]["weight"] == pytest.approx(0.3)

    def test_causal_overrides_noncausal(self):
        """If both causal and non-causal exist, causal average wins."""
        df = _make_df([
            ("GENE_X", "DIS", "Flu", 0.2, 0, "SYM1"),
            ("GENE_X", "DIS", "Flu", 0.8, 1, "SYM1"),
        ])
        G, target_id, _ = build_graph(df)
        # causal avg = 0.8/1 = 0.8  (not blended with the 0.2 non-causal)
        assert G[target_id]["GENE_X"]["weight"] == pytest.approx(0.8)
