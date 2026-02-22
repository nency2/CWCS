"""Tests for src.algorithms.pagerank.calculate_pagerank."""

import pytest
import networkx as nx

from src.algorithms.pagerank import calculate_pagerank


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _make_simple_graph():
    """One disease node with directed edges to three gene nodes,
    plus inter-gene edges so PageRank can propagate among genes."""
    G = nx.DiGraph()
    G.add_node("DIS", type="DISEASE")
    for gene in ("G1", "G2", "G3"):
        G.add_node(gene, type="GENE")
        G.add_edge("DIS", gene, weight=0.5, type="DIRECT_ASSOCIATION")
    # Add gene-to-gene edges to allow rank to flow among genes
    G.add_edge("G1", "G2", weight=0.3, type="CAUSAL_REGULATION")
    G.add_edge("G2", "G3", weight=0.3, type="CAUSAL_REGULATION")
    G.add_edge("G3", "G1", weight=0.3, type="CAUSAL_REGULATION")
    return G


def _make_star_graph():
    """Disease -> 3 genes, no inter-gene edges.  Used with seed genes."""
    G = nx.DiGraph()
    G.add_node("DIS", type="DISEASE")
    for gene in ("G1", "G2", "G3"):
        G.add_node(gene, type="GENE")
        G.add_edge("DIS", gene, weight=0.5, type="DIRECT_ASSOCIATION")
    return G


# ------------------------------------------------------------------ #
# Tests
# ------------------------------------------------------------------ #

class TestPageRankBasic:
    """test_pagerank_basic: build a simple DiGraph (1 disease -> 3 genes),
    call calculate_pagerank, verify returns dict with gene scores,
    disease node excluded from results."""

    def test_returns_dict(self):
        G = _make_simple_graph()
        result = calculate_pagerank(G, "DIS", seed_genes=["G1"])
        assert isinstance(result, dict)

    def test_gene_scores_present(self):
        G = _make_simple_graph()
        scores = calculate_pagerank(G, "DIS", seed_genes=["G1"])
        for gene in ("G1", "G2", "G3"):
            assert gene in scores

    def test_disease_excluded(self):
        G = _make_simple_graph()
        scores = calculate_pagerank(G, "DIS", seed_genes=["G1"])
        assert "DIS" not in scores

    def test_scores_are_positive(self):
        G = _make_simple_graph()
        scores = calculate_pagerank(G, "DIS", seed_genes=["G1"])
        for v in scores.values():
            assert v > 0


class TestPageRankWithSeedGenes:
    """test_pagerank_with_seed_genes: provide seed genes, verify
    personalization affects scores."""

    def test_seed_gene_boosted(self):
        """A seed gene that gets personalization weight should score higher
        than an identical non-seed gene (all else equal)."""
        G = _make_star_graph()
        scores_with_seed = calculate_pagerank(G, "DIS", seed_genes=["G1"])
        # G1 should score >= G2 because it receives personalization weight
        assert scores_with_seed["G1"] >= scores_with_seed["G2"]

    def test_different_from_no_seed(self):
        """When seed genes are provided the personalization vector changes,
        so scores should differ from providing a different seed set."""
        G = _make_simple_graph()
        scores_seed_g1 = calculate_pagerank(G, "DIS", seed_genes=["G1"])
        scores_seed_g2 = calculate_pagerank(G, "DIS", seed_genes=["G2"])
        # Different seed genes should produce different score distributions
        assert scores_seed_g1 != scores_seed_g2

    def test_invalid_seed_genes_ignored(self):
        """Seed genes not in the graph should not cause errors.
        Falls back to target-only personalization."""
        G = _make_simple_graph()
        scores = calculate_pagerank(G, "DIS", seed_genes=["NOT_IN_GRAPH"])
        # Should still return a dict (may be empty if no rank propagates
        # to gene nodes, but should not raise)
        assert isinstance(scores, dict)

    def test_multiple_seed_genes(self):
        """Multiple seed genes should all receive personalization weight."""
        G = _make_star_graph()
        scores = calculate_pagerank(G, "DIS", seed_genes=["G1", "G2"])
        # Both seeded genes should score higher than the non-seed
        assert scores["G1"] >= scores["G3"]
        assert scores["G2"] >= scores["G3"]


class TestPageRankEmptyGraph:
    """test_pagerank_empty_graph: pass None graph, verify returns {}."""

    def test_none_graph(self):
        assert calculate_pagerank(None, "DIS") == {}

    def test_none_graph_with_seeds(self):
        assert calculate_pagerank(None, "DIS", seed_genes=["G1"]) == {}


class TestPageRankMissingTarget:
    """test_pagerank_missing_target: pass target_id not in graph, verify returns {}."""

    def test_missing_target(self):
        G = _make_simple_graph()
        assert calculate_pagerank(G, "NOT_A_NODE") == {}

    def test_empty_graph_missing_target(self):
        G = nx.DiGraph()
        assert calculate_pagerank(G, "DIS") == {}


class TestPageRankNormalization:
    """test_pagerank_normalization: verify scores are normalized to [0,1] range."""

    def test_all_scores_in_unit_range(self):
        G = _make_simple_graph()
        scores = calculate_pagerank(G, "DIS", seed_genes=["G1"])
        for v in scores.values():
            assert 0.0 <= v <= 1.0

    def test_max_score_is_one(self):
        G = _make_simple_graph()
        scores = calculate_pagerank(G, "DIS", seed_genes=["G1"])
        assert max(scores.values()) == pytest.approx(1.0)

    def test_normalization_with_multiple_seeds(self):
        G = _make_simple_graph()
        scores = calculate_pagerank(G, "DIS", seed_genes=["G1", "G2"])
        assert max(scores.values()) == pytest.approx(1.0)
        for v in scores.values():
            assert 0.0 <= v <= 1.0
