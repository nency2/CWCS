"""
Tests for src.graph.neo4j_export module.
"""

import unittest
from unittest.mock import patch, MagicMock, call

import networkx as nx


class TestExportToNeo4jNoDriver(unittest.TestCase):
    """Test export_to_neo4j when Neo4j driver is not available."""

    @patch("src.graph.neo4j_export.NEO4J_AVAILABLE", False)
    def test_export_to_neo4j_no_driver(self):
        """When NEO4J_AVAILABLE is False, verify function returns without error."""
        from src.graph.neo4j_export import export_to_neo4j

        G = nx.DiGraph()
        G.add_node("gene1", type="GENE", name="TP53")
        G.add_edge("gene1", "disease1", weight=0.5, type="DIRECT_ASSOCIATION")

        # Should not raise any exception; just prints a skip message and returns
        try:
            export_to_neo4j(G, "TestDisease")
        except Exception as e:
            self.fail(f"export_to_neo4j raised {type(e).__name__} when NEO4J_AVAILABLE is False: {e}")


class TestExportToNeo4jNoneGraph(unittest.TestCase):
    """Test export_to_neo4j when graph is None."""

    def test_export_to_neo4j_none_graph(self):
        """Pass None graph, verify returns without error."""
        from src.graph.neo4j_export import export_to_neo4j

        # Should return immediately without error
        try:
            export_to_neo4j(None, "TestDisease")
        except Exception as e:
            self.fail(f"export_to_neo4j raised {type(e).__name__} with None graph: {e}")


class TestExportToNeo4jMocked(unittest.TestCase):
    """Test export_to_neo4j with mocked Neo4j driver."""

    @patch("src.graph.neo4j_export.NEO4J_AVAILABLE", True)
    @patch("src.graph.neo4j_export.GraphDatabase")
    def test_export_to_neo4j_mocked(self, mock_graph_db_cls):
        """Mock GraphDatabase.driver, verify session.run is called with expected Cypher queries."""
        from src.graph.neo4j_export import export_to_neo4j

        # Build a small test graph
        G = nx.DiGraph()
        G.add_node("gene1", type="GENE", name="TP53")
        G.add_node("gene2", type="GENE", name="BRCA1")
        G.add_node("disease1", type="DISEASE", name="Breast Cancer")
        G.add_edge("gene1", "disease1", weight=0.7, type="DIRECT_ASSOCIATION")
        G.add_edge("gene2", "gene1", weight=0.4, type="CAUSAL_REGULATION")

        # Setup mock driver -> session
        mock_session = MagicMock()
        mock_driver = MagicMock()
        mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)
        mock_graph_db_cls.driver.return_value = mock_driver

        export_to_neo4j(G, "Breast Cancer")

        # Verify driver was created
        mock_graph_db_cls.driver.assert_called_once()

        # Collect all Cypher query strings passed to session.run
        run_calls = mock_session.run.call_args_list
        cypher_queries = [c[0][0] for c in run_calls]

        # Verify constraint creation queries
        constraint_queries = [q for q in cypher_queries if "CREATE CONSTRAINT" in q]
        self.assertEqual(len(constraint_queries), 2, "Expected 2 constraint queries (Gene and Disease)")

        gene_constraint = [q for q in constraint_queries if "Gene" in q]
        disease_constraint = [q for q in constraint_queries if "Disease" in q]
        self.assertTrue(len(gene_constraint) >= 1, "Expected Gene constraint query")
        self.assertTrue(len(disease_constraint) >= 1, "Expected Disease constraint query")

        # Verify node insertion queries (UNWIND ... MERGE)
        merge_queries = [q for q in cypher_queries if "MERGE" in q and "UNWIND" in q]
        self.assertTrue(len(merge_queries) >= 2, "Expected at least 2 UNWIND/MERGE queries (genes + diseases)")

        # Verify edge insertion queries
        edge_queries = [q for q in cypher_queries if "DIRECT_ASSOCIATION" in q or "CAUSAL_REGULATION" in q]
        self.assertTrue(len(edge_queries) >= 2, "Expected edge queries for DIRECT_ASSOCIATION and CAUSAL_REGULATION")

        # Verify driver was closed
        mock_driver.close.assert_called_once()


class TestExportToNeo4jMockedOnlyDirect(unittest.TestCase):
    """Test export with only direct association edges (no causal)."""

    @patch("src.graph.neo4j_export.NEO4J_AVAILABLE", True)
    @patch("src.graph.neo4j_export.GraphDatabase")
    def test_export_only_direct_edges(self, mock_graph_db_cls):
        """Verify only DIRECT_ASSOCIATION edge query runs when no CAUSAL edges exist."""
        from src.graph.neo4j_export import export_to_neo4j

        G = nx.DiGraph()
        G.add_node("gene1", type="GENE", name="TP53")
        G.add_node("disease1", type="DISEASE", name="Lung Cancer")
        G.add_edge("gene1", "disease1", weight=0.5, type="DIRECT_ASSOCIATION")

        mock_session = MagicMock()
        mock_driver = MagicMock()
        mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)
        mock_graph_db_cls.driver.return_value = mock_driver

        export_to_neo4j(G, "Lung Cancer")

        run_calls = mock_session.run.call_args_list
        cypher_queries = [c[0][0] for c in run_calls]

        # Should have DIRECT_ASSOCIATION edge query but NOT CAUSAL_REGULATION
        direct_queries = [q for q in cypher_queries if "DIRECT_ASSOCIATION" in q]
        causal_queries = [q for q in cypher_queries if "CAUSAL_REGULATION" in q]
        self.assertTrue(len(direct_queries) >= 1, "Expected DIRECT_ASSOCIATION edge query")
        self.assertEqual(len(causal_queries), 0, "Expected no CAUSAL_REGULATION edge query")


if __name__ == "__main__":
    unittest.main()
