"""
Neo4j export functionality for gene-disease graphs.
"""

import networkx as nx
from typing import Optional

try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False

from ..config import NEO4J_URI, NEO4J_AUTH


def export_to_neo4j(G: nx.DiGraph, disease_name: str) -> None:
    """
    Export graph to Neo4j database.

    Args:
        G: NetworkX DiGraph to export
        disease_name: Name of the disease (used as context)
    """
    if G is None:
        return

    if not NEO4J_AVAILABLE:
        print("    Neo4j driver not installed. Skipping export.")
        return

    print(f"  Exporting graph for {disease_name} to Neo4j...")

    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)

        with driver.session() as session:
            # Create constraints
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (g:Gene) REQUIRE g.id IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (d:Disease) REQUIRE d.id IS UNIQUE")

            # Prepare Nodes
            nodes = []
            for n, d in G.nodes(data=True):
                label = "Disease" if d.get('type') == 'DISEASE' else "Gene"
                name = d.get('name', str(n))
                nodes.append({"id": str(n), "label": label, "name": name})

            # Prepare Edges
            edges = [
                {
                    "u": str(u),
                    "v": str(v),
                    "w": d['weight'],
                    "type": d.get('type', 'DIRECT')
                }
                for u, v, d in G.edges(data=True)
            ]

            # Batch insert nodes
            genes = [n for n in nodes if n['label'] == 'Gene']
            diseases = [n for n in nodes if n['label'] == 'Disease']

            if genes:
                session.run(
                    "UNWIND $batch AS r MERGE (g:Gene {id:r.id}) SET g.context=$c, g.name=r.name",
                    batch=genes, c=disease_name
                )
            if diseases:
                session.run(
                    "UNWIND $batch AS r MERGE (d:Disease {id:r.id}) SET d.name=r.name",
                    batch=diseases
                )

            # Batch insert edges
            direct = [e for e in edges if e['type'] == 'DIRECT_ASSOCIATION']
            causal = [e for e in edges if e['type'] == 'CAUSAL_REGULATION']

            if direct:
                session.run(
                    """UNWIND $b AS r MATCH (u{id:r.u}), (v{id:r.v})
                    MERGE (u)-[x:DIRECT_ASSOCIATION {disease:$c}]->(v) SET x.weight=r.w""",
                    b=direct, c=disease_name
                )
            if causal:
                session.run(
                    """UNWIND $b AS r MATCH (u{id:r.u}), (v{id:r.v})
                    MERGE (u)-[x:CAUSAL_REGULATION {disease:$c, source:'CRISPR'}]->(v) SET x.weight=r.w""",
                    b=causal, c=disease_name
                )

        driver.close()
        print(f"    Successfully exported to Neo4j.")

    except Exception as e:
        print(f"    Neo4j Export Failed: {e}")
