"""
Graph construction for Gene-Disease causal networks.
"""

import networkx as nx
import pandas as pd
from typing import Tuple, List, Set, Optional


def build_graph(
    df: pd.DataFrame,
    crispr_df: Optional[pd.DataFrame] = None,
    causal_boost: float = 1.5
) -> Tuple[nx.DiGraph, str, List[str]]:
    """
    Build a directed graph with Disease->Gene and Gene->Gene edges.

    Args:
        df: DataFrame with gene-disease association data
        crispr_df: Optional DataFrame with directed regulatory network
        causal_boost: Weight multiplier for causal edges (currently unused)

    Returns:
        Tuple of (graph, target_disease_id, list_of_seed_genes)
    """
    if df is None or len(df) == 0:
        return None, None, []

    G = nx.DiGraph()

    # Determine target disease
    disease_counts = df['id2'].value_counts() if 'id2' in df.columns else None
    target_disease_id = disease_counts.idxmax() if disease_counts is not None else "DISEASE_UNKNOWN"
    disease_name = df['disease'].iloc[0]

    seed_genes: Set[str] = set()

    # Add Disease Node
    G.add_node(target_disease_id, type='DISEASE', name=disease_name)

    # Add Disease -> Gene edges
    for _, row in df.iterrows():
        gene_id = str(row['id1'])
        disease_id = str(row.get('id2', target_disease_id))
        gene_symbol = str(row['Symbol']) if 'Symbol' in row else gene_id

        G.add_node(gene_id, type='GENE', name=gene_symbol)

        weight = row['pred_proba']
        is_causal = row.get('pred_label', row.get('Prediction', 0)) == 1

        if is_causal:
            seed_genes.add(gene_id)

        if disease_id:
            if not G.has_edge(disease_id, gene_id):
                G.add_edge(
                    disease_id, gene_id,
                    weight=0,
                    c_sum=0.0, c_count=0,    # Causal stats
                    nc_sum=0.0, nc_count=0,  # Non-causal stats
                    type='DIRECT_ASSOCIATION'
                )

            edge_data = G[disease_id][gene_id]

            if is_causal:
                edge_data['c_sum'] = edge_data.get('c_sum', 0.0) + weight
                edge_data['c_count'] = edge_data.get('c_count', 0) + 1
            else:
                edge_data['nc_sum'] = edge_data.get('nc_sum', 0.0) + weight
                edge_data['nc_count'] = edge_data.get('nc_count', 0) + 1

            # Calculate final weight
            if edge_data['c_count'] > 0:
                new_weight = edge_data['c_sum'] / edge_data['c_count']
            elif edge_data['nc_count'] > 0:
                new_weight = edge_data['nc_sum'] / edge_data['nc_count']
            else:
                new_weight = weight

            edge_data['weight'] = new_weight

    # Add Gene -> Gene edges from CRISPR/regulatory data
    if crispr_df is not None:
        print("    ... Injecting Causal Edges")
        valid_nodes = set(df['id1'].unique())
        count = 0

        for _, row in crispr_df.iterrows():
            src, tgt = row['source_gene'], row['target_gene']

            if src not in valid_nodes or tgt not in valid_nodes:
                continue

            w = row['score']

            if G.has_edge(src, tgt):
                G[src][tgt]['weight'] += w
                G[src][tgt]['type'] = 'CAUSAL_REGULATION'
            else:
                G.add_edge(src, tgt, weight=w, type='CAUSAL_REGULATION')
            count += 1

        print(f"    Added {count} directed causal edges.")

    return G, target_disease_id, list(seed_genes)
