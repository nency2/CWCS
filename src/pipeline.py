"""
Main CWCS pipeline: Text (Direct) + OmniPath (Causal) discovery.
"""

import os
import argparse
import pandas as pd
from typing import Optional

from .config import OUTPUT_DIR
from .data import load_unified_data, load_crispr_data, fetch_omnipath_network
from .graph import build_graph, export_to_neo4j
from .algorithms import calculate_pagerank, calculate_fusion


def run_pipeline(
    input_file: str,
    output_dir: str = OUTPUT_DIR,
    vq_scores: Optional[str] = None,
    crispr_file: str = 'crispr_gene_regulatory_network.tsv',
    min_papers: int = 1,
    export_neo4j: bool = False
) -> None:
    """
    Run the complete CWCS causal discovery pipeline.

    Args:
        input_file: Path to input bibliometrics TSV file
        output_dir: Directory to save results
        vq_scores: Optional path to external VQ scores file
        crispr_file: Path to CRISPR/regulatory network file
        min_papers: Minimum papers per gene-disease pair
        export_neo4j: Whether to export results to Neo4j
    """
    print("=" * 60)
    print("  CAUSAL DISCOVERY: Text (Direct) + OmniPath (Causal)")
    print("=" * 60)

    # 1. Load Main Data
    full_df = load_unified_data(input_file, min_papers=min_papers)

    # Use Symbol column for OmniPath matching
    if 'Symbol' in full_df.columns:
        unique_symbols = full_df['Symbol'].dropna().unique()
        symbol_to_id1 = dict(zip(
            full_df['Symbol'].astype(str),
            full_df['id1'].astype(str)
        ))
        print(f"  Found {len(unique_symbols)} unique gene symbols for OmniPath query")
    else:
        unique_symbols = full_df['id1'].unique()
        symbol_to_id1 = {str(x): str(x) for x in unique_symbols}
        print("  No Symbol column found, using id1")

    # 2. Check/Fetch CRISPR Data
    if not os.path.exists(crispr_file):
        crispr_df = fetch_omnipath_network(unique_symbols, crispr_file)
    else:
        crispr_df = load_crispr_data(crispr_file)

    # Convert CRISPR symbols to id1
    if crispr_df is not None and len(crispr_df) > 0:
        crispr_df['source_gene'] = crispr_df['source_gene'].map(
            lambda x: symbol_to_id1.get(str(x), x)
        )
        crispr_df['target_gene'] = crispr_df['target_gene'].map(
            lambda x: symbol_to_id1.get(str(x), x)
        )

    diseases = full_df['disease'].unique()
    all_scores = {}

    # 3. Process Each Disease
    for disease in diseases:
        print(f"\n  Processing: {disease}")
        df_sub = full_df[full_df['disease'] == disease].copy()

        # Build graph
        G, target_id, seed_genes = build_graph(df_sub, crispr_df=crispr_df)

        # Export to Neo4j (optional)
        if export_neo4j:
            export_to_neo4j(G, disease)

        # Calculate PageRank
        scores = calculate_pagerank(G, target_id, seed_genes)
        for gene, score in scores.items():
            all_scores[(disease, str(gene))] = score

    # 4. Update Output Files
    print(f"\n  Saving results to: {output_dir}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for split in ['train', 'val', 'test']:
        file_path = os.path.join(output_dir, f'{split}_aggregated.tsv')
        if not os.path.exists(file_path):
            continue

        df_split = pd.read_csv(file_path, sep='\t')
        df_split['matrix_pagerank_score'] = df_split.apply(
            lambda x: all_scores.get((x['disease'], str(x['id1'])), 0.0),
            axis=1
        )

        # Merge external VQ scores if present
        if vq_scores and os.path.exists(vq_scores):
            vq_df = pd.read_csv(vq_scores, sep='\t')
            vq_map = dict(zip(
                zip(
                    vq_df.disease,
                    vq_df.id1.astype(str).str.replace(r'\.0$', '', regex=True)
                ),
                vq_df.vq_star_fixed
            ))
            df_split['vq_star_mean'] = df_split.apply(
                lambda x: vq_map.get((x['disease'], str(x['id1'])), x.get('vq_star_mean', 0)),
                axis=1
            )

        df_split = calculate_fusion(df_split)
        df_split.to_csv(file_path, sep='\t', index=False)
        print(f"    Updated {split} set.")

    print("\n  Pipeline complete!")


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description='CWCS: Corpus-Wide Causal Scoring Pipeline'
    )
    parser.add_argument(
        '--input_file', type=str,
        default='complete_data_bibliometrics_with_all_diseases_biobert_svm_prediction.tsv',
        help='Input bibliometrics TSV file'
    )
    parser.add_argument(
        '--output_dir', type=str,
        default=OUTPUT_DIR,
        help='Output directory for results'
    )
    parser.add_argument(
        '--vq_scores', type=str,
        default='new_vq_star_scores.tsv',
        help='External VQ scores file'
    )
    parser.add_argument(
        '--crispr_file', type=str,
        default='crispr_gene_regulatory_network.tsv',
        help='CRISPR/regulatory network file'
    )
    parser.add_argument(
        '--min_papers', type=int,
        default=1,
        help='Minimum papers per gene-disease pair'
    )
    parser.add_argument(
        '--export_neo4j', action='store_true',
        help='Export results to Neo4j'
    )

    args = parser.parse_args()

    run_pipeline(
        input_file=args.input_file,
        output_dir=args.output_dir,
        vq_scores=args.vq_scores,
        crispr_file=args.crispr_file,
        min_papers=args.min_papers,
        export_neo4j=args.export_neo4j
    )


if __name__ == "__main__":
    main()
