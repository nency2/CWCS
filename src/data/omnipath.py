"""
OmniPath API integration for fetching regulatory network data.
"""

import pandas as pd
import requests
from io import StringIO
from ..config import OMNIPATH_URL


def fetch_omnipath_network(unique_genes, output_file: str) -> pd.DataFrame:
    """
    Query OmniPath for directed interactions between genes.

    Uses n_sources (number of databases agreeing) as confidence score.

    Args:
        unique_genes: List/array of gene symbols to query
        output_file: Path to save the resulting network

    Returns:
        DataFrame with source_gene, target_gene, and score columns
    """
    print("\n  Fetching OmniPath network with confidence scores...")
    print(f"  Targeting {len(unique_genes)} genes from your dataset...")

    params = {
        'datasets': 'tf_target,omnipath,pathwayextra,kinaseextra',
        'directed': 1,
        'genesymbols': 1,
        'fields': 'sources',
        'format': 'tsv'
    }

    try:
        print("  Downloading from OmniPath (this may take ~30s)...")
        response = requests.get(OMNIPATH_URL, params=params, timeout=120)
        response.raise_for_status()

        df_net = pd.read_csv(StringIO(response.text), sep='\t')
        print(f"  Downloaded {len(df_net)} raw interactions.")

        # Rename columns for consistency
        if 'source_genesymbol' in df_net.columns:
            df_net = df_net.rename(columns={
                'source_genesymbol': 'source_gene',
                'target_genesymbol': 'target_gene'
            })

        # Filter to genes in our dataset
        df_net['source_gene'] = df_net['source_gene'].astype(str)
        df_net['target_gene'] = df_net['target_gene'].astype(str)
        gene_set = set([str(g) for g in unique_genes])

        mask = df_net['source_gene'].isin(gene_set) & df_net['target_gene'].isin(gene_set)
        filtered_df = df_net[mask].copy()

        print(f"  Found {len(filtered_df)} directed interactions matching your genes.")

        if len(filtered_df) == 0:
            print("  WARNING: No overlaps found.")
            return None

        # Calculate confidence score based on number of sources
        if 'sources' in filtered_df.columns:
            filtered_df['n_sources'] = filtered_df['sources'].apply(
                lambda x: len(str(x).split(';')) if pd.notna(x) else 1
            )
            max_sources = filtered_df['n_sources'].max()
            filtered_df['score'] = filtered_df['n_sources'] / max_sources

            print(f"  Using n_sources as confidence (range: 1 to {max_sources} databases)")
            print(f"  Score distribution: min={filtered_df['score'].min():.3f}, "
                  f"median={filtered_df['score'].median():.3f}, max={filtered_df['score'].max():.3f}")
        else:
            print("  Sources field not found, using uniform score=1.0")
            filtered_df['score'] = 1.0

        # Save results
        final_df = filtered_df[['source_gene', 'target_gene', 'score']].copy()
        final_df.to_csv(output_file, sep='\t', index=False)
        print(f"  Saved to {output_file}")

        return final_df

    except Exception as e:
        print(f"  API Request Failed: {e}")
        return None
