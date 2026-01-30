#!/usr/bin/env python3
"""
COMPLETE CAUSAL DISCOVERY PIPELINE (ALIGNED)
1. Data Loading: Loads bibliometric data & calculates VQ* (Strictly aligned to Script 2 logic).
2. Auto-Fetch: If CRISPR data is missing, fetches directed regulatory network from OmniPath API.
3. Graph Building: Constructs Gene-Disease (Direct) + Gene-Gene (Causal Directed) graph.
   * NOTE: Text-based Gene-Gene co-occurrence is EXCLUDED.
4. Algorithms: Runs Matrix PageRank & Geometric Mean Fusion.
5. Neo4j: Exports the resulting graph.
"""

import os
import time
import argparse
import pandas as pd
import numpy as np
import networkx as nx
import requests
from io import StringIO
# from neo4j import GraphDatabase

# ==========================================
# CONFIGURATION
# ==========================================
K_RRF = 60
# Update these paths to your actual directories
GENE_DISCOVERY_DIR = '/data/users/nency/truth_discovery/gene_discovery/gene_discovery'
OUTPUT_DIR = os.path.join(GENE_DISCOVERY_DIR, 'merged_correct_omim_filtered', 'td_threshold_analysis', 'splits')

# Learned Beta from Script 2 (User Algo)
BETA = np.array([0.64406807, 0.23051492, 0.72941017])
LAM = 0.01  # Updated from 1 to 0.01 to match Script 2

# Neo4j Config
NEO4J_URI = "neo4j+s://b0737ede.databases.neo4j.io"
NEO4J_AUTH = ("neo4j", "LNAqQLWcky30j1S5yaS2O68I3qsScy6WSR1P-GMj9KQ")

# OmniPath API
OMNIPATH_URL = "https://omnipathdb.org/interactions"

# ==========================================
# PART 1: AUTO-FETCH OMNIPATH DATA
# ==========================================

def fetch_omnipath_network(unique_genes, output_file):
    """
    Queries OmniPath for directed interactions between genes in the provided list.
    Uses n_sources (number of databases agreeing) as confidence score.
    """
    print("\n🌐 Fetching OmniPath network with confidence scores...")
    print(f"   Targeting {len(unique_genes)} genes from your dataset...")

    params = {
        'datasets': 'tf_target,omnipath,pathwayextra,kinaseextra',
        'directed': 1,
        'genesymbols': 1,
        'fields': 'sources',  # Request sources field for confidence
        'format': 'tsv'
    }

    try:
        print("   Downloading from OmniPath (this may take ~30s)...")
        r = requests.get(OMNIPATH_URL, params=params, timeout=120)
        r.raise_for_status()
        
        df_net = pd.read_csv(StringIO(r.text), sep='\t')
        
        print(f"   ✓ Downloaded {len(df_net)} raw interactions.")

        # Rename for consistency
        if 'source_genesymbol' in df_net.columns:
            df_net = df_net.rename(columns={'source_genesymbol': 'source_gene', 'target_genesymbol': 'target_gene'})
        
        # Filter to genes in our dataset
        df_net['source_gene'] = df_net['source_gene'].astype(str)
        df_net['target_gene'] = df_net['target_gene'].astype(str)
        gene_set = set([str(g) for g in unique_genes])
        
        mask = df_net['source_gene'].isin(gene_set) & df_net['target_gene'].isin(gene_set)
        filtered_df = df_net[mask].copy()
        
        print(f"   ✓ Found {len(filtered_df)} directed interactions matching your genes.")

        if len(filtered_df) == 0:
            print("   ⚠️ WARNING: No overlaps found.")
            return None

        # Calculate n_sources (number of databases) as confidence
        if 'sources' in filtered_df.columns:
            filtered_df['n_sources'] = filtered_df['sources'].apply(
                lambda x: len(str(x).split(';')) if pd.notna(x) else 1
            )
            max_sources = filtered_df['n_sources'].max()
            filtered_df['score'] = filtered_df['n_sources'] / max_sources
            
            print(f"   ✓ Using n_sources as confidence (range: 1 to {max_sources} databases)")
            print(f"   Score distribution: min={filtered_df['score'].min():.3f}, median={filtered_df['score'].median():.3f}, max={filtered_df['score'].max():.3f}")
        else:
            print("   ⚠️ Sources field not found, using uniform score=1.0")
            filtered_df['score'] = 1.0

        # Save
        final_df = filtered_df[['source_gene', 'target_gene', 'score']].copy()
        final_df.to_csv(output_file, sep='\t', index=False)
        print(f"   💾 Saved to {output_file}")
        return final_df

    except Exception as e:
        print(f"   ❌ API Request Failed: {e}")
        return None

# ==========================================
# PART 2: DATA LOADING (UPDATED TO MATCH USER ALGO)
# ==========================================

def load_unified_data(min_papers=1, file_path='complete_data_bibliometrics.tsv'):
    """
    Loads data and calculates VQ* exactly like the 'run_user_algo' script.
    """
    print(f"  📖 Loading Unified Text Data: {file_path}")
    df = pd.read_csv(file_path, sep='\t')
    print(df.columns)
    
    # --- 1. CLEANING & TYPE CASTING ---
    if 'id1' not in df.columns: raise ValueError("Missing id1 column")
    df['id1'] = df['id1'].astype(str).str.replace(r'\.0$', '', regex=True)
    
    # Fix H-Index (Extract numeric)
    if 'hindex' in df.columns:
        df['hindex'] = pd.to_numeric(df['hindex'].astype(str).str.split().str[0], errors='coerce')
    
    # Fix Citations
    if 'citations' in df.columns:
        df['citations'] = pd.to_numeric(df['citations'], errors='coerce')
        
    # Fix Year
    if 'year' in df.columns:
        df['year'] = pd.to_numeric(df['year'], errors='coerce')
    elif 'Clean_Year' in df.columns:
        df['year'] = df['Clean_Year']
    elif 'year_diff' in df.columns:
         # Fallback logic if year missing
         df['year'] = 2025 - df['year_diff'] 

    # Drop NaNs (Critical for matching results)
    original_len = len(df)
    df.dropna(subset=['hindex', 'citations', 'year'], inplace=True)
    if len(df) < original_len:
        print(f"  ⚠️ Dropped {original_len - len(df)} rows due to missing values (matching Script 2).")

    # --- 2. FEATURE NORMALIZATION ---
    # Year Logic: (2025 - year) / (2025 - min_year)
    df['year_diff_norm'] = (2025 - df['year']) / (2025 - df['year'].min())
    
    # Min-Max Scaling
    df['hindex'] = df['hindex'].astype('float64')
    df['citations_scaled'] = (df['citations'] - df['citations'].min()) / (df['citations'].max() - df['citations'].min())
    df['hindex_scaled'] = (df['hindex'] - df['hindex'].min()) / (df['hindex'].max() - df['hindex'].min())

    # --- 3. RS CALCULATION ---
    f1 = df['hindex_scaled'].values
    f2 = df['citations_scaled'].values
    f3 = df['year_diff_norm'].values
    fs = np.vstack((f1, f2, f3)).T
    df['rs'] = fs @ BETA  # Matrix Multiply

    # --- 4. VQ* CALCULATION ---
    # Ensure Prediction column
    if 'pred_label' in df.columns:
        df['Prediction'] = df['pred_label']
    elif 'Prediction' not in df.columns:
        df['Prediction'] = 0 

    gd_pair_dict = {}
    vq_star_dict = {}
    
    # Grouping
    for _, row in df.iterrows():
        # Using (disease, id1) to match pipeline downstream requirements
        gd_pair = (row.get('disease', 'unknown'), str(row['id1']))
        
        if gd_pair not in gd_pair_dict: gd_pair_dict[gd_pair] = []
        gd_pair_dict[gd_pair].append((row['rs'], row['Prediction']))
    
    # Compute
    for gd_pair, src_list in gd_pair_dict.items():
        rs_values = np.array([x[0] for x in src_list])
        vqs_values = np.array([x[1] for x in src_list]).astype(float)
        
        # Exact Formula from Script 2 (including astype casts)
        numerator = (np.sum(rs_values * (vqs_values == 1).astype(float)) + 
                     LAM * np.sum(rs_values * (vqs_values == 0).astype(float) * vqs_values))
            
        denominator = (np.sum(rs_values * (vqs_values == 1).astype(float)) + 
                       LAM * np.sum(rs_values * (vqs_values == 0).astype(float)))
        
        vq_star = numerator / denominator if denominator != 0 else 0
        vq_star_dict[gd_pair] = vq_star
    
    # Map back
    df['vq*'] = df.apply(lambda row: vq_star_dict.get((row.get('disease', 'unknown'), str(row['id1'])), 0), axis=1)
    df['vq_star_mean'] = df['vq*']  # Alias for fusion
    
    
    # Fill pred_proba for Graph weights if missing
    # if 'pred_proba' not in df.columns:
    #     df['pred_proba'] = df['Prediction'].apply(lambda x: 0.9 if x == 1 else 0.1)

    # Filter by paper count
    if min_papers > 1:
        df = df.groupby(['disease', 'id1']).filter(lambda x: len(x) >= min_papers)
           
        # print(df)
        
    return df

def load_crispr_data(file_path):
    if not os.path.exists(file_path): return None
    print(f"  🧬 Loading Directed Network: {file_path}")
    try:
        df = pd.read_csv(file_path, sep='\t')
        df['source_gene'] = df['source_gene'].astype(str).str.replace(r'\.0$', '', regex=True)
        df['target_gene'] = df['target_gene'].astype(str).str.replace(r'\.0$', '', regex=True)
        return df
    except:
        return None

# ==========================================
# PART 3: GRAPH CONSTRUCTION
# ==========================================

def build_graph(df, crispr_df=None, causal_boost=1.5):
    """
    Builds:
    1. Disease -> Gene (Direct Evidence)
    2. Gene -> Gene (Directed Causal from CRISPR/OmniPath)
    """
    if df is None or len(df) == 0: return None, None, []

    G = nx.DiGraph()
    
    # Determine target disease ID and name
    disease_counts = df['id2'].value_counts() if 'id2' in df.columns else None
    target_disease_id = disease_counts.idxmax() if disease_counts is not None else "DISEASE_UNKNOWN"
    disease_name = df['disease'].iloc[0] # Assuming 'disease' column holds the name
    
    seed_genes = set()

    # Add Disease Node
    G.add_node(target_disease_id, type='DISEASE', name=disease_name)
    
    # 1. Disease -> Gene
    for _, row in df.iterrows():
        gene_id = str(row['id1'])
        disease_id = str(row.get('id2', target_disease_id))
        gene_symbol = str(row['Symbol']) if 'Symbol' in row else gene_id
        
        G.add_node(gene_id, type='GENE', name=gene_symbol)
        
        weight = row['pred_proba']
        
        # Check if this row is causal (using 'pred_label' as requested, or 'Prediction')
        is_causal = row.get('pred_label', row.get('Prediction', 0)) == 1
        
        if is_causal:
            # weight *= causal_boost
            seed_genes.add(gene_id)
            
        if disease_id:
            # If edge doesn't exist, initialize it with zeroed stats
            if not G.has_edge(disease_id, gene_id):
                G.add_edge(disease_id, gene_id, 
                           weight=0, 
                           c_sum=0.0, c_count=0,   # Causal stats
                           nc_sum=0.0, nc_count=0, # Non-causal stats
                           type='DIRECT_ASSOCIATION')
            
            # Retrieve mutable edge data
            edge_data = G[disease_id][gene_id]
            
            # Update the appropriate stats
            if is_causal:
                edge_data['c_sum'] = edge_data.get('c_sum', 0.0) + weight
                edge_data['c_count'] = edge_data.get('c_count', 0) + 1
            else:
                edge_data['nc_sum'] = edge_data.get('nc_sum', 0.0) + weight
                edge_data['nc_count'] = edge_data.get('nc_count', 0) + 1
            
            # Logic: If ANY causal links exist, use their average.
            # Otherwise, use the average of non-causal links.
            if edge_data['c_count'] > 0:
                new_weight = edge_data['c_sum'] / edge_data['c_count']
            elif edge_data['nc_count'] > 0:
                new_weight = edge_data['nc_sum'] / edge_data['nc_count']
            else:
                new_weight = weight # Should not be reached
                
            # Update the main weight attribute
            edge_data['weight'] = new_weight
        
    # 2. Gene -> Gene (Directed Causal)
    if crispr_df is not None:
        print("    ... Injecting Causal Edges")
        valid_nodes = set(df['id1'].unique()) 
        count = 0
        for _, row in crispr_df.iterrows():
            src, tgt = row['source_gene'], row['target_gene']
            # Only add if relevant to current gene set
            if src not in valid_nodes or tgt not in valid_nodes: continue

            w = row['score']  # Use confidence score directly
            
            if G.has_edge(src, tgt):
                G[src][tgt]['weight'] += w
                G[src][tgt]['type'] = 'CAUSAL_REGULATION'
            else:
                G.add_edge(src, tgt, weight=w, type='CAUSAL_REGULATION')
            count += 1
        print(f"    ✓ Added {count} directed causal edges.")

    return G, target_disease_id, list(seed_genes)

# ==========================================
# PART 4: NEO4J EXPORT
# ==========================================

def export_to_neo4j(G, disease_name):
    if G is None: return
    print(f"  ☁️  Exporting graph for {disease_name} to Neo4j...")
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
        with driver.session() as session:
            # Constraints
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (g:Gene) REQUIRE g.id IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (d:Disease) REQUIRE d.id IS UNIQUE")
            
            # Prepare Nodes
            # Extract 'name' attribute from G
            nodes = []
            for n, d in G.nodes(data=True):
                  label = "Disease" if d.get('type') == 'DISEASE' else "Gene"
                  name = d.get('name', str(n))
                  nodes.append({"id": str(n), "label": label, "name": name})
            
            # Prepare Edges
            edges = [{"u": str(u), "v": str(v), "w": d['weight'], "type": d.get('type', 'DIRECT')} for u,v,d in G.edges(data=True)]
            
            # Batch Nodes
            genes = [n for n in nodes if n['label'] == 'Gene']
            diseases = [n for n in nodes if n['label'] == 'Disease']
            
            if genes:
                session.run("UNWIND $batch AS r MERGE (g:Gene {id:r.id}) SET g.context=$c, g.name=r.name", batch=genes, c=disease_name)
            if diseases:
                session.run("UNWIND $batch AS r MERGE (d:Disease {id:r.id}) SET d.name=r.name", batch=diseases)

            # Batch Edges
            direct = [e for e in edges if e['type'] == 'DIRECT_ASSOCIATION']
            causal = [e for e in edges if e['type'] == 'CAUSAL_REGULATION']
            
            if direct:
                session.run("""UNWIND $b AS r MATCH (u{id:r.u}), (v{id:r.v}) 
                            MERGE (u)-[x:DIRECT_ASSOCIATION {disease:$c}]->(v) SET x.weight=r.w""", b=direct, c=disease_name)
            if causal:
                session.run("""UNWIND $b AS r MATCH (u{id:r.u}), (v{id:r.v}) 
                            MERGE (u)-[x:CAUSAL_REGULATION {disease:$c, source:'CRISPR'}]->(v) SET x.weight=r.w""", b=causal, c=disease_name)
                
        driver.close()
    except Exception as e:
        print(f"    ⚠️ Neo4j Export Failed: {e}")

# ==========================================
# PART 5: ALGORITHMS (PAGERANK + FUSION)
# ==========================================

def calculate_pagerank(G, target_id, seed_genes=None, alpha=0.85):
    """ Matrix PageRank on Reverse Graph """
    if G is None or target_id not in G: return {}
    
    G_rev = G.reverse()
    for n in G_rev.nodes():
        if not G_rev.has_edge(n, n): G_rev.add_edge(n, n, weight=0.01)

    nodes = list(G_rev.nodes())
    n_nodes = len(nodes)
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    
    adj = nx.to_numpy_array(G_rev, nodelist=nodes, weight='weight')
    row_sums = adj.sum(axis=1)
    row_sums[row_sums == 0] = 1.0
    W = adj / row_sums[:, np.newaxis]
    
    p = np.zeros(n_nodes)
    if seed_genes:
        valid = [s for s in seed_genes if s in node_to_idx]
        if valid:
            if target_id in node_to_idx: p[node_to_idx[target_id]] = 0.5
            sw = 0.5 / len(valid)
            for s in valid: p[node_to_idx[s]] = sw
        elif target_id in node_to_idx: p[node_to_idx[target_id]] = 1.0
    elif target_id in node_to_idx:
        p[node_to_idx[target_id]] = 1.0
        
    if p.sum() > 0: p = p / p.sum()
    
    try:
        r = np.linalg.solve(np.eye(n_nodes) - alpha * W.T, (1 - alpha) * p)
        scores = {nodes[i]: float(r[i]) for i in range(n_nodes)}
        scores.pop(target_id, None)
        max_s = max(scores.values()) if scores else 1
        return {k: v/max_s for k, v in scores.items() if max_s > 0}
    except:
        return {}

def calculate_fusion(df):
    df['matrix_pagerank_score'] = df['matrix_pagerank_score'].fillna(0)
    df['vq_star_mean'] = df['vq_star_mean'].fillna(0)
    
    # RRF
    df['rank_pr'] = df['matrix_pagerank_score'].rank(ascending=False)
    df['rank_vq'] = df['vq_star_mean'].rank(ascending=False)
    df['rrf_score'] = (1/(K_RRF + df['rank_pr'])) + (1/(K_RRF + df['rank_vq']))
    
    # Geometric Mean (Primary Metric)
    df['geo_mean_score'] = np.sqrt(df['matrix_pagerank_score'] * df['vq_star_mean'])
    return df

# ==========================================
# MAIN EXECUTION
# ==========================================

def run_pipeline(args):
    print("="*60)
    print("  CAUSAL DISCOVERY: Text (Direct) + OmniPath (Causal)")
    print("="*60)
    
    # 1. Load Main Data
    full_df = load_unified_data(min_papers=args.min_papers, file_path=args.input_file)
    
    # Use Symbol column for OmniPath matching (OmniPath uses gene symbols, not Entrez IDs)
    if 'Symbol' in full_df.columns:
        unique_symbols = full_df['Symbol'].dropna().unique()
        # Create mapping: Symbol -> id1 (for converting back CRISPR edges)
        symbol_to_id1 = dict(zip(full_df['Symbol'].astype(str), full_df['id1'].astype(str)))
        print(f"  Found {len(unique_symbols)} unique gene symbols for OmniPath query")
    else:
        unique_symbols = full_df['id1'].unique()
        symbol_to_id1 = {str(x): str(x) for x in unique_symbols}
        print("  ⚠️ No Symbol column found, using id1 (may not match OmniPath)")
    
    # 2. Check/Fetch CRISPR Data
    if not os.path.exists(args.crispr_file):
        crispr_df = fetch_omnipath_network(unique_symbols, args.crispr_file)
    else:
        crispr_df = load_crispr_data(args.crispr_file)
    
    # Convert CRISPR symbols to id1 for graph building
    if crispr_df is not None and len(crispr_df) > 0:
        crispr_df['source_gene'] = crispr_df['source_gene'].map(lambda x: symbol_to_id1.get(str(x), x))
        crispr_df['target_gene'] = crispr_df['target_gene'].map(lambda x: symbol_to_id1.get(str(x), x))
        
    diseases = full_df['disease'].unique()
    all_scores = {}
    
    # 3. Process Each Disease
    for disease in diseases:
        print(f"\n👉 Processing: {disease}")
        df_sub = full_df[full_df['disease'] == disease].copy()
        
        # Build
        G, target_id, seed_genes = build_graph(df_sub, crispr_df=crispr_df)
        
        # Export
        export_to_neo4j(G, disease)
        
        # Analyze
        scores = calculate_pagerank(G, target_id, seed_genes)
        for gene, score in scores.items():
            all_scores[(disease, str(gene))] = score

    # 4. Update Output Files
    print(f"\n💾 Saving results to: {args.output_dir}")
    if not os.path.exists(args.output_dir): os.makedirs(args.output_dir)

    for split in ['train', 'val', 'test']:
        file_path = os.path.join(args.output_dir, f'{split}_aggregated.tsv')
        if not os.path.exists(file_path): continue
        
        df_split = pd.read_csv(file_path, sep='\t')
        df_split['matrix_pagerank_score'] = df_split.apply(lambda x: all_scores.get((x['disease'], str(x['id1'])), 0.0), axis=1)
        
        # Merge external VQ scores if present
        if args.vq_scores and os.path.exists(args.vq_scores):
             vq_df = pd.read_csv(args.vq_scores, sep='\t')
             vq_map = dict(zip(zip(vq_df.disease, vq_df.id1.astype(str).str.replace(r'\.0$', '', regex=True)), vq_df.vq_star_fixed))
             df_split['vq_star_mean'] = df_split.apply(lambda x: vq_map.get((x['disease'], str(x['id1'])), x.get('vq_star_mean', 0)), axis=1)

        df_split = calculate_fusion(df_split)
        df_split.to_csv(file_path, sep='\t', index=False)
        print(f"    ✓ Updated {split} set.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, default='complete_data_bibliometrics_with_all_diseases_biobert_svm_prediction.tsv')
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR)
    parser.add_argument('--vq_scores', type=str, default='new_vq_star_scores.tsv')
    parser.add_argument('--crispr_file', type=str, default='crispr_gene_regulatory_network.tsv')
    parser.add_argument('--min_papers', type=int, default=1)
    
    args = parser.parse_args()
    run_pipeline(args)