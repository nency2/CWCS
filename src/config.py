"""
Configuration settings for CWCS pipeline.
"""

import os
import numpy as np

# ==================================================
# RRF (Reciprocal Rank Fusion) Configuration
# ==================================================
K_RRF = 60

# ==================================================
# Model Parameters (Learned from User Algorithm)
# ==================================================
BETA = np.array([0.64406807, 0.23051492, 0.72941017])
LAM = 0.01  # Lambda parameter for VQ* calculation

# ==================================================
# Directory Configuration
# ==================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(DATA_DIR, 'results')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')

# Default paths (update these for your environment)
GENE_DISCOVERY_DIR = '/data/users/nency/truth_discovery/gene_discovery/gene_discovery'
OUTPUT_DIR = os.path.join(GENE_DISCOVERY_DIR, 'merged_correct_omim_filtered', 'td_threshold_analysis', 'splits')

# ==================================================
# Neo4j Configuration
# ==================================================
NEO4J_URI = "neo4j+s://b0737ede.databases.neo4j.io"
NEO4J_AUTH = ("neo4j", "LNAqQLWcky30j1S5yaS2O68I3qsScy6WSR1P-GMj9KQ")

# ==================================================
# External API Configuration
# ==================================================
OMNIPATH_URL = "https://omnipathdb.org/interactions"

# ==================================================
# LLM Configuration
# ==================================================
VLLM_ENDPOINT = "http://localhost:8000/v1"
VLLM_API_KEY = "token-abc123"
VLLM_MODEL = "Henrychur/MMed-Llama-3-8B"

# Token limits for LLM inference
MAX_TOKENS = 7000
MAX_CHARS = 25000
