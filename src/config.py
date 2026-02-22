"""
Configuration settings for CWCS pipeline.

Sensitive values (credentials, API keys) and environment-specific paths
are loaded from environment variables.  Copy .env.example to .env and
fill in your values, or export them in your shell before running.
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

# External data paths -- override via environment variables
GENE_DISCOVERY_DIR = os.environ.get(
    'CWCS_GENE_DISCOVERY_DIR',
    os.path.join(BASE_DIR, 'gene_discovery'),
)
OUTPUT_DIR = os.environ.get(
    'CWCS_OUTPUT_DIR',
    os.path.join(GENE_DISCOVERY_DIR, 'merged_correct_omim_filtered',
                 'td_threshold_analysis', 'splits'),
)

# ==================================================
# Neo4j Configuration
# ==================================================
NEO4J_URI = os.environ.get('NEO4J_URI', 'neo4j+s://localhost:7687')
NEO4J_AUTH = (
    os.environ.get('NEO4J_USER', 'neo4j'),
    os.environ.get('NEO4J_PASSWORD', ''),
)

# ==================================================
# External API Configuration
# ==================================================
OMNIPATH_URL = os.environ.get(
    'OMNIPATH_URL', 'https://omnipathdb.org/interactions'
)

# ==================================================
# LLM Configuration
# ==================================================
VLLM_ENDPOINT = os.environ.get('VLLM_ENDPOINT', 'http://localhost:8000/v1')
VLLM_API_KEY = os.environ.get('VLLM_API_KEY', '')
VLLM_MODEL = os.environ.get('VLLM_MODEL', 'Henrychur/MMed-Llama-3-8B')

# Token limits for LLM inference
MAX_TOKENS = int(os.environ.get('VLLM_MAX_TOKENS', '7000'))
MAX_CHARS = int(os.environ.get('VLLM_MAX_CHARS', '25000'))
