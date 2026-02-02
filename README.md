# CWCS - Corpus-Wide Causal Scoring Framework

A framework for determining causal scores of Gene-Disease pairs by aggregating causal scores from MultiCens and Causal Scoring Truth Discovery (CSTD) algorithms.

## Overview

CWCS combines multiple approaches to identify causal relationships between genes and diseases:

1. **Text-based Evidence**: Processes bibliometric data from scientific literature
2. **Network Analysis**: Integrates directed regulatory networks from OmniPath
3. **LLM Inference**: Uses large language models for causal relationship classification
4. **Score Fusion**: Combines PageRank and VQ* scores using RRF and geometric mean

## Project Structure

```
CWCS/
├── src/                          # Source code
│   ├── algorithms/               # Scoring algorithms
│   │   ├── pagerank.py          # Matrix PageRank implementation
│   │   └── fusion.py            # RRF and Geometric Mean fusion
│   ├── data/                     # Data loading modules
│   │   ├── loader.py            # Bibliometric data loading
│   │   └── omnipath.py          # OmniPath API integration
│   ├── graph/                    # Graph construction
│   │   ├── builder.py           # NetworkX graph building
│   │   └── neo4j_export.py      # Neo4j database export
│   ├── inference/                # LLM inference
│   │   └── llm_classifier.py    # Causal relationship classifier
│   ├── utils/                    # Utility functions
│   │   └── data_preprocessing.py
│   ├── config.py                 # Configuration settings
│   └── pipeline.py               # Main pipeline orchestration
├── notebooks/                    # Jupyter notebooks
│   ├── CWCS_algo_and_plot_code.ipynb
│   ├── AD_complete_application.ipynb
│   ├── PD_complete_application.ipynb
│   └── LLM_Inference_All_disease.ipynb
├── data/                         # Data directory
│   ├── raw/                      # Raw input data
│   └── results/                  # Output results
├── examples/                     # Example scripts
├── run_pipeline.py               # Pipeline entry point
├── run_llm_inference.py          # LLM inference entry point
├── requirements.txt              # Python dependencies
├── setup.py                      # Package setup
└── LICENSE                       # GPL-3.0 License
```

## Installation

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/nency2/CWCS.git
cd CWCS

# Install core dependencies
pip install -r requirements.txt

# Or install as a package
pip install -e .
```

### Optional Dependencies

```bash
# For Neo4j integration
pip install -e ".[neo4j]"

# For LLM inference
pip install -e ".[llm]"

# For Jupyter notebooks
pip install -e ".[notebooks]"

# All optional dependencies
pip install -e ".[neo4j,llm,notebooks]"
```

## Usage

### Running the Pipeline

```bash
# Basic usage
python run_pipeline.py --input_file data/raw/bibliometrics.tsv --output_dir data/results/

# With all options
python run_pipeline.py \
    --input_file data/raw/bibliometrics.tsv \
    --output_dir data/results/ \
    --crispr_file data/raw/crispr_network.tsv \
    --min_papers 2 \
    --export_neo4j
```

### Running LLM Inference

```bash
python run_llm_inference.py \
    --input_file data/raw/sentences.tsv \
    --output_dir data/results/ \
    --model_name MMed-Llama-3-8B
```

### Python API

```python
from src.pipeline import run_pipeline
from src.data import load_unified_data
from src.algorithms import calculate_pagerank, calculate_fusion
from src.graph import build_graph

# Load data
df = load_unified_data('data.tsv', min_papers=1)

# Build graph
G, target_id, seed_genes = build_graph(df)

# Calculate scores
scores = calculate_pagerank(G, target_id, seed_genes)

# Or run the complete pipeline
run_pipeline(
    input_file='data.tsv',
    output_dir='results/',
    min_papers=1
)
```

## Algorithm Details

### VQ* Calculation

The VQ* (Veracity Quality Star) score aggregates source reliability:

```
VQ* = (Σ rs_i * I(causal) + λ * Σ rs_i * I(non-causal) * vq) /
      (Σ rs_i * I(causal) + λ * Σ rs_i * I(non-causal))
```

Where:
- `rs` = Reliability score based on h-index, citations, and publication year
- `λ` = Regularization parameter (default: 0.01)

### PageRank

Matrix PageRank on the reversed Gene-Disease graph with personalization:
- Disease node receives 50% of personalization
- Seed genes (known causal) share remaining 50%

### Score Fusion

Two fusion methods are implemented:

1. **RRF (Reciprocal Rank Fusion)**:
   ```
   RRF = 1/(k + rank_PR) + 1/(k + rank_VQ)
   ```

2. **Geometric Mean** (Primary):
   ```
   GeoMean = sqrt(PageRank * VQ*)
   ```

## Data Format

### Input Data (Bibliometrics TSV)

Required columns:
- `id1`: Gene ID (Entrez)
- `id2`: Disease ID (OMIM)
- `disease`: Disease name
- `sentence`: Text evidence
- `hindex`: Author h-index
- `citations`: Citation count
- `year`: Publication year
- `pred_label`: Binary prediction (0/1)
- `pred_proba`: Prediction probability

Optional:
- `Symbol`: Gene symbol (for OmniPath matching)

### CRISPR/Regulatory Network TSV

Required columns:
- `source_gene`: Regulator gene
- `target_gene`: Target gene
- `score`: Confidence score

## Notebooks

- **CWCS_algo_and_plot_code.ipynb**: Core algorithm implementation and visualizations
- **AD_complete_application.ipynb**: Alzheimer's Disease analysis
- **PD_complete_application.ipynb**: Parkinson's Disease analysis
- **LLM_Inference_All_disease.ipynb**: LLM-based inference experiments

## Configuration

Edit `src/config.py` to customize:

```python
# Algorithm parameters
K_RRF = 60                    # RRF constant
BETA = [0.644, 0.231, 0.729]  # Feature weights
LAM = 0.01                    # Lambda for VQ*

# Neo4j connection
NEO4J_URI = "neo4j+s://..."
NEO4J_AUTH = ("user", "password")

# LLM settings
VLLM_ENDPOINT = "http://localhost:8000/v1"
VLLM_MODEL = "Henrychur/MMed-Llama-3-8B"
```

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## Citation

If you use CWCS in your research, please cite:

```bibtex
@software{cwcs2024,
  title={CWCS: Corpus-Wide Causal Scoring Framework},
  author={CWCS Team},
  year={2024},
  url={https://github.com/nency2/CWCS}
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
