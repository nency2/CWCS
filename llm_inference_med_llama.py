import os
import json
import datetime
import pandas as pd
import numpy as np
from tqdm import tqdm
from openai import OpenAI
import tiktoken
import instructor
from pydantic import BaseModel, Field, model_validator
from typing import Literal, Any

# ============================
# ==== vLLM CONFIG ===========
# ============================

VLLM_ENDPOINT = "http://localhost:8000/v1"
VLLM_API_KEY = "token-abc123"
VLLM_MODEL = "Henrychur/MMed-Llama-3-8B"

base_client = OpenAI(
    base_url=VLLM_ENDPOINT,
    api_key=VLLM_API_KEY
)

client = instructor.from_openai(base_client, mode=instructor.Mode.JSON)

# ============================
# ==== Custom JSON Encoder ===
# ============================

class NpEncoder(json.JSONEncoder):
    """Custom encoder to handle standard Pandas/Numpy types during JSON logging."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if pd.isna(obj):
            return None
        return super(NpEncoder, self).default(obj)

# ============================
# ==== Pydantic Schema =======
# ============================

class CausalAnalysis(BaseModel):
    relationship: Literal["Causal", "Not causal"] = Field(
        ..., 
        description="Classify the relationship. Value must be exactly 'Causal' or 'Not causal'."
    )
    score: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="Confidence score (0.0 to 1.0)."
    )
   

    @model_validator(mode='before')
    @classmethod
    def robust_validator(cls, data: Any) -> Any:
        if isinstance(data, list):
            return {"relationship": "Not causal", "score": 0.0, "reasoning": ""}
            
        if isinstance(data, dict):
            # Unwrap "data" or "result" wrappers
            if "data" in data and isinstance(data["data"], dict):
                data = data["data"]
            elif "result" in data and isinstance(data["result"], dict):
                data = data["result"]
                
        return data

# ============================
# ==== Data Loading ==========
# ============================

try:
    tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
except:
    tokenizer = tiktoken.get_encoding("cl100k_base")

MAX_TOKENS = 7000   # Safe limit for Input
MAX_CHARS = 25000    # Safe limit for Input

DATA_PATH = "/data/users/nency/truth_discovery/gene_discovery/gene_discovery/merged_correct_omim_filtered/complete_data_bibliometrics_with_all_diseases_biobert_svm_prediction_updated.tsv"
DATA_NAME = "all_disease"
MODEL_NAME = "MMed-Llama-3-8B"

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Could not find {DATA_PATH}")

data = pd.read_csv(DATA_PATH, sep='\t')

# Ensure prediction columns exist (fill with N/A if missing)
if "pred_label" not in data.columns:
    data["pred_label"] = "N/A"
if "pred_proba" not in data.columns:
    data["pred_proba"] = "N/A"

# Find Ground Truth Label Column
LABEL_COL = None
for col in ["label", "Label", "labels", "Labels", "ground_truth", "class"]:
    if col in data.columns:
        LABEL_COL = col
        print(f"Found ground truth label column: '{LABEL_COL}'")
        break

# ============================
# ==== Grouping Logic ========
# ============================
grouped_data = []

for (id1, id2), group in data.groupby(["id1", "id2"]):
    # Construct a detailed evidence string containing Sentence + Model Preds
    evidence_list = []
    for index, row in group.iterrows():
        sent = row["sentence"]
        p_label = row.get("pred_label", "N/A")
        p_proba = row.get("pred_proba", "N/A")
        
        # Format: Text ... | Previous Model says: ...
        evidence_entry = f"- Text: \"{sent}\"\n  (Prior Model Prediction: {p_label}, Confidence: {p_proba})"
        evidence_list.append(evidence_entry)

    # Join all evidence for this pair
    full_evidence_context = "\n".join(evidence_list)
    
    # Get ground truth (mode or first)
    label = None
    if LABEL_COL:
        unique_labels = group[LABEL_COL].dropna().unique()
        if len(unique_labels) >= 1:
            label = group[LABEL_COL].mode().iloc[0] if not group[LABEL_COL].mode().empty else unique_labels[0]
    
    grouped_data.append({
        "id1": id1,
        "id2": id2,
        "evidence_context": full_evidence_context,
        "original_label": label
    })

print(f"Processing {len(grouped_data)} unique gene-disease pairs...")

# ============================
# ==== Output Folder =========
# ============================

BASE_RESULTS_DIR = "/data/users/nency/llm_results"
os.makedirs(BASE_RESULTS_DIR, exist_ok=True)
DATESTAMP = datetime.datetime.now().strftime("%Y-%m-%d")

SUMMARY_CSV_PATH = os.path.join(
    BASE_RESULTS_DIR, f"{DATA_NAME}_{MODEL_NAME}_llm_results_{DATESTAMP}.csv"
)
LOG_JSONL_PATH = os.path.join(
    BASE_RESULTS_DIR, f"{DATA_NAME}_{MODEL_NAME}_raw_responses_{DATESTAMP}.jsonl"
)

for f in [SUMMARY_CSV_PATH, LOG_JSONL_PATH]:
    if os.path.exists(f):
        os.remove(f)

# ============================
# ==== Prompts (Aligned with GPT-4o) =======
# ============================
SYSTEM_PROMPT = """You are a biomedical expert. Your task is to determine whether a Causal relationship exists between @GeneSrc$ and @DiseaseTgt$ in overall text data provided to you and your confidence score for the prediction.

You will be provided with abstracts from various published papers and predictions from BioBERT+SVM for each abstract, including a 'Label' (1=Causal, 0=Not Causal) and a 'Confidence Score'.

**Instructions:**
- Analyze all the text evidence for explicit causal links.
- Use the provided 'Prior Model Predictions' as supplementary signals. 
- Associations are NOT necesarily causal relations. Look for explicit mention of causal link.
- score value depends on strength of evidence , try to be variable in score based on evidence
Respond ONLY with valid JSON in this exact format:
{
  "relationship": "Causal" or "Not causal",
  "score": <float between 0 and 1>,
}
"""


def build_user_prompt(evidence_context, gene, disease):
    return (
        f"Gene: {gene}\n"
        f"Disease: {disease}\n\n"
        f"EVIDENCE & PRIOR PREDICTIONS:\n"
        f"{evidence_context}\n\n"
        f"Task: Does {gene} cause {disease}?\n"
        f"Consider both the text semantics and the provided model confidence scores.\n"
        f"Return only valid JSON."
    )

# ============================
# ==== vLLM Call =============
# ============================

def call_vllm(item):
    id1, id2 = item["id1"], item["id2"]
    evidence = item["evidence_context"]
    original_label = item["original_label"]
    
    # Apply token/char limits to evidence
    if len(evidence) > MAX_CHARS:
        evidence = evidence[:MAX_CHARS]
    tokens = tokenizer.encode(evidence)
    if len(tokens) > MAX_TOKENS:
        truncated_tokens = tokens[:MAX_TOKENS]
        evidence = tokenizer.decode(truncated_tokens)
    
    try:
        resp = client.chat.completions.create(
            model=VLLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(evidence, id1, id2)}
            ],
            response_model=CausalAnalysis,
            max_tokens=500,
            temperature=0,
            max_retries=2
        )

        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "id1": str(id1),
            "id2": str(id2),
            "original_label": original_label,
            "evidence_snippet": evidence[:200],
            "response": resp.model_dump()
        }
        
        with open(LOG_JSONL_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, cls=NpEncoder) + "\n")

        return {
            "id1": id1,
            "id2": id2,
            "original_label": original_label,
            "llm_relationship": resp.relationship,
            "llm_score": resp.score,
            "llm_reasoning": resp.reasoning if hasattr(resp, 'reasoning') else ""
        }

    except Exception as e:
        print(f"\nError processing {id1}-{id2}: {str(e)}")
        with open(LOG_JSONL_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "timestamp": datetime.datetime.now().isoformat(),
                "id1": str(id1),
                "id2": str(id2),
                "original_label": original_label,
                "error": str(e)
            }, cls=NpEncoder) + "\n")
            
        return {
            "id1": id1, 
            "id2": id2, 
            "original_label": original_label,
            "llm_relationship": "Error", 
            "llm_score": 0.0,
            "llm_reasoning": str(e),
            "error_msg": str(e)
        }

# ============================
# ==== Process All ===========
# ============================

def process_all():
    results = []
    
    for item in tqdm(grouped_data, desc="Processing pairs"):
        results.append(call_vllm(item))
        
    return results

if __name__ == "__main__":
    start_time = datetime.datetime.now()
    # print(f"Processing {len(grouped_data)} unique gene-disease pairs...")

    
    results = process_all()
    
    df_results = pd.DataFrame(results)
    cols_order = ["id1", "id2", "original_label", "llm_relationship", "llm_score", "llm_reasoning", "error_msg"]
    
    # Ensure all columns exist before selecting
    for c in cols_order:
        if c not in df_results.columns:
            df_results[c] = None

    df_results = df_results[cols_order]
    df_results.to_csv(SUMMARY_CSV_PATH, index=False)
    
    print(f"\n✅ Done! Time: {datetime.datetime.now() - start_time}")
    print("📄 CSV:", SUMMARY_CSV_PATH)
    print("📄 Logs:", LOG_JSONL_PATH)
    if not df_results.empty:
        print(f"📊 Causal found: {(df_results['llm_relationship'] == 'Causal').sum()}")