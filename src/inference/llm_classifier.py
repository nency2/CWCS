"""
LLM-based causal relationship classifier.
"""

import os
import json
import datetime
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Literal, Optional
from tqdm import tqdm

try:
    from openai import OpenAI
    import instructor
    from pydantic import BaseModel, Field, model_validator
    import tiktoken
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False

from ..config import VLLM_ENDPOINT, VLLM_API_KEY, VLLM_MODEL, MAX_TOKENS, MAX_CHARS


class NpEncoder(json.JSONEncoder):
    """Custom encoder to handle Pandas/Numpy types during JSON logging."""

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


if LLM_AVAILABLE:
    class CausalAnalysis(BaseModel):
        """Pydantic model for LLM response validation."""

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
                return {"relationship": "Not causal", "score": 0.0}

            if isinstance(data, dict):
                if "data" in data and isinstance(data["data"], dict):
                    data = data["data"]
                elif "result" in data and isinstance(data["result"], dict):
                    data = data["result"]

            return data
else:
    CausalAnalysis = None


SYSTEM_PROMPT = """You are a biomedical expert. Your task is to determine whether a Causal relationship exists between @GeneSrc$ and @DiseaseTgt$ in overall text data provided to you and your confidence score for the prediction.

You will be provided with abstracts from various published papers and predictions from BioBERT+SVM for each abstract, including a 'Label' (1=Causal, 0=Not Causal) and a 'Confidence Score'.

**Instructions:**
- Analyze all the text evidence for explicit causal links.
- Use the provided 'Prior Model Predictions' as supplementary signals.
- Associations are NOT necessarily causal relations. Look for explicit mention of causal link.
- score value depends on strength of evidence, try to be variable in score based on evidence

Respond ONLY with valid JSON in this exact format:
{
  "relationship": "Causal" or "Not causal",
  "score": <float between 0 and 1>
}
"""


def build_user_prompt(evidence_context: str, gene: str, disease: str) -> str:
    """Build the user prompt for LLM inference."""
    return (
        f"Gene: {gene}\n"
        f"Disease: {disease}\n\n"
        f"EVIDENCE & PRIOR PREDICTIONS:\n"
        f"{evidence_context}\n\n"
        f"Task: Does {gene} cause {disease}?\n"
        f"Consider both the text semantics and the provided model confidence scores.\n"
        f"Return only valid JSON."
    )


def create_client():
    """Create and return the LLM client."""
    if not LLM_AVAILABLE:
        raise ImportError("LLM dependencies not installed. Install with: pip install openai instructor tiktoken pydantic")

    base_client = OpenAI(
        base_url=VLLM_ENDPOINT,
        api_key=VLLM_API_KEY
    )
    return instructor.from_openai(base_client, mode=instructor.Mode.JSON)


def call_llm(
    item: Dict[str, Any],
    client,
    tokenizer,
    log_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Call LLM for a single gene-disease pair.

    Args:
        item: Dictionary with id1, id2, evidence_context, original_label
        client: Instructor-wrapped OpenAI client
        tokenizer: Tiktoken tokenizer
        log_path: Optional path to log responses

    Returns:
        Dictionary with prediction results
    """
    id1, id2 = item["id1"], item["id2"]
    evidence = item["evidence_context"]
    original_label = item["original_label"]

    # Apply token/char limits
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

        if log_path:
            log_entry = {
                "timestamp": datetime.datetime.now().isoformat(),
                "id1": str(id1),
                "id2": str(id2),
                "original_label": original_label,
                "evidence_snippet": evidence[:200],
                "response": resp.model_dump()
            }
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, cls=NpEncoder) + "\n")

        return {
            "id1": id1,
            "id2": id2,
            "original_label": original_label,
            "llm_relationship": resp.relationship,
            "llm_score": resp.score,
            "llm_reasoning": getattr(resp, 'reasoning', "")
        }

    except Exception as e:
        print(f"\nError processing {id1}-{id2}: {str(e)}")

        if log_path:
            with open(log_path, "a", encoding="utf-8") as f:
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


def process_all_pairs(
    grouped_data: List[Dict[str, Any]],
    log_path: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Process all gene-disease pairs through the LLM.

    Args:
        grouped_data: List of dictionaries with id1, id2, evidence_context, original_label
        log_path: Optional path to log responses

    Returns:
        List of prediction results
    """
    if not LLM_AVAILABLE:
        raise ImportError("LLM dependencies not installed")

    client = create_client()

    try:
        tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
    except:
        tokenizer = tiktoken.get_encoding("cl100k_base")

    results = []
    for item in tqdm(grouped_data, desc="Processing pairs"):
        results.append(call_llm(item, client, tokenizer, log_path))

    return results
