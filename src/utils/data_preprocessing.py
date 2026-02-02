"""
Data preprocessing utilities.
"""

import pandas as pd
from typing import List, Dict, Any, Optional


def group_evidence_data(
    data: pd.DataFrame,
    label_col: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Group data by gene-disease pairs and combine evidence.

    Args:
        data: DataFrame with sentence-level predictions
        label_col: Name of the ground truth label column

    Returns:
        List of dictionaries with grouped evidence for each pair
    """
    # Find ground truth label column
    if label_col is None:
        for col in ["label", "Label", "labels", "Labels", "ground_truth", "class"]:
            if col in data.columns:
                label_col = col
                print(f"Found ground truth label column: '{label_col}'")
                break

    # Ensure prediction columns exist
    if "pred_label" not in data.columns:
        data["pred_label"] = "N/A"
    if "pred_proba" not in data.columns:
        data["pred_proba"] = "N/A"

    grouped_data = []

    for (id1, id2), group in data.groupby(["id1", "id2"]):
        # Construct evidence string
        evidence_list = []
        for _, row in group.iterrows():
            sent = row["sentence"]
            p_label = row.get("pred_label", "N/A")
            p_proba = row.get("pred_proba", "N/A")

            evidence_entry = (
                f'- Text: "{sent}"\n'
                f"  (Prior Model Prediction: {p_label}, Confidence: {p_proba})"
            )
            evidence_list.append(evidence_entry)

        full_evidence_context = "\n".join(evidence_list)

        # Get ground truth label
        label = None
        if label_col and label_col in group.columns:
            unique_labels = group[label_col].dropna().unique()
            if len(unique_labels) >= 1:
                mode_result = group[label_col].mode()
                label = mode_result.iloc[0] if not mode_result.empty else unique_labels[0]

        grouped_data.append({
            "id1": id1,
            "id2": id2,
            "evidence_context": full_evidence_context,
            "original_label": label
        })

    return grouped_data
