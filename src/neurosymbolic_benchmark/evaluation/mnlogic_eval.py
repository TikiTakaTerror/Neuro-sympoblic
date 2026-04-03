"""Evaluation helpers for MNLogic models."""

from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd
import torch
from torch import nn


@torch.no_grad()
def evaluate_mnlogic_classifier(
    model: torch.nn.Module,
    loader,
    device: torch.device,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """Evaluate a binary classifier on an MNLogic split."""

    criterion = nn.CrossEntropyLoss()
    model.eval()

    total_loss = 0.0
    total_examples = 0
    total_correct = 0
    rows = []

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        concepts = batch["concepts"]
        sample_ids = batch["sample_id"]

        logits = model(images)
        loss = criterion(logits, labels)
        probabilities = torch.softmax(logits, dim=1)
        predictions = torch.argmax(probabilities, dim=1)

        batch_size = labels.size(0)
        total_examples += batch_size
        total_loss += loss.item() * batch_size
        total_correct += (predictions == labels).sum().item()

        for row_idx in range(batch_size):
            concept_values = concepts[row_idx].tolist()
            rows.append(
                {
                    "sample_id": int(sample_ids[row_idx].item()),
                    "true_label": int(labels[row_idx].item()),
                    "predicted_label": int(predictions[row_idx].item()),
                    "prob_class_0": float(probabilities[row_idx, 0].item()),
                    "prob_class_1": float(probabilities[row_idx, 1].item()),
                    "concept_1": int(concept_values[0]),
                    "concept_2": int(concept_values[1]),
                    "concept_3": int(concept_values[2]),
                    "concept_4": int(concept_values[3]),
                }
            )

    metrics = {
        "loss": total_loss / max(total_examples, 1),
        "accuracy": total_correct / max(total_examples, 1),
        "num_examples": total_examples,
    }

    predictions_df = pd.DataFrame(rows).sort_values("sample_id").reset_index(drop=True)
    return metrics, predictions_df

