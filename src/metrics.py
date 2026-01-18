from typing import Dict
import torch
from sklearn.metrics import accuracy_score


def classification_metrics(
    y_true: torch.Tensor,
    y_pred: torch.Tensor
) -> Dict[str, float]:
    """
    Computes basic classification metrics.

    Args:
        y_true: ground truth labels (Tensor)
        y_pred: predicted labels (Tensor)

    Returns:
        dict with accuracy
    """
    y_true_np = y_true.detach().cpu().numpy()
    y_pred_np = y_pred.detach().cpu().numpy()

    acc = accuracy_score(y_true_np, y_pred_np)

    return {
        "acc": acc
    }
