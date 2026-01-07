"""
Evaluation Metrics Module

This module provides evaluation metrics for binary classification tasks,
with a focus on ranking metrics that are particularly important for
imbalanced datasets like fraud detection or "bad buy" prediction.

The primary metric is the Gini coefficient, which measures how well
the model ranks positive examples above negative ones.

Author: Ilya Sevastyanov
"""

import numpy as np
from sklearn.metrics import roc_auc_score


def compute_gini(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Compute the Gini coefficient for binary classification.
    
    The Gini coefficient is derived from the ROC AUC score:
        Gini = 2 * AUC - 1
    
    This transformation has nice properties:
        - Range: [-1, 1] instead of [0, 1] for AUC
        - Gini = 0: Random predictions (AUC = 0.5)
        - Gini = 1: Perfect predictions (AUC = 1.0)
        - Gini = -1: Perfectly wrong predictions (AUC = 0)
    
    The Gini coefficient is particularly useful in:
        - Credit scoring (assessing loan default risk)
        - Insurance (pricing and underwriting)
        - Fraud detection
        - Any binary classification with imbalanced classes
    
    Why Gini instead of accuracy?
    For imbalanced datasets (like our vehicle data with ~12% bad buys),
    accuracy can be misleading. A model predicting all zeros achieves
    88% accuracy but is useless. Gini measures ranking quality instead.
    
    Relationship to other metrics:
        - Gini = 2 * AUC - 1
        - Gini is equivalent to Mann-Whitney U statistic (normalized)
        - Gini equals the Somers' D statistic for binary outcomes
    
    Args:
        y_true: True binary labels (0 or 1)
        y_prob: Predicted probabilities for the positive class
        
    Returns:
        Gini coefficient in range [-1, 1]
        
    Example:
        >>> y_true = np.array([0, 0, 1, 1])
        >>> y_prob = np.array([0.1, 0.3, 0.7, 0.9])
        >>> gini = compute_gini(y_true, y_prob)
        >>> print(f"Gini: {gini:.4f}")  # Should be close to 1.0
    """
    auc = roc_auc_score(y_true, y_prob)
    return 2 * auc - 1


def print_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    dataset_name: str = "Dataset"
) -> dict:
    """
    Print and return key metrics for binary classification.
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        dataset_name: Name to display in output
        
    Returns:
        Dictionary with computed metrics
    """
    gini = compute_gini(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    
    print(f"{dataset_name} Results:")
    print(f"  Gini coefficient: {gini:.4f}")
    print(f"  ROC AUC: {auc:.4f}")
    
    return {'gini': gini, 'auc': auc}
