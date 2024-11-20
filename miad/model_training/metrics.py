"""
Binary Classification Metrics Module

This module provides comprehensive metrics for evaluating binary classifier performance,
including threshold-dependent metrics, probability-based metrics, and imbalanced
dataset metrics.
"""
# Standard library imports
from typing import Dict, Tuple, Union

# Third-party imports
import numpy as np
import pandas as pd
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score
)


def max_f1_score(
        y_true: Union[list, np.ndarray, pd.Series],
        y_proba: Union[list, np.ndarray, pd.Series]
) -> Tuple[float, float]:
    """
    Compute the threshold that yields the maximum F1 score for binary classification.

    Args:
        y_true: Actual binary labels (0 or 1)
        y_proba: Predicted probabilities (between 0 and 1)

    Returns:
        Tuple containing:
            - optimal_threshold: Threshold that yields the highest F1 score
            - best_f1: Highest F1 score achieved

    Raises:
        ValueError: If inputs are invalid
        RuntimeError: If computation fails
    """
    try:
        # Convert inputs to numpy arrays
        y_true_arr = np.asarray(y_true)
        y_proba_arr = np.asarray(y_proba)

        # Validate inputs
        if len(y_true_arr) != len(y_proba_arr):
            raise ValueError("Length mismatch between y_true and y_proba")

        # Generate thresholds
        thresholds = np.arange(0, 1, 0.001)

        # Find optimal threshold and score
        scores = [f1_score(y_true, (y_proba >= t).astype('int')) for t in thresholds]
        best_idx = np.argmax(scores)
        optimal_threshold = thresholds[best_idx]
        best_f1 = scores[best_idx]

        return optimal_threshold, best_f1

    except Exception as e:
        raise RuntimeError(f"Error computing maximum F1 score: {str(e)}")


def binary_classifier_metrics(
        y_true: Union[list, np.ndarray, pd.Series],
        y_proba: Union[list, np.ndarray, pd.Series],
        threshold: float
) -> Dict[str, float]:
    """
    Compute and summarize performance metrics for a binary classifier.

    Args:
        y_true: Actual binary labels (0 or 1)
        y_proba: Predicted probabilities (between 0 and 1)
        threshold: Classification threshold

    Returns:
        Dictionary containing performance metrics:
            - G-mean: Geometric mean of sensitivity and specificity
            - Recall/TPR: True Positive Rate (sensitivity)
            - Precision: Positive Predictive Value
            - Accuracy: Overall accuracy
            - F0.5 Score: F-score with beta=0.5
            - F1 Score: Harmonic mean of precision and recall
            - F2 Score: F-score with beta=2
            - TNR: True Negative Rate (specificity)
            - FPR: False Positive Rate
            - FNR: False Negative Rate
            - ROC AUC: Area under ROC curve
            - PR AUC: Area under Precision-Recall curve
            - Brier Score: Mean squared error of predictions

    Raises:
        ValueError: If inputs have different lengths or invalid values
        RuntimeError: If metric calculation fails
    """
    try:
        # Input validation
        if len(y_true) != len(y_proba):
            raise ValueError("Length mismatch between y_true and y_proba")
        if not 0 <= threshold <= 1:
            raise ValueError("Threshold must be between 0 and 1")

        # Convert inputs to numpy arrays
        y_true_arr = np.array(y_true)
        y_proba_arr = np.array(y_proba)

        # Generate binary predictions
        y_pred = (y_proba_arr >= threshold).astype('int')

        # Calculate confusion matrix elements
        tp = ((y_true_arr == 1) & (y_pred == 1)).sum()
        tn = ((y_true_arr == 0) & (y_pred == 0)).sum()
        fp = ((y_true_arr == 0) & (y_pred == 1)).sum()
        fn = ((y_true_arr == 1) & (y_pred == 0)).sum()

        # Calculate metrics
        metrics_dict = {
            'G-mean': geometric_mean_score(y_true_arr, y_pred),
            'Recall': recall_score(y_true_arr, y_pred),
            'Precision': precision_score(y_true_arr, y_pred),
            'Accuracy': accuracy_score(y_true_arr, y_pred),
            'F0.5 Score': fbeta_score(y_true_arr, y_pred, beta=0.5),
            'F1 Score': f1_score(y_true_arr, y_pred),
            'F2 Score': fbeta_score(y_true_arr, y_pred, beta=2),
            'TPR': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'TNR': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'FPR': fp / (fp + tn) if (fp + tn) > 0 else 0,
            'FNR': fn / (fn + tp) if (fn + tp) > 0 else 0,
            'ROC AUC': roc_auc_score(y_true_arr, y_proba_arr),
            'PR AUC': average_precision_score(y_true_arr, y_proba_arr),
            'Brier Score': brier_score_loss(y_true_arr, y_proba_arr)
        }

        return metrics_dict

    except Exception as e:
        raise RuntimeError(f"Error calculating metrics: {str(e)}")
