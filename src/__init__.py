"""
MLP Classifier from Scratch

A complete implementation of Multi-Layer Perceptron for binary classification
using only NumPy, with comparisons to scikit-learn and PyTorch.

Modules:
    activations: Activation functions and their derivatives
    mlp: Core MLP implementation with SGD optimization
    mlp_adam: Extended MLP with Adam optimizer and early stopping
    preprocessing: Data preparation utilities
    metrics: Evaluation metrics (Gini coefficient)
"""

from .activations import ActivationFunctions, get_activation
from .mlp import MLP
from .mlp_adam import MLPWithAdam
from .preprocessing import temporal_split, prepare_features, get_feature_columns
from .metrics import compute_gini, print_metrics

__version__ = '1.0.0'
__author__ = 'Ilya Sevastyanov'

__all__ = [
    'ActivationFunctions',
    'get_activation',
    'MLP',
    'MLPWithAdam',
    'temporal_split',
    'prepare_features',
    'get_feature_columns',
    'compute_gini',
    'print_metrics',
]
