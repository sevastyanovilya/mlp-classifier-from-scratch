# MLP Classifier from Scratch

**Binary Classification with Custom Neural Network Implementation**

A complete implementation of a Multi-Layer Perceptron (MLP) classifier from scratch using NumPy, with comparisons to scikit-learn and PyTorch implementations. The project demonstrates fundamental deep learning concepts including backpropagation, gradient descent optimization, and the Adam optimizer.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![NumPy](https://img.shields.io/badge/NumPy-1.20+-green.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## Table of Contents

- [Problem Statement](#problem-statement)
- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Technical Details](#technical-details)
- [License](#license)

## Problem Statement

The task is binary classification for predicting vehicle purchase quality using the "Don't Get Kicked!" dataset. The goal is to identify vehicles that are at risk of being "bad buys" (IsBadBuy=1) — cars with hidden problems that make them poor investments.

The dataset contains 72,983 vehicle transactions with 33 features including vehicle specifications, pricing information, and auction details.

## Key Features

- **MLP Implementation from Scratch**: Complete forward and backward propagation using only NumPy
- **Multiple Activation Functions**: Sigmoid, ReLU, and Cosine activations with analytical derivatives
- **Xavier/Glorot Initialization**: Proper weight initialization for stable training
- **Adam Optimizer**: Custom implementation with momentum and adaptive learning rates
- **Early Stopping**: Prevents overfitting by monitoring validation loss
- **Temporal Split**: Time-based train/validation/test split to prevent data leakage
- **Framework Comparison**: Side-by-side evaluation with scikit-learn and PyTorch

## Project Structure

```
mlp-classifier-from-scratch/
├── src/
│   ├── __init__.py
│   ├── activations.py      # Activation functions and derivatives
│   ├── mlp.py              # Core MLP implementation with SGD
│   ├── mlp_adam.py         # MLP with Adam optimizer
│   ├── preprocessing.py    # Data preparation utilities
│   └── metrics.py          # Evaluation metrics (Gini coefficient)
├── data/
│   └── .gitkeep            # Download instructions in README
├── notebooks/
│   └── mlp_experiments.ipynb  # Complete experiments and analysis
├── README.md
├── requirements.txt
└── LICENSE
```

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/sevastyanovilya/mlp-classifier-from-scratch.git
cd mlp-classifier-from-scratch
pip install -r requirements.txt
```

### Data Setup

Download the training data from [Kaggle "Don't Get Kicked!" competition](https://www.kaggle.com/c/DontGetKicked/data) and place `training.csv` in the `data/` directory.

## Usage

### Quick Start with Notebook

The easiest way to explore the project is through the Jupyter notebook:

```bash
jupyter notebook notebooks/mlp_experiments.ipynb
```

### Using the MLP Classes Directly

```python
from src.mlp import MLP
from src.preprocessing import prepare_features, temporal_split

# Load and prepare data
df_train, df_valid, df_test = temporal_split(data)
X_train, X_valid, X_test, y_train, y_valid, y_test = prepare_features(
    df_train, df_valid, df_test, categorical_cols, numerical_cols
)

# Train MLP with sigmoid activation
model = MLP(
    n_hidden=100,
    activation='sigmoid',
    learning_rate=0.01,
    n_epochs=100,
    batch_size=32,
    random_state=42
)
model.fit(X_train, y_train, X_valid, y_valid)

# Get predictions
predictions = model.predict_proba(X_test)[:, 1]
```

### Using Adam Optimizer

```python
from src.mlp_adam import MLPWithAdam

model = MLPWithAdam(
    n_hidden=100,
    activation='sigmoid',
    learning_rate=0.001,
    n_epochs=200,
    batch_size=32,
    patience=20  # Early stopping patience
)
model.fit(X_train, y_train, X_valid, y_valid)
```

## Model Architecture

The implemented MLP has the following architecture:

```
Input Layer (1716 features after preprocessing)
    ↓
Hidden Layer (100 neurons, configurable activation)
    ↓
Output Layer (1 neuron, sigmoid activation)
```

### Weight Initialization

Xavier/Glorot uniform initialization is used for stable gradient flow:

```
W ~ Uniform(-limit, limit) where limit = sqrt(6 / (fan_in + fan_out))
```

### Supported Activations

| Activation | Function | Derivative |
|------------|----------|------------|
| Sigmoid | σ(x) = 1/(1+e^(-x)) | σ(x)·(1-σ(x)) |
| ReLU | max(0, x) | 1 if x > 0, else 0 |
| Cosine | cos(x) | -sin(x) |

## Results

### Model Comparison on Validation Set

| Model | Gini Coefficient | ROC AUC |
|-------|------------------|---------|
| **Custom MLP (Sigmoid)** | **0.4906** | 0.7453 |
| Custom MLP (Adam) | 0.4796 | 0.7398 |
| PyTorch MLP | 0.4893 | 0.7447 |
| sklearn MLPClassifier | 0.4575 | 0.7288 |

### Activation Function Comparison

| Activation | Gini Coefficient |
|------------|------------------|
| Sigmoid | 0.4906 |
| Cosine | 0.4884 |
| ReLU | 0.4712 |

### Final Test Set Performance

| Dataset | Gini Coefficient |
|---------|------------------|
| Train | 0.5448 |
| Validation | 0.4906 |
| Test | 0.4991 |

The small gap between validation and test Gini (0.0085) indicates good generalization across different time periods.

## Technical Details

### Temporal Split Strategy

To simulate real-world deployment where models predict future outcomes, the data is split chronologically:

- **Training Set** (33%): Jan 2009 — Sep 2009
- **Validation Set** (33%): Sep 2009 — May 2010  
- **Test Set** (33%): May 2010 — Dec 2010

### Preprocessing Pipeline

1. **Missing Values**: Categorical features filled with 'missing', numerical with median
2. **Encoding**: One-Hot Encoding for categorical features (1699 dimensions)
3. **Scaling**: StandardScaler for numerical features (17 dimensions)
4. **Final Dimensionality**: 1716 features

### Adam Optimizer Implementation

The Adam optimizer combines momentum with adaptive learning rates:

```
m_t = β₁·m_{t-1} + (1-β₁)·g_t          # First moment estimate
v_t = β₂·v_{t-1} + (1-β₂)·g_t²         # Second moment estimate
m̂_t = m_t / (1-β₁ᵗ)                    # Bias correction
v̂_t = v_t / (1-β₂ᵗ)                    # Bias correction
θ_t = θ_{t-1} - α·m̂_t / (√v̂_t + ε)    # Parameter update
```

Default hyperparameters: β₁=0.9, β₂=0.999, ε=10⁻⁸

### Why Custom Implementation Outperforms sklearn?

The custom implementation achieves better results due to:

1. **Xavier Initialization**: Specifically designed for sigmoid activations
2. **Pure SGD**: sklearn's default includes momentum (0.9) and L2 regularization (α=0.0001)
3. **Float64 Precision**: NumPy's default vs PyTorch's Float32

## Future Improvements

- [ ] Add dropout regularization
- [ ] Implement learning rate scheduling
- [ ] Support for multiple hidden layers
- [ ] Add batch normalization
- [ ] GPU acceleration with CuPy

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Dataset: [Kaggle "Don't Get Kicked!" Competition](https://www.kaggle.com/c/DontGetKicked)
- Theoretical foundation: [Deep Learning Book](https://www.deeplearningbook.org/) by Goodfellow, Bengio, and Courville
