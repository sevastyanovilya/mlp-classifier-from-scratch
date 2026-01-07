"""
Activation Functions Module

This module provides implementations of common activation functions used in neural networks,
along with their analytical derivatives for backpropagation. Each function is implemented
as a static method for easy access without instantiation.

The derivatives are essential for computing gradients during backpropagation:
    dL/dz = dL/da * da/dz, where da/dz is the activation derivative

Author: Ilya Sevastyanov
"""

import numpy as np


class ActivationFunctions:
    """
    A collection of activation functions and their derivatives for neural networks.
    
    All methods are static to allow direct usage without class instantiation.
    Each activation function has a corresponding derivative function for backpropagation.
    """

    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        """
        Sigmoid activation function: σ(x) = 1 / (1 + exp(-x))
        
        The sigmoid function "squashes" any real-valued input into the range (0, 1),
        making it historically popular for binary classification outputs and hidden layers.
        
        Properties:
            - Output range: (0, 1)
            - Centered at 0.5 when x = 0
            - Smooth and differentiable everywhere
            - Can suffer from vanishing gradients for |x| > 5
        
        Args:
            x: Input array of any shape
            
        Returns:
            Array of same shape with values in (0, 1)
        """
        # Clip to prevent numerical overflow in exp()
        x = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
        """
        Derivative of sigmoid function: σ'(x) = σ(x) * (1 - σ(x))
        
        This elegant form is derived from the quotient rule:
            d/dx [1/(1+e^-x)] = e^-x / (1+e^-x)^2 = σ(x)(1-σ(x))
        
        The maximum value of σ'(x) is 0.25 (at x=0), which contributes to
        the vanishing gradient problem in deep networks.
        
        Args:
            x: Input array (pre-activation values, z, not post-activation a)
            
        Returns:
            Array of derivatives at each point
        """
        s = ActivationFunctions.sigmoid(x)
        return s * (1 - s)

    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        """
        Rectified Linear Unit: ReLU(x) = max(0, x)
        
        ReLU is the most popular activation for modern neural networks due to:
            - Computational efficiency (simple thresholding)
            - Sparsity (many neurons output exactly 0)
            - Reduced vanishing gradient (gradient is 1 for positive inputs)
        
        Potential issue: "Dying ReLU" - neurons that get stuck outputting 0
        
        Args:
            x: Input array of any shape
            
        Returns:
            Array with negative values replaced by 0
        """
        return np.maximum(0, x)

    @staticmethod
    def relu_derivative(x: np.ndarray) -> np.ndarray:
        """
        Derivative of ReLU: ReLU'(x) = 1 if x > 0, else 0
        
        Note: Technically undefined at x=0, but we use 0 by convention.
        This matches the subgradient used in optimization.
        
        Args:
            x: Input array (pre-activation values)
            
        Returns:
            Binary array: 1 where x > 0, 0 otherwise
        """
        return (x > 0).astype(float)

    @staticmethod
    def cosine(x: np.ndarray) -> np.ndarray:
        """
        Cosine activation function: cos(x)
        
        An unconventional but interesting activation that creates periodic
        decision boundaries. Useful for specific applications requiring
        oscillatory behavior.
        
        Properties:
            - Output range: [-1, 1]
            - Periodic with period 2π
            - Differentiable everywhere
        
        Args:
            x: Input array of any shape
            
        Returns:
            Array of cosine values
        """
        return np.cos(x)

    @staticmethod
    def cosine_derivative(x: np.ndarray) -> np.ndarray:
        """
        Derivative of cosine: cos'(x) = -sin(x)
        
        Args:
            x: Input array (pre-activation values)
            
        Returns:
            Array of negative sine values
        """
        return -np.sin(x)


def get_activation(name: str):
    """
    Factory function to get activation function and its derivative by name.
    
    Args:
        name: One of 'sigmoid', 'relu', 'cosine'
        
    Returns:
        Tuple of (activation_function, derivative_function)
        
    Raises:
        ValueError: If activation name is not recognized
    """
    activations = {
        'sigmoid': (ActivationFunctions.sigmoid, ActivationFunctions.sigmoid_derivative),
        'relu': (ActivationFunctions.relu, ActivationFunctions.relu_derivative),
        'cosine': (ActivationFunctions.cosine, ActivationFunctions.cosine_derivative),
    }
    
    if name not in activations:
        raise ValueError(f"Unknown activation: {name}. Supported: {list(activations.keys())}")
    
    return activations[name]
