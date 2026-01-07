"""
Multi-Layer Perceptron (MLP) Implementation from Scratch

This module implements a fully-connected feedforward neural network with one hidden layer,
using only NumPy for matrix operations. The implementation follows the classical architecture:

    Input → Hidden Layer (with activation) → Output Layer (sigmoid for binary classification)

Key concepts demonstrated:
    - Xavier/Glorot weight initialization for stable training
    - Forward propagation with matrix operations
    - Backpropagation using the chain rule
    - Mini-batch Stochastic Gradient Descent (SGD)
    - Binary Cross-Entropy loss for classification

Mathematical notation used throughout:
    - X: Input data matrix (n_samples, n_features)
    - W1, W2: Weight matrices for layers 1 and 2
    - b1, b2: Bias vectors
    - z1, z2: Pre-activation values (linear combinations)
    - a1, a2: Post-activation values
    - dW, db: Gradients with respect to weights and biases

Author: Ilya Sevastyanov
"""

import numpy as np
from .activations import ActivationFunctions, get_activation


class MLP:
    """
    Multi-Layer Perceptron with one hidden layer for binary classification.
    
    This implementation uses:
        - Xavier/Glorot uniform initialization
        - Configurable activation function for hidden layer
        - Sigmoid activation for output (probability estimation)
        - Binary Cross-Entropy loss
        - Mini-batch SGD optimization
    
    The network architecture:
        Input (n_features) → Hidden (n_hidden, activation) → Output (1, sigmoid)
    
    Attributes:
        n_hidden: Number of neurons in hidden layer
        learning_rate: Step size for gradient descent
        n_epochs: Number of complete passes through training data
        batch_size: Number of samples per gradient update
        random_state: Seed for reproducibility
        verbose: Whether to print training progress
        
        W1, b1: Weights and biases for input → hidden
        W2, b2: Weights and biases for hidden → output
        train_losses: History of training losses per epoch
        valid_losses: History of validation losses per epoch
    """

    def __init__(
        self,
        n_hidden: int = 100,
        activation: str = 'sigmoid',
        learning_rate: float = 0.01,
        n_epochs: int = 100,
        batch_size: int = 32,
        random_state: int = 42,
        verbose: bool = True
    ):
        """
        Initialize MLP with specified hyperparameters.
        
        Args:
            n_hidden: Number of neurons in the hidden layer. More neurons increase
                     model capacity but also risk of overfitting.
            activation: Activation function name ('sigmoid', 'relu', 'cosine')
                       or a callable. Used for hidden layer only.
            learning_rate: Step size for parameter updates. Too high causes
                          divergence, too low causes slow convergence.
            n_epochs: Number of complete passes through the training data.
            batch_size: Number of samples per mini-batch. Affects gradient
                       variance and memory usage.
            random_state: Random seed for weight initialization and shuffling.
            verbose: If True, print progress every 10 epochs.
        """
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.verbose = verbose

        # Set up activation function with its derivative
        self._set_activation(activation)

        # Weights are initialized during fit() when we know input dimensions
        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None

        # Track training history for visualization and debugging
        self.train_losses = []
        self.valid_losses = []

    def _set_activation(self, activation):
        """
        Configure the activation function and its derivative.
        
        Supports both string identifiers for built-in activations and
        custom callable functions. For custom functions, numerical
        differentiation is used as a fallback.
        
        Args:
            activation: String name or callable function
        """
        if callable(activation):
            # Custom activation function provided
            self.activation = activation
            self.activation_derivative = None  # Will use numerical differentiation
            self.activation_name = getattr(activation, '__name__', 'custom')
        else:
            # Use built-in activation from our library
            self.activation, self.activation_derivative = get_activation(activation)
            self.activation_name = activation

    def _numerical_derivative(self, x: np.ndarray, eps: float = 1e-7) -> np.ndarray:
        """
        Compute numerical derivative using central difference approximation.
        
        This is a fallback for custom activation functions without an
        analytical derivative. The formula:
            f'(x) ≈ [f(x + eps) - f(x - eps)] / (2 * eps)
        
        Central difference is more accurate than forward/backward difference
        because the O(eps²) error terms cancel out.
        
        Args:
            x: Points at which to evaluate the derivative
            eps: Step size for finite difference (smaller = more accurate
                 but more susceptible to floating-point errors)
                 
        Returns:
            Numerical approximation of the derivative at each point
        """
        return (self.activation(x + eps) - self.activation(x - eps)) / (2 * eps)

    def _initialize_weights(self, n_features: int):
        """
        Initialize weights using Xavier/Glorot uniform distribution.
        
        Xavier initialization addresses the vanishing/exploding gradient problem
        by keeping the variance of activations consistent across layers.
        
        For uniform distribution: W ~ Uniform(-limit, limit)
        where limit = sqrt(6 / (fan_in + fan_out))
        
        This ensures:
            - Var(W) = 2 / (fan_in + fan_out)
            - Activations maintain reasonable variance through the network
        
        Biases are initialized to zero, which is standard practice.
        
        Args:
            n_features: Number of input features (determines W1 size)
        """
        np.random.seed(self.random_state)

        # First layer: input → hidden
        # fan_in = n_features, fan_out = n_hidden
        limit1 = np.sqrt(6.0 / (n_features + self.n_hidden))
        self.W1 = np.random.uniform(-limit1, limit1, (n_features, self.n_hidden))
        self.b1 = np.zeros((1, self.n_hidden))

        # Second layer: hidden → output
        # fan_in = n_hidden, fan_out = 1
        limit2 = np.sqrt(6.0 / (self.n_hidden + 1))
        self.W2 = np.random.uniform(-limit2, limit2, (self.n_hidden, 1))
        self.b2 = np.zeros((1, 1))

    def _forward(self, X: np.ndarray) -> dict:
        """
        Perform forward propagation through the network.
        
        Computes activations layer by layer:
            z1 = X @ W1 + b1     (linear combination)
            a1 = activation(z1)  (non-linear transformation)
            z2 = a1 @ W2 + b2    (linear combination)
            a2 = sigmoid(z2)     (output probability)
        
        All intermediate values are cached for use in backpropagation,
        which needs them to compute gradients efficiently.
        
        Args:
            X: Input data of shape (n_samples, n_features)
            
        Returns:
            Dictionary containing all intermediate values:
                - 'X': Original input
                - 'z1': Pre-activation values for hidden layer
                - 'a1': Hidden layer activations
                - 'z2': Pre-activation for output
                - 'a2': Output probabilities
        """
        # Hidden layer: linear transformation followed by activation
        z1 = X @ self.W1 + self.b1
        a1 = self.activation(z1)

        # Output layer: linear transformation followed by sigmoid
        # Sigmoid is always used for binary classification output
        z2 = a1 @ self.W2 + self.b2
        a2 = ActivationFunctions.sigmoid(z2)

        # Cache all values needed for backpropagation
        return {'X': X, 'z1': z1, 'a1': a1, 'z2': z2, 'a2': a2}

    def _compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute Binary Cross-Entropy (BCE) loss.
        
        BCE is the standard loss for binary classification:
            L = -1/n * Σ[y*log(p) + (1-y)*log(1-p)]
        
        This loss has nice properties:
            - Convex with respect to the logits (before sigmoid)
            - Heavily penalizes confident wrong predictions
            - Gradient has simple form: (predicted - actual)
        
        Args:
            y_true: True labels (0 or 1)
            y_pred: Predicted probabilities (from sigmoid output)
            
        Returns:
            Mean loss over all samples
        """
        # Handle case where y_pred might be 2D (from predict_proba)
        if y_pred.ndim == 2:
            y_pred = y_pred[:, 1] if y_pred.shape[1] == 2 else y_pred.flatten()

        # Clip predictions to prevent log(0) which gives -inf
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        
        # Binary cross-entropy formula
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss

    def _backward(self, y_true: np.ndarray, cache: dict) -> dict:
        """
        Perform backpropagation to compute gradients.
        
        Uses the chain rule to propagate error from output to input:
        
        For output layer (with BCE loss + sigmoid):
            dL/dz2 = a2 - y  (simplified gradient from loss derivative)
            dL/dW2 = a1.T @ dz2 / m
            dL/db2 = mean(dz2)
        
        For hidden layer:
            dL/da1 = dz2 @ W2.T  (error propagated back)
            dL/dz1 = dL/da1 * activation'(z1)  (chain rule)
            dL/dW1 = X.T @ dz1 / m
            dL/db1 = mean(dz1)
        
        The division by m (batch size) averages gradients over samples.
        
        Args:
            y_true: True labels of shape (batch_size,)
            cache: Dictionary from forward pass containing intermediate values
            
        Returns:
            Dictionary of gradients: {'dW1', 'db1', 'dW2', 'db2'}
        """
        m = y_true.shape[0]  # Batch size for averaging
        y_true = y_true.reshape(-1, 1)  # Ensure column vector

        # Retrieve cached values from forward pass
        X = cache['X']
        z1 = cache['z1']
        a1 = cache['a1']
        a2 = cache['a2']

        # Output layer gradients
        # For BCE loss with sigmoid: dL/dz2 = a2 - y (elegant simplification!)
        dz2 = a2 - y_true
        dW2 = (1/m) * (a1.T @ dz2)
        db2 = (1/m) * np.sum(dz2, axis=0, keepdims=True)

        # Hidden layer gradients (applying chain rule)
        # First, propagate error back through weights
        da1 = dz2 @ self.W2.T
        
        # Then, apply derivative of activation function
        if self.activation_derivative is not None:
            dz1 = da1 * self.activation_derivative(z1)
        else:
            # Fallback to numerical derivative for custom functions
            dz1 = da1 * self._numerical_derivative(z1)

        dW1 = (1/m) * (X.T @ dz1)
        db1 = (1/m) * np.sum(dz1, axis=0, keepdims=True)

        return {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}

    def _update_weights_sgd(self, gradients: dict):
        """
        Update weights using vanilla Stochastic Gradient Descent.
        
        SGD update rule: w = w - learning_rate * gradient
        
        This is the simplest optimization algorithm. More advanced
        optimizers like Adam add momentum and adaptive learning rates.
        
        Args:
            gradients: Dictionary with gradient arrays from backpropagation
        """
        self.W1 -= self.learning_rate * gradients['dW1']
        self.b1 -= self.learning_rate * gradients['db1']
        self.W2 -= self.learning_rate * gradients['dW2']
        self.b2 -= self.learning_rate * gradients['db2']

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_valid: np.ndarray = None,
        y_valid: np.ndarray = None
    ):
        """
        Train the neural network on the given data.
        
        Training loop structure:
            For each epoch:
                1. Shuffle data (different order each epoch)
                2. For each mini-batch:
                    a. Forward pass
                    b. Compute loss
                    c. Backward pass (compute gradients)
                    d. Update weights
                3. Optionally evaluate on validation set
                4. Log progress
        
        Mini-batch training balances between:
            - Batch gradient descent (stable but slow)
            - Stochastic gradient descent (noisy but fast)
        
        Args:
            X: Training features of shape (n_samples, n_features)
            y: Training labels of shape (n_samples,)
            X_valid: Optional validation features for monitoring
            y_valid: Optional validation labels
            
        Returns:
            self: Fitted model instance (allows method chaining)
        """
        n_samples, n_features = X.shape
        self._initialize_weights(n_features)
        n_batches = int(np.ceil(n_samples / self.batch_size))

        for epoch in range(self.n_epochs):
            # Shuffle data at the start of each epoch
            # This helps prevent learning order-dependent patterns
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            epoch_loss = 0

            # Process each mini-batch
            for i in range(n_batches):
                start_idx = i * self.batch_size
                end_idx = min((i + 1) * self.batch_size, n_samples)

                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]

                # Forward → Loss → Backward → Update (the training cycle)
                cache = self._forward(X_batch)
                batch_loss = self._compute_loss(y_batch, cache['a2'])
                epoch_loss += batch_loss * len(y_batch)

                gradients = self._backward(y_batch, cache)
                self._update_weights_sgd(gradients)

            # Average loss over all samples
            epoch_loss /= n_samples
            self.train_losses.append(epoch_loss)

            # Validate if validation set provided
            if X_valid is not None:
                valid_proba = self.predict_proba(X_valid)[:, 1]
                valid_loss = self._compute_loss(y_valid, valid_proba)
                self.valid_losses.append(valid_loss)

            # Log progress every 10 epochs
            if self.verbose and (epoch + 1) % 10 == 0:
                msg = f"Epoch {epoch+1}/{self.n_epochs} - Train Loss: {epoch_loss:.4f}"
                if X_valid is not None:
                    msg += f" - Valid Loss: {valid_loss:.4f}"
                print(msg)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for input samples.
        
        Returns probability of each class in scikit-learn compatible format:
            Column 0: P(class=0) = 1 - P(class=1)
            Column 1: P(class=1) = sigmoid output
        
        Args:
            X: Features of shape (n_samples, n_features)
            
        Returns:
            Array of shape (n_samples, 2) with class probabilities
        """
        cache = self._forward(X)
        prob_1 = cache['a2'].flatten()  # Probability of positive class
        prob_0 = 1 - prob_1             # Probability of negative class
        return np.column_stack([prob_0, prob_1])

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict class labels for input samples.
        
        Applies threshold to probability predictions. Default threshold
        of 0.5 optimizes for accuracy, but can be adjusted to optimize
        for other metrics (e.g., lower threshold for higher recall).
        
        Args:
            X: Features of shape (n_samples, n_features)
            threshold: Classification threshold (default 0.5)
            
        Returns:
            Array of predicted labels (0 or 1)
        """
        proba = self.predict_proba(X)[:, 1]
        return (proba >= threshold).astype(int)
