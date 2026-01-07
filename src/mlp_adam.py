"""
MLP with Adam Optimizer and Early Stopping

This module extends the base MLP implementation with the Adam optimizer,
one of the most popular and effective optimization algorithms for neural networks.
It also includes early stopping to prevent overfitting.

Adam combines the best of two worlds:
    - Momentum: Accelerates learning by accumulating gradients from past steps
    - RMSprop: Adapts learning rates per-parameter using second moment estimates

The algorithm was introduced by Kingma and Ba (2015) in:
"Adam: A Method for Stochastic Optimization"

Key advantages over vanilla SGD:
    - Less sensitive to learning rate choice
    - Handles sparse gradients well
    - Works well with noisy gradients and mini-batches
    - Bias correction for initial steps

Author: Ilya Sevastyanov
"""

import numpy as np
from .mlp import MLP


class MLPWithAdam(MLP):
    """
    Multi-Layer Perceptron with Adam optimizer and early stopping.
    
    Inherits the core MLP architecture but replaces SGD with Adam optimization.
    Adam maintains two moving averages for each parameter:
        - First moment (mean of gradients) - provides momentum
        - Second moment (mean of squared gradients) - provides per-parameter learning rates
    
    Early stopping monitors validation loss and stops training when the model
    stops improving, preventing overfitting.
    
    Additional Attributes:
        beta1: Exponential decay rate for first moment (default 0.9)
        beta2: Exponential decay rate for second moment (default 0.999)
        epsilon: Small constant for numerical stability (default 1e-8)
        patience: Number of epochs without improvement before stopping
        min_delta: Minimum change to qualify as an improvement
        m: First moment estimates (mean of gradients)
        v: Second moment estimates (mean of squared gradients)
        t: Time step counter for bias correction
    """

    def __init__(
        self,
        n_hidden: int = 100,
        activation: str = 'sigmoid',
        learning_rate: float = 0.001,
        n_epochs: int = 100,
        batch_size: int = 32,
        random_state: int = 42,
        verbose: bool = True,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        patience: int = 15,
        min_delta: float = 1e-4
    ):
        """
        Initialize MLP with Adam optimizer.
        
        Note: Default learning rate is 0.001 (vs 0.01 for SGD) because Adam
        already includes momentum and adaptive rates.
        
        Args:
            n_hidden: Number of hidden neurons
            activation: Activation function for hidden layer
            learning_rate: Base learning rate (α in Adam formula)
            n_epochs: Maximum number of epochs
            batch_size: Mini-batch size
            random_state: Random seed for reproducibility
            verbose: Print training progress
            beta1: First moment decay rate. Higher = more smoothing.
                   0.9 means gradients are averaged over ~10 steps.
            beta2: Second moment decay rate. Higher = longer memory.
                   0.999 means squared gradients averaged over ~1000 steps.
            epsilon: Numerical stability term. Prevents division by zero.
            patience: Early stopping patience. Training stops after this
                     many epochs without improvement.
            min_delta: Minimum improvement threshold. Changes smaller than
                      this are not considered improvements.
        """
        super().__init__(
            n_hidden, activation, learning_rate, n_epochs,
            batch_size, random_state, verbose
        )

        # Adam hyperparameters (default values from original paper)
        self.beta1 = beta1   # First moment decay (momentum term)
        self.beta2 = beta2   # Second moment decay (adaptive learning rate)
        self.epsilon = epsilon  # Prevents division by zero

        # Early stopping configuration
        self.patience = patience
        self.min_delta = min_delta

        # Adam state (initialized during training)
        self.m = {}  # First moment estimates
        self.v = {}  # Second moment estimates
        self.t = 0   # Time step (for bias correction)

        # Best model tracking for early stopping
        self.best_weights = None
        self.best_valid_loss = float('inf')

    def _initialize_adam_state(self):
        """
        Initialize Adam moving averages to zeros.
        
        The first and second moment estimates are initialized to zero vectors
        of the same shape as each parameter. This causes bias toward zero in
        early iterations, which is corrected by the bias correction terms.
        """
        self.m = {
            'W1': np.zeros_like(self.W1),
            'b1': np.zeros_like(self.b1),
            'W2': np.zeros_like(self.W2),
            'b2': np.zeros_like(self.b2)
        }
        self.v = {
            'W1': np.zeros_like(self.W1),
            'b1': np.zeros_like(self.b1),
            'W2': np.zeros_like(self.W2),
            'b2': np.zeros_like(self.b2)
        }
        self.t = 0

    def _save_weights(self) -> dict:
        """
        Save current weights for potential restoration.
        
        Used by early stopping to remember the best model state.
        Deep copies are used to prevent issues with mutable arrays.
        
        Returns:
            Dictionary with copies of all weight arrays
        """
        return {
            'W1': self.W1.copy(),
            'b1': self.b1.copy(),
            'W2': self.W2.copy(),
            'b2': self.b2.copy()
        }

    def _restore_weights(self, weights: dict):
        """
        Restore weights from a saved state.
        
        Args:
            weights: Dictionary from _save_weights()
        """
        self.W1 = weights['W1'].copy()
        self.b1 = weights['b1'].copy()
        self.W2 = weights['W2'].copy()
        self.b2 = weights['b2'].copy()

    def _update_weights_adam(self, gradients: dict):
        """
        Update weights using the Adam optimizer.
        
        Adam algorithm for each parameter θ:
        
        1. Compute first moment (momentum):
           m_t = β₁ * m_{t-1} + (1 - β₁) * g_t
           This is an exponential moving average of gradients.
           
        2. Compute second moment (adaptive learning rate):
           v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²
           This is an exponential moving average of squared gradients.
           
        3. Bias correction (for early iterations):
           m̂_t = m_t / (1 - β₁^t)
           v̂_t = v_t / (1 - β₂^t)
           
           Since m and v are initialized to zero, they're biased toward
           zero in early steps. This correction compensates for that.
           
        4. Update parameters:
           θ_t = θ_{t-1} - α * m̂_t / (√v̂_t + ε)
           
           The learning rate is effectively scaled by 1/√v̂, which means:
           - Parameters with consistently large gradients get smaller updates
           - Parameters with small/sparse gradients get larger updates
        
        Args:
            gradients: Dictionary with gradient arrays from backpropagation
        """
        self.t += 1  # Increment time step

        for param_name, grad_name in [('W1', 'dW1'), ('b1', 'db1'),
                                       ('W2', 'dW2'), ('b2', 'db2')]:
            g = gradients[grad_name]

            # Update biased first moment estimate (momentum)
            self.m[param_name] = self.beta1 * self.m[param_name] + (1 - self.beta1) * g

            # Update biased second moment estimate (RMSprop-like)
            self.v[param_name] = self.beta2 * self.v[param_name] + (1 - self.beta2) * (g ** 2)

            # Compute bias-corrected estimates
            # This is important for early iterations when m and v are close to 0
            m_corrected = self.m[param_name] / (1 - self.beta1 ** self.t)
            v_corrected = self.v[param_name] / (1 - self.beta2 ** self.t)

            # Update parameters
            # The epsilon term prevents division by zero when v is very small
            param = getattr(self, param_name)
            param -= self.learning_rate * m_corrected / (np.sqrt(v_corrected) + self.epsilon)
            setattr(self, param_name, param)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_valid: np.ndarray = None,
        y_valid: np.ndarray = None
    ):
        """
        Train the neural network with Adam optimizer and early stopping.
        
        Extends parent's fit method with:
            - Adam optimization instead of vanilla SGD
            - Early stopping based on validation loss
            - Automatic restoration of best weights
        
        Early stopping logic:
            - Track best validation loss seen so far
            - If validation loss doesn't improve by min_delta for 'patience' epochs,
              stop training
            - Restore weights from the best epoch
        
        Args:
            X: Training features of shape (n_samples, n_features)
            y: Training labels of shape (n_samples,)
            X_valid: Validation features (required for early stopping)
            y_valid: Validation labels
            
        Returns:
            self: Fitted model instance
        """
        n_samples, n_features = X.shape

        # Initialize weights and Adam state
        self._initialize_weights(n_features)
        self._initialize_adam_state()

        n_batches = int(np.ceil(n_samples / self.batch_size))
        epochs_without_improvement = 0

        for epoch in range(self.n_epochs):
            # Shuffle training data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            epoch_loss = 0

            # Process mini-batches
            for i in range(n_batches):
                start_idx = i * self.batch_size
                end_idx = min((i + 1) * self.batch_size, n_samples)

                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]

                cache = self._forward(X_batch)
                batch_loss = self._compute_loss(y_batch, cache['a2'])
                epoch_loss += batch_loss * len(y_batch)

                gradients = self._backward(y_batch, cache)
                self._update_weights_adam(gradients)  # Adam instead of SGD

            epoch_loss /= n_samples
            self.train_losses.append(epoch_loss)

            # Validation and Early Stopping
            if X_valid is not None:
                valid_pred = self.predict_proba(X_valid)[:, 1]
                valid_loss = self._compute_loss(y_valid, valid_pred)
                self.valid_losses.append(valid_loss)

                # Check if this is the best model so far
                if valid_loss < self.best_valid_loss - self.min_delta:
                    self.best_valid_loss = valid_loss
                    self.best_weights = self._save_weights()
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                # Early stopping check
                if epochs_without_improvement >= self.patience:
                    if self.verbose:
                        print(f"Early stopping at epoch {epoch+1}")
                    break

            # Log progress
            if self.verbose and (epoch + 1) % 10 == 0:
                msg = f"Epoch {epoch+1}/{self.n_epochs} - Train Loss: {epoch_loss:.4f}"
                if X_valid is not None:
                    msg += f" - Valid Loss: {valid_loss:.4f}"
                print(msg)

        # Restore best weights if early stopping was triggered
        if self.best_weights is not None:
            self._restore_weights(self.best_weights)
            if self.verbose:
                print(f"\nRestored best weights (Valid Loss = {self.best_valid_loss:.4f})")

        return self
