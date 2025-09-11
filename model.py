import numpy as np


class Monomial1HiddenNN:
    """
    One-hidden-layer NN with monomial activation.
    No biases. Supports user-specified weights; otherwise standard distribution init.

    Shapes:
      W: (hidden_dim, input_dim)
      U: (output_dim, hidden_dim)
      X : (input_dim, n)  -> Y_hat:  (output_dim, n)
    """

    def __init__(self, input_dim, hidden_dim, output_dim=1, degree=1, W=None, U=None, seed=42):
        if degree < 1:
            raise ValueError("degree must be >= 1")
        self.input_dim, self.hidden_dim, self.output_dim = int(input_dim), int(hidden_dim), int(output_dim)
        self.degree = int(degree)
        rng = np.random.default_rng(seed)

        # Initialize weights as specified or randomly
        if W is None:
            self.W = rng.standard_normal((self.hidden_dim, self.input_dim))
        else:
            self.W = np.asarray(W, dtype=float)
            if self.W.shape != (self.hidden_dim, self.input_dim):
                raise ValueError(f"W must have shape {(self.hidden_dim, self.input_dim)}")

        if U is None:
            self.U = rng.standard_normal((self.output_dim, self.hidden_dim))
        else:
            self.U = np.asarray(U, dtype=float)
            if self.U.shape != (self.output_dim, self.hidden_dim):
                raise ValueError(f"U must have shape {(self.output_dim, self.hidden_dim)}")

    def forward(self, X):
        """Compute Y_hat for inputs X of shape (input_dim, n)."""
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[0] != self.input_dim:
            raise ValueError(f"X must have shape ({self.input_dim}, n)")
        Z = (self.W @ X) ** self.degree  # monomial activation
        Y_hat = self.U @ Z
        return Z, Y_hat

    def loss(self, X, Y):
        """
        Square loss: 0.5 * mean((Y_hat - Y)^2)
        Y may be (1, n) when output_dim == 1, else (output_dim, n).
        """
        _, Y_hat = self.forward(X)
        Y = np.asarray(Y, dtype=float)
        if Y.ndim == 1 and self.output_dim == 1:
            Y = Y.reshape(1, -1)
        if Y.shape != Y_hat.shape:
            raise ValueError(f"Shapes must match: Y {Y.shape} vs Y_hat {Y_hat.shape}")
        se = (Y_hat - Y) ** 2
        return 0.5 * float(np.mean(se))

    def set_weights(self, W=None, U=None):
        # Optionally replace weights after construction (shape-checked).
        if W is not None:
            W = np.asarray(W, dtype=float)
            if W.shape != self.W.shape:
                raise ValueError(f"W must have shape {self.W.shape}")
            self.W = W

        if U is not None:
            U = np.asarray(U, dtype=float)
            if U.shape != self.U.shape:
                raise ValueError(f"U must have shape {self.U.shape}")
            self.U = U
