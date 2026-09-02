# region Imports
import numpy as np

# endregion


# region Activation Functions
def tanh(x: np.ndarray) -> np.ndarray:
    """Hyperbolic tangent activation function."""
    return np.tanh(x)


def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax activation function."""
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


# endregion


# region Network Definition
class SimpleRecurrentNetwork:
    """Single-layer Elman Recurrent Neural Network (RNN) for sequence classification."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        self.hidden_size = hidden_size

        # Weight matrices and biases
        self.Wxh = np.random.randn(input_size, hidden_size) * 0.01
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.bh = np.zeros((1, hidden_size))

        self.Why = np.random.randn(hidden_size, output_size) * 0.01
        self.by = np.zeros((1, output_size))

        print(f"[RNN-Init] Initialized Wxh: {self.Wxh.shape}, Whh: {self.Whh.shape}")
        print(f"[RNN-Init] Initialized bh: {self.bh.shape}, by: {self.by.shape}")
        print(f"[RNN-Init] Initialized Why: {self.Why.shape}")

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Compute forward pass across sequence timesteps.

        Args:
            X: Input tensor of shape (batch_size, seq_len, input_size).

        Returns:
            Output probability distribution of shape (batch_size, output_size).
        """
        batch_size, seq_len, _ = X.shape
        h = np.zeros((batch_size, self.hidden_size))

        for t in range(seq_len):
            x_t = X[:, t, :]
            h = tanh(np.dot(x_t, self.Wxh) + np.dot(h, self.Whh) + self.bh)

        output = np.dot(h, self.Why) + self.by
        return softmax(output)


# endregion


# region Main Routine
def main() -> None:
    batch_size = 5
    seq_len = 8
    input_dim = 10
    hidden_dim = 20
    output_dim = 3

    rnn = SimpleRecurrentNetwork(input_dim, hidden_dim, output_dim)

    dummy_input = np.random.randn(batch_size, seq_len, input_dim)
    print(f"[RNN-Main] Dummy sequence input shape: {dummy_input.shape}")

    predictions = rnn.forward(dummy_input)
    print(f"[RNN-Main] Predictions shape: {predictions.shape}")
    print(f"[RNN-Main] Predictions:\n{predictions}")

    prob_sums = np.sum(predictions, axis=1)
    print(f"[RNN-Main] Sum of probabilities per sample:\n{prob_sums}")

    predicted_classes = np.argmax(predictions, axis=1)
    print(f"[RNN-Main] Predicted classes: {predicted_classes}")


if __name__ == "__main__":
    main()
# endregion
