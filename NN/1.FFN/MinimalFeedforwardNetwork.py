# region Imports
import numpy as np

# endregion


# region Activation
def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-x))


def relu(x: np.ndarray) -> np.ndarray:
    """Rectified Linear Unit (ReLU) activation function."""
    return np.maximum(0, x)


def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax activation function."""
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


# endregion


# region Network
class MinimalFeedforwardNetwork:
    """Two-layer feedforward neural network for classification."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))

        print(f"[FFN-Init] Initialized W1: {self.W1.shape}, b1: {self.b1.shape}")
        print(f"[FFN-Init] Initialized W2: {self.W2.shape}, b2: {self.b2.shape}")

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Compute forward pass through hidden and output layers."""
        z1 = np.dot(X, self.W1) + self.b1
        a1 = relu(z1)
        z2 = np.dot(a1, self.W2) + self.b2
        return softmax(z2)


# endregion


# region Main
def main() -> None:
    input_dim = 10
    hidden_dim = 20
    output_dim = 3

    nn = MinimalFeedforwardNetwork(input_dim, hidden_dim, output_dim)

    dummy_input = np.random.randn(5, input_dim)
    print(f"[FFN-Main] Dummy input shape: {dummy_input.shape}")

    predictions = nn.forward(dummy_input)
    print(f"[FFN-Main] Predictions shape: {predictions.shape}")
    print(f"[FFN-Main] Predictions:\n{predictions}")

    prob_sums = np.sum(predictions, axis=1)
    print(f"[FFN-Main] Sum of probabilities per sample:\n{prob_sums}")

    predicted_classes = np.argmax(predictions, axis=1)
    print(f"[FFN-Main] Predicted classes: {predicted_classes}")


if __name__ == "__main__":
    main()

# endregion
