# region Imports
import numpy as np

# endregion


# region Activation
def relu(x: np.ndarray) -> np.ndarray:
    """Rectified Linear Unit activation."""
    return np.maximum(0, x)


def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax activation."""
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


# endregion


# region Dataset
def generate_spiral_data(
    samples_per_class: int = 100, num_classes: int = 3
) -> tuple[np.ndarray, np.ndarray]:
    """Generate 2D spiral classification dataset."""
    np.random.seed(42)
    total_samples = samples_per_class * num_classes
    X = np.zeros((total_samples, 2), dtype=np.float64)
    y = np.zeros(total_samples, dtype=np.int64)

    for class_idx in range(num_classes):
        indices = range(
            class_idx * samples_per_class, (class_idx + 1) * samples_per_class
        )
        radius = np.linspace(0.0, 1.0, samples_per_class)
        theta = (
            np.linspace(class_idx * 4, (class_idx + 1) * 4, samples_per_class)
            + np.random.randn(samples_per_class) * 0.2
        )
        X[indices] = np.column_stack((radius * np.sin(theta), radius * np.cos(theta)))
        y[indices] = class_idx

    return X, y


# endregion


# region Network
class TrainableFeedforwardNetwork:
    """Two-layer feedforward neural network with analytical backpropagation."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        learning_rate: float = 1.0,
    ) -> None:
        self.learning_rate = learning_rate
        self.W1 = np.random.randn(input_size, hidden_size) * 0.1
        self.b1 = np.zeros((1, hidden_size), dtype=np.float64)
        self.W2 = np.random.randn(hidden_size, output_size) * 0.1
        self.b2 = np.zeros((1, output_size), dtype=np.float64)

    def forward(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute forward pass and return pre-activations and probabilities."""
        z1 = np.dot(X, self.W1) + self.b1
        a1 = relu(z1)
        z2 = np.dot(a1, self.W2) + self.b2
        probs = softmax(z2)
        return z1, a1, probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for given inputs."""
        _, _, probs = self.forward(X)
        return np.argmax(probs, axis=1)

    def train_step(self, X: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        """Execute one forward-backward cycle and update weights via SGD."""
        num_samples = X.shape[0]

        # 1. Forward Pass
        z1, a1, probs = self.forward(X)

        # 2. Cross-Entropy Loss & Accuracy
        loss = float(-np.mean(np.log(probs[np.arange(num_samples), y] + 1e-15)))
        acc = float(np.mean(np.argmax(probs, axis=1) == y))

        # 3. Backward Pass (Analytical Chain Rule)
        dscores = probs.copy()
        dscores[np.arange(num_samples), y] -= 1.0
        dscores /= num_samples

        dW2 = np.dot(a1.T, dscores)
        db2 = np.sum(dscores, axis=0, keepdims=True)
        dz1 = np.dot(dscores, self.W2.T) * (z1 > 0)
        dW1 = np.dot(X.T, dz1)
        db1 = np.sum(dz1, axis=0, keepdims=True)

        # 4. Gradient Descent Parameter Update
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2

        return loss, acc


# endregion


# region Main
def main() -> None:
    X, y = generate_spiral_data(samples_per_class=100, num_classes=3)
    print(f"[TrainableFFN-Data] Spiral dataset created: X={X.shape}, y={y.shape}")

    model = TrainableFeedforwardNetwork(
        input_size=2,
        hidden_size=64,
        output_size=3,
        learning_rate=1.0,
    )
    epochs = 1000

    for epoch in range(epochs + 1):
        loss, acc = model.train_step(X, y)
        if epoch % 200 == 0:
            print(
                f"[TrainableFFN-Train] Epoch {epoch:4d}/{epochs} | "
                f"Loss: {loss:.4f} | Accuracy: {acc * 100:.2f}%"
            )


if __name__ == "__main__":
    main()

# endregion
