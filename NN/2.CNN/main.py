# region Imports
import numpy as np

# endregion


# region Activation Functions
def relu(x: np.ndarray) -> np.ndarray:
    """Rectified Linear Unit (ReLU) activation function."""
    return np.maximum(0, x)


def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax activation function."""
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def conv2d(X: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    """2D spatial convolution forward pass (stride=1, valid padding)."""
    batch_size, _, h_in, w_in = X.shape
    out_channels, _, k_h, k_w = W.shape
    h_out = h_in - k_h + 1
    w_out = w_in - k_w + 1
    output = np.zeros((batch_size, out_channels, h_out, w_out))

    for c in range(out_channels):
        for i in range(h_out):
            for j in range(w_out):
                patch = X[:, :, i : i + k_h, j : j + k_w]
                output[:, c, i, j] = np.sum(patch * W[c], axis=(1, 2, 3)) + b[c]
    return output


def max_pool2d(X: np.ndarray, pool_size: int = 2, stride: int = 2) -> np.ndarray:
    """2D spatial max pooling forward pass."""
    batch_size, channels, h, w = X.shape
    h_out = (h - pool_size) // stride + 1
    w_out = (w - pool_size) // stride + 1
    output = np.zeros((batch_size, channels, h_out, w_out))

    for i in range(h_out):
        for j in range(w_out):
            h_start = i * stride
            w_start = j * stride
            h_end = h_start + pool_size
            w_end = w_start + pool_size
            patch = X[:, :, h_start:h_end, w_start:w_end]
            output[:, :, i, j] = np.max(patch, axis=(2, 3))
    return output


# endregion


# region Network Definition
class SimpleConvolutionalNetwork:
    """Minimal Convolutional Neural Network (Conv2D -> ReLU -> MaxPool -> Dense)."""

    def __init__(
        self,
        in_channels: int = 1,
        num_filters: int = 4,
        filter_size: int = 3,
        pool_size: int = 2,
        img_h: int = 8,
        img_w: int = 8,
        output_size: int = 3,
    ) -> None:
        self.pool_size = pool_size

        # Spatial dimensions after valid convolution
        conv_h = img_h - filter_size + 1
        conv_w = img_w - filter_size + 1

        # Spatial dimensions after max pooling
        pool_h = conv_h // pool_size
        pool_w = conv_w // pool_size
        flatten_dim = num_filters * pool_h * pool_w

        # Convolution parameters
        self.W_conv = (
            np.random.randn(num_filters, in_channels, filter_size, filter_size) * 0.01
        )
        self.b_conv = np.zeros(num_filters)

        # Fully connected projection parameters
        self.W_fc = np.random.randn(flatten_dim, output_size) * 0.01
        self.b_fc = np.zeros((1, output_size))

        print(
            f"[CNN-Init] Initialized W_conv: {self.W_conv.shape}, "
            f"b_conv: {self.b_conv.shape}"
        )
        print(
            f"[CNN-Init] Initialized W_fc: {self.W_fc.shape}, b_fc: {self.b_fc.shape}"
        )

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Compute forward pass through Conv, ReLU, Pool, and Linear layers.

        Args:
            X: Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            Output probability distribution of shape (batch_size, output_size).
        """
        conv_out = conv2d(X, self.W_conv, self.b_conv)
        relu_out = relu(conv_out)
        pool_out = max_pool2d(relu_out, self.pool_size, self.pool_size)
        flattened = pool_out.reshape(X.shape[0], -1)
        logits = np.dot(flattened, self.W_fc) + self.b_fc
        return softmax(logits)


# endregion


# region Main Routine
def main() -> None:
    batch_size = 5
    channels = 1
    height = 8
    width = 8
    output_dim = 3

    cnn = SimpleConvolutionalNetwork(
        in_channels=channels,
        num_filters=4,
        filter_size=3,
        pool_size=2,
        img_h=height,
        img_w=width,
        output_size=output_dim,
    )

    dummy_input = np.random.randn(batch_size, channels, height, width)
    print(f"[CNN-Main] Dummy image input shape: {dummy_input.shape}")

    predictions = cnn.forward(dummy_input)
    print(f"[CNN-Main] Predictions shape: {predictions.shape}")
    print(f"[CNN-Main] Predictions:\n{predictions}")

    prob_sums = np.sum(predictions, axis=1)
    print(f"[CNN-Main] Sum of probabilities per sample:\n{prob_sums}")

    predicted_classes = np.argmax(predictions, axis=1)
    print(f"[CNN-Main] Predicted classes: {predicted_classes}")


if __name__ == "__main__":
    main()
# endregion
