# region Imports
import torch
import torch.nn as nn

# endregion


# region Network
class PtMinimalFeedforwardNetwork(nn.Module):
    """Two-layer feedforward neural network for classification in PyTorch."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=-1)

        print(
            f"[PtFFN-Init] Initialized W1: {list(self.fc1.weight.shape)}, "
            f"b1: {list(self.fc1.bias.shape)}"
        )
        print(
            f"[PtFFN-Init] Initialized W2: {list(self.fc2.weight.shape)}, "
            f"b2: {list(self.fc2.bias.shape)}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute forward pass through hidden and output layers."""
        z1 = self.fc1(x)
        a1 = self.relu(z1)
        z2 = self.fc2(a1)
        return self.softmax(z2)


# endregion


# region Main
def main() -> None:
    input_dim = 10
    hidden_dim = 20
    output_dim = 3

    model = PtMinimalFeedforwardNetwork(input_dim, hidden_dim, output_dim)

    dummy_input = torch.randn(5, input_dim)
    print(f"[PtFFN-Main] Dummy input shape: {list(dummy_input.shape)}")

    predictions = model(dummy_input)
    print(f"[PtFFN-Main] Predictions shape: {list(predictions.shape)}")
    print(f"[PtFFN-Main] Predictions:\n{predictions}")

    prob_sums = torch.sum(predictions, dim=1)
    print(f"[PtFFN-Main] Sum of probabilities per sample:\n{prob_sums}")

    predicted_classes = torch.argmax(predictions, dim=1)
    print(f"[PtFFN-Main] Predicted classes: {predicted_classes.tolist()}")


if __name__ == "__main__":
    main()

# endregion
