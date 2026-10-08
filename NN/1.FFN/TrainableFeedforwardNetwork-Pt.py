# region Imports
import torch
import torch.nn as nn
import torch.optim as optim

# endregion


# region Dataset
def generate_spiral_data(
    samples_per_class: int = 100, num_classes: int = 3
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate 2D spiral classification dataset as PyTorch Tensors."""
    torch.manual_seed(42)
    total_samples = samples_per_class * num_classes
    X = torch.zeros((total_samples, 2), dtype=torch.float32)
    y = torch.zeros(total_samples, dtype=torch.int64)

    for class_idx in range(num_classes):
        indices = slice(class_idx * samples_per_class, (class_idx + 1) * samples_per_class)
        radius = torch.linspace(0.0, 1.0, samples_per_class)
        theta = (
            torch.linspace(class_idx * 4.0, (class_idx + 1) * 4.0, samples_per_class)
            + torch.randn(samples_per_class) * 0.2
        )
        X[indices] = torch.stack([radius * torch.sin(theta), radius * torch.cos(theta)], dim=1)
        y[indices] = class_idx

    return X, y


# endregion


# region Network
class PyTorchFeedforwardNetwork(nn.Module):
    """Two-layer feedforward neural network in PyTorch."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

        # Match TrainableFeedforwardNetwork initialization scale
        with torch.no_grad():
            self.fc1.weight.copy_(torch.randn_like(self.fc1.weight) * 0.1)
            self.fc1.bias.zero_()
            self.fc2.weight.copy_(torch.randn_like(self.fc2.weight) * 0.1)
            self.fc2.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute logits forward pass."""
        z1 = self.fc1(x)
        a1 = self.relu(z1)
        z2 = self.fc2(a1)
        return z2

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Predict class labels for given inputs."""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.argmax(logits, dim=1)


# endregion


# region Main
def main() -> None:
    X, y = generate_spiral_data(samples_per_class=100, num_classes=3)
    print(f"[PyTorchFFN-Data] Spiral dataset created: X={list(X.shape)}, y={list(y.shape)}")

    model = PyTorchFeedforwardNetwork(
        input_size=2,
        hidden_size=16,
        output_size=3,
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=1.0)
    epochs = 5000

    for epoch in range(epochs + 1):
        # 1. Forward Pass
        logits = model(X)

        # 2. Loss & Accuracy
        loss = criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean().item()

        # 3. Backward Pass (Autograd) & Optimizer Step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 200 == 0:
            print(
                f"[PyTorchFFN-Train] Epoch {epoch:4d}/{epochs} | "
                f"Loss: {loss.item():.4f} | Accuracy: {acc * 100:.2f}%"
            )


if __name__ == "__main__":
    main()

# endregion
