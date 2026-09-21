"""XOR with a 2-2-1 PyTorch network (tanh hidden layer, logit output).

The network is trained full-batch on the four XOR examples with
BCEWithLogitsLoss and Adam. After training it reports the final loss, the four
probabilities and labels, and the first-layer weight gradient, and saves a plot
of the loss curve.

Run as a script:

    python xor_net.py
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")  # Save the plot to a file instead of opening a window.

import torch
from torch import nn

SEED = 0
STEPS = 3000
LEARNING_RATE = 0.05
THRESHOLD = 0.5
PLOT_PATH = Path(__file__).with_name("loss_curve.png")

# The four XOR examples: each row of X is (x1, x2), and Y holds the label.
X = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
Y = torch.tensor([[0.0], [1.0], [1.0], [0.0]])


class XORNet(nn.Module):
    """A 2 -> 2 -> 1 network that outputs one logit per example."""

    def __init__(self) -> None:
        super().__init__()
        self.hidden = nn.Linear(2, 2)  # W1 (2x2) and b1 (2)
        self.output = nn.Linear(2, 1)  # W2 (1x2) and b2 (1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the logits for a batch of inputs (the sigmoid is applied in the loss)."""
        return self.output(torch.tanh(self.hidden(x)))


def main() -> None:
    torch.manual_seed(SEED)  # Reproducible random weight initialisation.
    model = XORNet()
    loss_fn = nn.BCEWithLogitsLoss()  # Sigmoid + binary cross-entropy in one stable op.
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    with torch.no_grad():
        initial_loss = loss_fn(model(X), Y).item()
    print(f"Initial loss: {initial_loss:.4f}")

    losses: list[float] = []
    for _ in range(STEPS):
        logits = model(X)  # Forward pass on all four examples (full batch).
        loss = loss_fn(logits, Y)  # Scalar mean BCE loss.
        optimiser.zero_grad()  # Clear gradients left over from the previous step.
        loss.backward()  # Reverse-mode AD: fills each parameter's .grad with dL/dparam.
        optimiser.step()  # Adam updates the parameters using those gradients.
        losses.append(loss.item())

    # One more forward and backward pass on the trained model, so that the
    # gradient we print belongs to the final parameters.
    optimiser.zero_grad()
    logits = model(X)
    final_loss = loss_fn(logits, Y)
    final_loss.backward()
    probabilities = torch.sigmoid(logits).detach()
    labels = (probabilities > THRESHOLD).float()

    print(f"Final loss: {final_loss.item():.6f}")
    print("Probabilities and predicted labels:")
    for x, p, label, target in zip(X, probabilities, labels, Y):
        print(f"  {tuple(x.tolist())} -> p = {p.item():.4f}, label = {int(label.item())}, target = {int(target.item())}")
    print(f"dL/dW1 (gradient of the first-layer weights):\n{model.hidden.weight.grad}")

    # Check 1: the loss should end well below the ln 2 (about 0.693) plateau of a model that outputs 0.5 everywhere.
    print(f"Loss below ln 2: {final_loss.item() < 0.693}")
    # Check 2: every thresholded label should equal its target.
    print(f"All four labels correct: {bool(torch.equal(labels, Y))}")

    plt.figure(figsize=(6, 4))
    plt.plot(range(1, STEPS + 1), losses)
    plt.xlabel("Training step")
    plt.ylabel("BCE loss")
    plt.title("XOR training loss")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    print(f"Loss curve saved to {PLOT_PATH.name}")


if __name__ == "__main__":
    main()
