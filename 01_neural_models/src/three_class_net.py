"""Three-class extension of the XOR sensor problem (2 -> 2 -> 3 network, softmax).

Class 0 is (0, 0), class 1 is (0, 1) or (1, 0), and class 2 is (1, 1). Only the
output layer and the loss differ from xor_net.py: three logits are trained with
multiclass cross-entropy.

Run as a script:

    python three_class_net.py
"""

import torch
from torch import nn

SEED = 0
STEPS = 3000
LEARNING_RATE = 0.05

X = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
Y = torch.tensor([0, 1, 1, 2])  # Class indices, as CrossEntropyLoss expects.


class ThreeClassNet(nn.Module):
    """A 2 -> 2 -> 3 network that outputs three logits per example."""

    def __init__(self) -> None:
        super().__init__()
        self.hidden = nn.Linear(2, 2)  # Unchanged from the binary network.
        self.output = nn.Linear(2, 3)  # W2 is now 3x2 and b2 has 3 entries.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the logits for a batch of inputs (softmax is applied in the loss)."""
        return self.output(torch.tanh(self.hidden(x)))


def main() -> None:
    torch.manual_seed(SEED)
    model = ThreeClassNet()
    loss_fn = nn.CrossEntropyLoss()  # Softmax + multiclass cross-entropy in one stable op.
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"Final weight matrix shape: {tuple(model.output.weight.shape)}")
    with torch.no_grad():
        print(f"Logits per example: {model(X).shape[1]}")
        print(f"Initial loss: {loss_fn(model(X), Y).item():.4f}")

    for _ in range(STEPS):
        loss = loss_fn(model(X), Y)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    with torch.no_grad():
        logits = model(X)
        probabilities = torch.softmax(logits, dim=1)
        print(f"Final loss: {loss_fn(logits, Y).item():.6f}")
        print("Predicted class probabilities:")
        for x, p, target in zip(X, probabilities, Y):
            print(f"  {tuple(x.tolist())} -> {[round(v, 4) for v in p.tolist()]}, predicted {int(p.argmax())}, target {int(target)}")
        print(f"All four classes correct: {bool(torch.equal(probabilities.argmax(dim=1), Y))}")

        # Check: the components of one softmax vector should sum to about 1.
        example = probabilities[1]
        print(f"\nExample (0, 1): p = {example.tolist()}, sum = {example.sum().item():.8f}")

        # Optional diagnostic: softmax is unchanged if the same constant is added to all logits.
        shifted = torch.softmax(logits[1] + 100.0, dim=0)
        print(f"After adding 100 to all logits: p = {shifted.tolist()}")
        print(f"Max difference from the original: {(shifted - example).abs().max().item():.2e}")

        # The naive softmax exp(z) / sum(exp(z)) overflows for large logits, unlike the max-subtracted one.
        big = logits[1] + 100.0
        naive = torch.exp(big) / torch.exp(big).sum()
        stable = torch.exp(big - big.max()) / torch.exp(big - big.max()).sum()
        print(f"Naive softmax at +100: {naive.tolist()}, max-subtracted: {stable.tolist()}")
        huge = logits[1] + 1000.0
        print(f"Naive softmax at +1000: {(torch.exp(huge) / torch.exp(huge).sum()).tolist()}")

    # Check: the gradient of the loss with respect to the logits should be (p - y) / N.
    logits = model(X).detach().requires_grad_()
    loss_fn(logits, Y).backward()
    expected = (torch.softmax(logits.detach(), dim=1) - nn.functional.one_hot(Y, 3).float()) / len(Y)
    print(f"\nLogit gradient equals (p - y) / N: {torch.allclose(logits.grad, expected)}")


if __name__ == "__main__":
    main()
