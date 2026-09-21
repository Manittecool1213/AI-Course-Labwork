"""Task 4 experiments for the XOR network: gradient check, symmetry and activations.

Part A/B: learning check and backpropagation check on the tanh network.
Part C:   zero (and identical) weight initialisation.
Part D:   sigmoid vs tanh vs ReLU hidden activation, plus a diagnostic that
          tells saturated sigmoid units apart from dead ReLU units.

Run as a script:

    python xor_experiments.py
"""

from collections.abc import Callable

import torch
from torch import nn

from xor_net import LEARNING_RATE, SEED, STEPS, THRESHOLD, X, Y, XORNet

ACTIVATIONS: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "sigmoid": torch.sigmoid,
    "tanh": torch.tanh,
    "relu": torch.relu,
}


class FlexNet(nn.Module):
    """The same 2 -> 2 -> 1 network as XORNet, with a selectable hidden activation."""

    def __init__(self, activation: str = "tanh") -> None:
        super().__init__()
        self.hidden = nn.Linear(2, 2)
        self.output = nn.Linear(2, 1)
        self.activation = ACTIVATIONS[activation]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the logits for a batch of inputs."""
        return self.output(self.activation(self.hidden(x)))


def train(
    model: nn.Module, steps: int = STEPS, record_at: tuple[int, ...] = ()
) -> tuple[list[float], dict[int, torch.Tensor]]:
    """Train full-batch with Adam, returning the loss history and W1 snapshots.

    Snapshots of the hidden weight matrix are taken after the listed step numbers.
    """
    loss_fn = nn.BCEWithLogitsLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    losses: list[float] = []
    snapshots: dict[int, torch.Tensor] = {}
    for step in range(1, steps + 1):
        loss = loss_fn(model(X), Y)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        losses.append(loss.item())
        if step in record_at:
            snapshots[step] = model.hidden.weight.detach().clone()  # type: ignore[union-attr]
    return losses, snapshots


def first_layer_grad(model: nn.Module) -> torch.Tensor:
    """Return dL/dW1 for the current parameters (one forward and backward pass)."""
    model.zero_grad()
    nn.BCEWithLogitsLoss()(model(X), Y).backward()
    return model.hidden.weight.grad.clone()  # type: ignore[union-attr]


def all_correct(model: nn.Module) -> bool:
    """Return True if every thresholded prediction matches its label."""
    with torch.no_grad():
        return bool(torch.equal((torch.sigmoid(model(X)) > THRESHOLD).float(), Y))


def part_a_and_b() -> None:
    """Basic learning check (A) and the backpropagation check (B) for tanh."""
    print("=== Part A: basic learning check (tanh, seed 0) ===")
    torch.manual_seed(SEED)
    model = XORNet()
    loss_fn = nn.BCEWithLogitsLoss()
    with torch.no_grad():
        print(f"Initial loss: {loss_fn(model(X), Y).item():.4f}")
    losses, _ = train(model)
    with torch.no_grad():
        probabilities = torch.sigmoid(model(X))
    print(f"Final loss: {losses[-1]:.6f} (after {STEPS} steps)")
    for x, p in zip(X, probabilities):
        print(f"  {tuple(x.tolist())} -> p = {p.item():.6f}, label = {int(p.item() > THRESHOLD)}")
    print(f"All four labels correct: {all_correct(model)}")

    print("\n=== Part B: backpropagation check (initial tanh network, float64) ===")
    torch.manual_seed(SEED)
    model = XORNet().double()
    x, y = X.double(), Y.double()
    loss_fn = nn.BCEWithLogitsLoss()

    loss_fn(model(x), y).backward()
    autograd = model.hidden.weight.grad.clone()
    print(f"parameter.grad for W1 (dL/dW1):\n{autograd}")

    # The mean loss is the average of the per-example losses, so by linearity of
    # the derivative its gradient is the average of the per-example gradients.
    per_example = []
    for i in range(len(x)):
        model.zero_grad()
        loss_fn(model(x[i : i + 1]), y[i : i + 1]).backward()
        per_example.append(model.hidden.weight.grad.clone())
    average = torch.stack(per_example).mean(dim=0)
    print(f"Matches the average of the four per-example gradients: {torch.allclose(autograd, average)}")

    # Central finite differences on each entry of W1 as an independent check.
    eps = 1e-6
    numeric = torch.zeros_like(autograd)
    with torch.no_grad():
        for i in range(2):
            for j in range(2):
                original = model.hidden.weight[i, j].item()
                model.hidden.weight[i, j] = original + eps
                up = loss_fn(model(x), y).item()
                model.hidden.weight[i, j] = original - eps
                down = loss_fn(model(x), y).item()
                model.hidden.weight[i, j] = original
                numeric[i, j] = (up - down) / (2 * eps)
    print(f"Finite-difference gradient:\n{numeric}")
    print(f"Max |autograd - finite difference|: {(autograd - numeric).abs().max().item():.2e}")


def part_c() -> None:
    """Zero and identical initialisation keep the two hidden units the same."""
    for name, value in [("zeros", 0.0), ("all 0.5", 0.5)]:
        print(f"\n=== Part C: all weights initialised to {name} (tanh) ===")
        model = XORNet()
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.fill_(value)
        print(f"Initial dL/dW1:\n{first_layer_grad(model)}")
        losses, snapshots = train(model, record_at=(1, 10, 100, STEPS))
        for step, weights in snapshots.items():
            identical = torch.equal(weights[0], weights[1])
            print(f"  step {step:>4}: W1 rows = {weights[0].tolist()} and {weights[1].tolist()}, identical: {identical}")
        with torch.no_grad():
            probabilities = torch.sigmoid(model(X)).flatten().tolist()
        print(f"Final loss: {losses[-1]:.4f}, probabilities: {[round(p, 4) for p in probabilities]}")
        print(f"All four labels correct: {all_correct(model)}")


def part_d() -> None:
    """Compare the three hidden activations on the same seed, then across seeds."""
    print("\n=== Part D: activation experiment (seed 0, early = gradient at step 1) ===")
    print(f"{'Activation':<10} {'Final loss':>12} {'4/4 correct':>12} {'Early ||dL/dW1||':>18}")
    for name in ACTIVATIONS:
        torch.manual_seed(SEED)
        model = FlexNet(name)
        early = first_layer_grad(model).norm().item()
        losses, _ = train(model)
        print(f"{name:<10} {losses[-1]:>12.6f} {str(all_correct(model)):>12} {early:>18.4f}")

    print("\nRepeatability over seeds 0-19:")
    for name in ACTIVATIONS:
        successes, early_norms, final_losses = 0, [], []
        for seed in range(20):
            torch.manual_seed(seed)
            model = FlexNet(name)
            early_norms.append(first_layer_grad(model).norm().item())
            losses, _ = train(model)
            final_losses.append(losses[-1])
            successes += all_correct(model)
        mean_norm = sum(early_norms) / len(early_norms)
        print(
            f"  {name:<8} 4/4 correct in {successes}/20 runs, mean early norm {mean_norm:.4f}, "
            f"worst final loss {max(final_losses):.4f}"
        )


def part_d_diagnostic() -> None:
    """Separate saturated sigmoid units from dead ReLU units using (pre-)activations."""
    print("\n=== Think About It: pre-activations and derivatives ===")
    # Sigmoid: at initialisation the largest derivative h(1 - h) per unit is close to its maximum of 0.25.
    for seed in range(3):
        torch.manual_seed(seed)
        model = FlexNet("sigmoid")
        with torch.no_grad():
            h = torch.sigmoid(model.hidden(X))
        print(f"  sigmoid seed {seed}: largest derivative per unit at init = {(h * (1 - h)).max(dim=0).values.tolist()}")

    # Sigmoid after training (seed 0 got stuck): units are saturated for every input.
    torch.manual_seed(SEED)
    model = FlexNet("sigmoid")
    train(model)
    with torch.no_grad():
        pre = model.hidden(X)
        h = torch.sigmoid(pre)
    print(f"  trained sigmoid (seed 0) pre-activations:\n{pre}\n  activations:\n{h}\n  derivatives h(1 - h):\n{h * (1 - h)}")

    # ReLU: a unit is dead if its pre-activation is <= 0 for every input, so its output and derivative are exactly 0.
    for seed in range(20):
        torch.manual_seed(seed)
        model = FlexNet("relu")
        with torch.no_grad():
            dead = ((model.hidden(X) > 0).sum(dim=0) == 0).nonzero().flatten().tolist()
        if dead:
            print(f"  relu seed {seed}: unit(s) {dead} dead at initialisation, early ||dL/dW1|| = {first_layer_grad(model).norm().item():.4f}")


def part_e() -> None:
    """Linear baselines: neither a single affine map nor stacked affine layers learn XOR."""
    print("\n=== Part E: linear baselines (Task 1 prediction) ===")
    loss_fn = nn.BCEWithLogitsLoss()
    for name, build in [
        ("single affine + sigmoid", lambda: nn.Linear(2, 1)),
        ("2-2-1 without hidden activation", lambda: nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 1))),
    ]:
        for seed in range(3):
            torch.manual_seed(seed)
            model = build()
            optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
            for _ in range(STEPS):
                loss = loss_fn(model(X), Y)
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
            with torch.no_grad():
                p = torch.sigmoid(model(X)).flatten()
            correct = int(((p > THRESHOLD).float() == Y.flatten()).sum().item())
            print(f"  {name}, seed {seed}: final loss {loss.item():.4f}, {correct}/4 correct, p = {[round(v, 3) for v in p.tolist()]}")


def main() -> None:
    part_a_and_b()
    part_c()
    part_d()
    part_d_diagnostic()
    part_e()


if __name__ == "__main__":
    main()
