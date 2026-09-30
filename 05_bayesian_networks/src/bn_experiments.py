"""Experiments for the Bayesian network lab: CPTs, tests, prediction, generation and comparison.

Run as a script (also writes the generated sentences to the outputs folder):

    python bn_experiments.py
"""

from __future__ import annotations

import random
from pathlib import Path

from language_model import (
    DATASET,
    END,
    START,
    FirstOrderModel,
    MarkovLanguageModel,
    SecondOrderModel,
    normalisation_totals,
)

SEED = 0
NUM_SENTENCES = 20
OUTPUT_DIR = Path(__file__).with_name("outputs")


def show_distribution(model: MarkovLanguageModel, context: tuple[str, ...]) -> str:
    """Format P(next | context) sorted by probability, and its argmax."""
    distribution = model.distribution(context)
    ordered = sorted(distribution.items(), key=lambda item: (-item[1], item[0]))
    text = ", ".join(f"{token}: {p:.3f}" for token, p in ordered)
    return f"P(next | {' '.join(context)}) = {{{text}}}   argmax = {model.most_probable(context)}"


def generate_sentences(model: MarkovLanguageModel, mode: str, count: int, seed: int) -> list[str]:
    """Generate sentences with a fixed seed so the results can be reproduced."""
    rng = random.Random(seed)
    return [model.generate(mode, rng) for _ in range(count)]


def first_order_part(model: FirstOrderModel) -> None:
    print("=== Part IV: first-order CPT P(next | current) ===")
    for word in ["the", "cat", "dog", "sat", "ran"]:
        print(show_distribution(model, (word,)))
    print("\nZero-probability transitions for these words (next token never observed):")
    for word in ["the", "cat", "dog", "sat", "ran"]:
        seen = set(model.distribution((word,)))
        unseen = sorted(model.vocabulary - seen - {START})
        print(f"  after '{word}': {unseen}")

    print("\n=== Part VII: normalisation, sum_v P(v | w) for every context ===")
    for context, total in normalisation_totals(model).items():
        print(f"  {context[0]:<8} {total:.6f}")

    print("\n=== Part VIII: next-word prediction ===")
    for word in [START, "the", "cat", "dog", "sat", "on", "ran", "to", "mat"]:
        print(show_distribution(model, (word,)))

    print("\n=== Question 7: unseen word ===")
    try:
        model.distribution(("bird",))
    except KeyError as error:
        print(f"  distribution(('bird',)) raised {type(error).__name__}: {error}")


def generation_part(name: str, model: MarkovLanguageModel) -> list[str]:
    print(f"\n=== Generation, {name} ===")
    sampled = generate_sentences(model, "sample", NUM_SENTENCES, SEED)
    for sentence in sampled:
        print(" ", sentence)
    print(f"Greedy (5 sentences):")
    for sentence in generate_sentences(model, "greedy", 5, SEED):
        print(" ", sentence)
    print("Sampling (5 sentences):")
    for sentence in sampled[:5]:
        print(" ", sentence)
    return sampled


def comparison_part(first: FirstOrderModel, second: SecondOrderModel, samples: dict[str, list[str]]) -> None:
    print("\n=== Part XIII: comparison ===")
    training = {" ".join(s.lower().split()) for s in DATASET}
    for name, model in [("first-order", first), ("second-order", second)]:
        sentences = model.all_sentences(max_tokens=12)
        print(f"{name}:")
        print(f"  contexts observed: {len(model.counts)}")
        print(f"  parameters (nonzero CPT entries): {model.num_parameters}")
        print(f"  full CPT size |V|^{model.order} x (|V|-1): {model.cpt_size}")
        print(f"  zero entries in observed rows: {model.num_zero_entries_in_observed_rows}")
        print(f"  unseen contexts: {model.num_unseen_contexts}")
        print(f"  distinct sentences of up to 12 words with nonzero probability: {len(sentences)}")
        print(f"  ... of which not in the training data: {len(set(sentences) - training)}")
        print(f"  probability mass ending within 60 steps: {model.probability_mass_ending_within(60):.6f}")
        generated = samples[name]
        print(f"  20 sampled: {len(set(generated))} distinct, {len(set(generated) - training)} not in training data")
        print(f"  greedy: {set(generate_sentences(model, 'greedy', 5, SEED))}")
        print(f"  training sentence probabilities: {[round(model.sentence_probability(s), 4) for s in DATASET]}")


def main() -> None:
    first, second = FirstOrderModel(DATASET), SecondOrderModel(DATASET)
    first_order_part(first)

    samples = {"first-order": generation_part("first-order", first)}
    print("\n=== Second-order CPT and normalisation ===")
    for context in sorted(second.counts):
        print(show_distribution(second, context))
    print(f"all rows sum to 1: {all(abs(t - 1) < 1e-9 for t in normalisation_totals(second).values())}")
    samples["second-order"] = generation_part("second-order", second)

    comparison_part(first, second, samples)

    OUTPUT_DIR.mkdir(exist_ok=True)
    for name, model in [("first_order", first), ("second_order", second)]:
        key = name.replace("_", "-")
        lines = [f"# 20 sentences sampled from the {key} model (seed {SEED})"] + samples[key]
        lines += ["", f"# 5 sentences generated greedily from the {key} model"]
        lines += generate_sentences(model, "greedy", 5, SEED)
        (OUTPUT_DIR / f"generated_{name}.txt").write_text("\n".join(lines) + "\n")
    print(f"\nGenerated sentences saved to {OUTPUT_DIR.name}/")


if __name__ == "__main__":
    main()
