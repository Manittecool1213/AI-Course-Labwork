"""Tests for the Markov language models. Run with: pytest

The tests check the model against its probabilistic specification: exact
probabilities computed by hand, the normalisation invariant, and agreement with
an independent count of the training data.
"""

from __future__ import annotations

import math
import random
from collections import Counter

import pytest

from language_model import (
    DATASET,
    END,
    MAX_LENGTH,
    START,
    FirstOrderModel,
    MarkovLanguageModel,
    SecondOrderModel,
    UnseenContextError,
    is_normalised,
    normalisation_totals,
    tokenise,
)

VOCABULARY = ["the", "cat", "dog", "sat", "ran", "on", "to", "mat", "rug", "park"]


@pytest.fixture
def first() -> FirstOrderModel:
    return FirstOrderModel(DATASET)


@pytest.fixture
def second() -> SecondOrderModel:
    return SecondOrderModel(DATASET)


def random_corpus(rng: random.Random) -> list[str]:
    """Random sentences over a small vocabulary."""
    return [" ".join(rng.choices(VOCABULARY, k=rng.randint(1, 8))) for _ in range(rng.randint(1, 12))]


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------
def test_tokenise_lower_cases_and_splits() -> None:
    assert tokenise("The Cat  SAT") == ["the", "cat", "sat"]


def test_padding_uses_order_start_tokens_and_one_end_token(first: FirstOrderModel, second: SecondOrderModel) -> None:
    assert first.padded("The cat") == [START, "the", "cat", END]
    assert second.padded("The cat") == [START, START, "the", "cat", END]


# ---------------------------------------------------------------------------
# The conditional probability tables, checked against hand counts
# ---------------------------------------------------------------------------
def test_first_order_probabilities_after_the(first: FirstOrderModel) -> None:
    # "the" occurs 12 times: cat 3, dog 3, mat 2, rug 2, park 2.
    assert first.distribution(("the",)) == pytest.approx(
        {"cat": 3 / 12, "dog": 3 / 12, "mat": 2 / 12, "rug": 2 / 12, "park": 2 / 12}
    )


def test_first_order_other_rows(first: FirstOrderModel) -> None:
    assert first.distribution(("cat",)) == pytest.approx({"sat": 2 / 3, "ran": 1 / 3})
    assert first.distribution(("dog",)) == pytest.approx({"sat": 2 / 3, "ran": 1 / 3})
    assert first.distribution(("sat",)) == {"on": 1.0}
    assert first.distribution(("ran",)) == {"to": 1.0}
    assert first.distribution((START,)) == {"the": 1.0}
    assert first.distribution(("mat",)) == {END: 1.0}


def test_zero_probability_transitions_are_omitted_and_score_zero(first: FirstOrderModel) -> None:
    assert "dog" not in first.distribution(("cat",))
    assert first.sentence_probability("cat dog") == 0.0


def test_second_order_rows(second: SecondOrderModel) -> None:
    assert second.distribution((START, START)) == {"the": 1.0}
    assert second.distribution((START, "the")) == pytest.approx({"cat": 0.5, "dog": 0.5})
    assert second.distribution(("the", "cat")) == pytest.approx({"sat": 2 / 3, "ran": 1 / 3})
    assert second.distribution(("on", "the")) == pytest.approx({"mat": 0.5, "rug": 0.5})
    assert second.distribution(("to", "the")) == {"park": 1.0}


@pytest.mark.parametrize("order", [1, 2, 3])
@pytest.mark.parametrize("seed", range(20))
def test_cpt_matches_an_independent_count(order: int, seed: int) -> None:
    """Recount the transitions with zip over the padded tokens, and compare."""
    corpus = random_corpus(random.Random(seed))
    model = MarkovLanguageModel(order, corpus)
    reference: dict[tuple[str, ...], Counter[str]] = {}
    for sentence in corpus:
        tokens = [START] * order + sentence.split() + [END]
        for window in zip(*(tokens[i:] for i in range(order + 1))):
            reference.setdefault(window[:-1], Counter())[window[-1]] += 1
    assert set(model.counts) == set(reference)
    for context, row in reference.items():
        total = sum(row.values())
        assert model.distribution(context) == pytest.approx({t: c / total for t, c in row.items()})


# ---------------------------------------------------------------------------
# Normalisation: for every context w, the sum over v of P(v | w) is 1
# ---------------------------------------------------------------------------
def test_default_models_are_normalised(first: FirstOrderModel, second: SecondOrderModel) -> None:
    for model in (first, second):
        assert is_normalised(model)
        for total in normalisation_totals(model).values():
            assert total == pytest.approx(1.0)


@pytest.mark.parametrize("order", [1, 2, 3])
@pytest.mark.parametrize("seed", range(20))
def test_random_models_are_normalised(order: int, seed: int) -> None:
    assert is_normalised(MarkovLanguageModel(order, random_corpus(random.Random(seed))))


def test_normalisation_check_detects_a_broken_model() -> None:
    """A model that divides by (total + 1) gives row sums below 1, and the check must notice."""

    class Broken(FirstOrderModel):
        def distribution(self, context):  # type: ignore[no-untyped-def]
            row = self.counts[self._as_context(context)]
            return {token: count / (sum(row.values()) + 1) for token, count in row.items()}

    broken = Broken(DATASET)
    assert not is_normalised(broken)
    assert normalisation_totals(broken)[("the",)] == pytest.approx(12 / 13)


@pytest.mark.parametrize("seed", range(10))
def test_probability_mass_over_sentences_approaches_one(seed: int) -> None:
    """The sentence probabilities of a proper model add up to 1 (all sentences end eventually)."""
    model = MarkovLanguageModel(random.Random(seed).randint(1, 2), random_corpus(random.Random(seed)))
    masses = [model.probability_mass_ending_within(steps) for steps in (5, 20, 60)]
    assert masses == sorted(masses)
    assert masses[-1] <= 1.0 + 1e-9


def test_default_probability_mass(first: FirstOrderModel, second: SecondOrderModel) -> None:
    assert first.probability_mass_ending_within(60) > 0.9999
    assert second.probability_mass_ending_within(10) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Sentence probabilities (chain rule)
# ---------------------------------------------------------------------------
def test_sentence_probability_is_the_product_of_conditionals(first: FirstOrderModel, second: SecondOrderModel) -> None:
    # P(the|START) P(cat|the) P(sat|cat) P(on|sat) P(the|on) P(mat|the) P(END|mat)
    assert first.sentence_probability("the cat sat on the mat") == pytest.approx(1 * 0.25 * (2 / 3) * 1 * 1 * (2 / 12) * 1)
    assert second.sentence_probability("the cat sat on the mat") == pytest.approx(1 * 0.5 * (2 / 3) * 1 * 1 * 0.5 * 1)


def test_unseen_word_gives_probability_zero_and_log_minus_infinity(first: FirstOrderModel) -> None:
    assert first.sentence_probability("the bird sat") == 0.0
    assert first.sentence_log_probability("the bird sat") == -math.inf


# ---------------------------------------------------------------------------
# Prediction and unseen contexts
# ---------------------------------------------------------------------------
def test_most_probable_word(first: FirstOrderModel) -> None:
    assert first.most_probable(("sat",)) == "on"
    assert first.most_probable(("cat",)) == "sat"


def test_ties_are_broken_by_first_seen(first: FirstOrderModel) -> None:
    # After "the", cat and dog both have probability 0.25; cat was seen first.
    assert first.most_probable(("the",)) == "cat"


def test_unseen_context_raises(first: FirstOrderModel, second: SecondOrderModel) -> None:
    with pytest.raises(UnseenContextError):
        first.distribution(("bird",))
    with pytest.raises(UnseenContextError):
        second.distribution(("dog", "cat"))
    assert "bird" not in first.counts  # Looking up an unseen context must not create it.


def test_wrong_context_length_raises(first: FirstOrderModel, second: SecondOrderModel) -> None:
    with pytest.raises(ValueError):
        first.distribution(("the", "cat"))
    with pytest.raises(ValueError):
        second.distribution(("cat",))


def test_order_must_be_positive() -> None:
    with pytest.raises(ValueError):
        MarkovLanguageModel(0)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", range(50))
def test_sampled_sentences_only_use_observed_transitions(seed: int, first: FirstOrderModel, second: SecondOrderModel) -> None:
    """Every transition in a sampled sentence was seen in training.

    A first-order sentence can occasionally reach the length limit while cycling
    (the dog sat on the dog ran ...). It is then cut off before <END>, so only its
    transitions are checked, not the final step to <END>.
    """
    for model in (first, second):
        sentence = model.generate("sample", random.Random(seed))
        assert sentence
        tokens = model.padded(sentence)[:-1]  # Drop the <END> that the model did not generate.
        for i in range(model.order, len(tokens)):
            assert tokens[i] in model.distribution(tuple(tokens[i - model.order : i]))
        if len(sentence.split()) < MAX_LENGTH:
            assert model.sentence_probability(sentence) > 0


def test_generation_is_reproducible_with_a_seed(first: FirstOrderModel) -> None:
    assert first.generate("sample", random.Random(3)) == first.generate("sample", random.Random(3))


def test_greedy_generation_is_deterministic(first: FirstOrderModel, second: SecondOrderModel) -> None:
    for model in (first, second):
        assert len({model.generate("greedy") for _ in range(5)}) == 1


def test_greedy_first_order_cycles_until_the_length_limit(first: FirstOrderModel) -> None:
    """the -> cat -> sat -> on -> the -> cat ... never reaches <END>, so the limit is needed."""
    sentence = first.generate("greedy")
    assert len(sentence.split()) == MAX_LENGTH
    assert sentence.startswith("the cat sat on the cat sat on")
    assert len(first.generate("greedy", max_length=5).split()) == 5


def test_greedy_second_order_ends(second: SecondOrderModel) -> None:
    assert second.generate("greedy") == "the cat sat on the mat"


def test_unknown_mode_raises(first: FirstOrderModel) -> None:
    with pytest.raises(ValueError):
        first.generate("beam")


def test_sampling_frequencies_match_sentence_probabilities(first: FirstOrderModel, second: SecondOrderModel) -> None:
    rng = random.Random(0)
    n = 20000
    counts = Counter(first.generate("sample", rng) for _ in range(n))
    assert counts["the rug"] / n == pytest.approx(first.sentence_probability("the rug"), abs=0.01)
    assert counts["the dog sat on the mat"] / n == pytest.approx(first.sentence_probability("the dog sat on the mat"), abs=0.01)

    counts = Counter(second.generate("sample", rng) for _ in range(6000))
    for sentence in DATASET:
        assert counts[sentence] / 6000 == pytest.approx(1 / 6, abs=0.02)


# ---------------------------------------------------------------------------
# Comparing the two models
# ---------------------------------------------------------------------------
def test_second_order_model_memorises_the_training_sentences(second: SecondOrderModel) -> None:
    assert set(second.all_sentences()) == set(DATASET)


def test_first_order_model_generates_sentences_that_are_not_in_the_data(first: FirstOrderModel) -> None:
    assert len(set(first.all_sentences(max_tokens=12)) - set(DATASET)) > 0
    assert first.sentence_probability("the dog ran to the mat") > 0


def test_more_context_means_more_parameters_and_more_unseen_contexts(first: FirstOrderModel, second: SecondOrderModel) -> None:
    assert second.num_parameters > first.num_parameters
    assert second.cpt_size > first.cpt_size
    assert second.num_unseen_contexts > first.num_unseen_contexts == 0
    assert (first.num_parameters, second.num_parameters) == (17, 19)
    assert (first.cpt_size, second.cpt_size) == (12 * 11, 144 * 11)
