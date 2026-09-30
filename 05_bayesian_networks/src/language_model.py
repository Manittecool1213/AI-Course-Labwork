"""Autoregressive language models as Bayesian networks, built from counts.

A Markov language model of order n estimates P(X_t | X_{t-n}, ..., X_{t-1}).
As a Bayesian network, each token X_t has the previous n tokens as parents, and
the model is stored as a conditional probability table (CPT) with one row per
context. The CPT is estimated by counting how often each token follows each
context in the training data:

    P(w | context) = C(context, w) / sum_v C(context, v)

Every sentence is padded with n <START> tokens at the front and one <END> token
at the back, so that the first word is generated from P(X_1 | <START> ...).

No machine-learning library is used, only ordinary Python data structures.
"""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict

START = "<START>"
END = "<END>"

# A context is the tuple of the previous `order` tokens.
Context = tuple[str, ...]

DATASET = [
    "the cat sat on the mat",
    "the cat sat on the rug",
    "the dog sat on the mat",
    "the dog ran to the park",
    "the cat ran to the park",
    "the dog sat on the rug",
]

MAX_LENGTH = 30  # Safety limit on the number of tokens in a generated sentence.


class UnseenContextError(KeyError):
    """Raised when asking for the distribution after a context never seen in training."""


def tokenise(sentence: str) -> list[str]:
    """Lower-case a sentence and split it into word tokens (without special tokens)."""
    return sentence.lower().split()


class MarkovLanguageModel:
    """A word-level Markov language model of a given order, estimated from counts.

    Attributes:
        order: The number of previous tokens the next token depends on.
        counts: For each context, how many times each token followed it.
    """

    def __init__(self, order: int, sentences: list[str] | None = None) -> None:
        """Create a model and, if sentences are given, train it on them.

        Args:
            order: The number of previous tokens used as context (at least 1).
            sentences: Training sentences, as plain text.
        """
        if order < 1:
            raise ValueError("The order must be at least 1.")
        self.order = order
        self.counts: dict[Context, Counter[str]] = defaultdict(Counter)
        if sentences is not None:
            self.train(sentences)

    # ------------------------------------------------------------------
    # Training: counting transitions
    # ------------------------------------------------------------------
    def padded(self, sentence: str) -> list[str]:
        """Tokenise a sentence and add `order` <START> tokens and one <END> token."""
        return [START] * self.order + tokenise(sentence) + [END]

    def train(self, sentences: list[str]) -> None:
        """Count how often each token follows each context in the sentences."""
        for sentence in sentences:
            tokens = self.padded(sentence)
            for i in range(self.order, len(tokens)):
                context = tuple(tokens[i - self.order : i])
                self.counts[context][tokens[i]] += 1

    # ------------------------------------------------------------------
    # The conditional probability table P(X_t | context)
    # ------------------------------------------------------------------
    def distribution(self, context: Context) -> dict[str, float]:
        """Return P(next token | context) as a dictionary over the tokens seen after it.

        Tokens that were never observed after the context have probability 0 and are omitted.

        Raises:
            UnseenContextError: If the context never occurred in the training data.
        """
        context = self._as_context(context)
        row = self.counts.get(context)
        if not row:
            raise UnseenContextError(context)
        total = sum(row.values())
        return {token: count / total for token, count in row.items()}

    def cpt(self) -> dict[Context, dict[str, float]]:
        """Return the whole conditional probability table, one row per observed context."""
        return {context: self.distribution(context) for context in self.counts}

    def _as_context(self, context: Context | str) -> Context:
        """Accept a single word for a first-order model, or a tuple of `order` words."""
        if isinstance(context, str):
            context = (context,)
        if len(context) != self.order:
            raise ValueError(f"A context needs {self.order} token(s), got {len(context)}.")
        return tuple(context)

    def most_probable(self, context: Context) -> str:
        """Return argmax_w P(w | context). Ties go to the token that was seen first."""
        distribution = self.distribution(context)
        return max(distribution, key=distribution.__getitem__)

    # ------------------------------------------------------------------
    # Model size
    # ------------------------------------------------------------------
    @property
    def vocabulary(self) -> set[str]:
        """Every token in the training data, including <START> and <END>."""
        return {token for row in self.counts.values() for token in row} | {START}

    @property
    def num_parameters(self) -> int:
        """The number of nonzero entries in the CPT, i.e. distinct (context, token) pairs seen."""
        return sum(len(row) for row in self.counts.values())

    @property
    def cpt_size(self) -> int:
        """The number of entries the full CPT could have: |V|^order contexts times |V| - 1 next tokens.

        <START> can never be generated, so it is not counted as a possible next token.
        """
        v = len(self.vocabulary)
        return v**self.order * (v - 1)

    @property
    def num_zero_entries_in_observed_rows(self) -> int:
        """Zero-probability entries within the rows of contexts that were observed."""
        return len(self.counts) * (len(self.vocabulary) - 1) - self.num_parameters

    @property
    def num_unseen_contexts(self) -> int:
        """Contexts (of non-<END> tokens) with no row at all, so no prediction can be made from them."""
        allowed = len(self.vocabulary) - 1  # <END> never appears inside a context.
        return allowed**self.order - len(self.counts)

    # ------------------------------------------------------------------
    # Probabilities of sentences and generation
    # ------------------------------------------------------------------
    def sentence_probability(self, sentence: str) -> float:
        """Return P(sentence) as the product of the conditional probabilities (chain rule).

        The probability is 0 if any transition was never observed.
        """
        tokens = self.padded(sentence)
        probability = 1.0
        for i in range(self.order, len(tokens)):
            context = tuple(tokens[i - self.order : i])
            try:
                probability *= self.distribution(context).get(tokens[i], 0.0)
            except UnseenContextError:
                return 0.0
        return probability

    def sentence_log_probability(self, sentence: str) -> float:
        """Return log P(sentence), or -infinity if the sentence has probability 0."""
        probability = self.sentence_probability(sentence)
        return math.log(probability) if probability > 0 else -math.inf

    def generate(self, mode: str = "sample", rng: random.Random | None = None, max_length: int = MAX_LENGTH) -> str:
        """Generate a sentence by repeatedly choosing the next token until <END>.

        Args:
            mode: "sample" draws each token from P(w | context); "greedy" always
                takes the most probable token.
            rng: The random number generator used for sampling.
            max_length: The most tokens to generate. Greedy generation can cycle
                forever without reaching <END>, so it needs a limit.

        Returns:
            The generated words, joined by spaces, without <START> and <END>.
        """
        if mode not in ("sample", "greedy"):
            raise ValueError(f"Unknown mode {mode!r}; use 'sample' or 'greedy'.")
        rng = rng or random.Random()
        context: Context = (START,) * self.order
        words: list[str] = []
        while len(words) < max_length:
            if mode == "greedy":
                token = self.most_probable(context)
            else:
                distribution = self.distribution(context)
                token = rng.choices(list(distribution), weights=list(distribution.values()))[0]
            if token == END:
                break
            words.append(token)
            context = context[1:] + (token,)
        return " ".join(words)

    def probability_mass_ending_within(self, steps: int) -> float:
        """The probability that a sampled sentence has ended after at most `steps` tokens.

        Computed exactly by propagating the distribution over contexts, so it is
        an independent check that the model is a proper distribution over sentences:
        it should approach 1 as `steps` grows.
        """
        state: dict[Context, float] = {(START,) * self.order: 1.0}
        ended = 0.0
        for _ in range(steps):
            next_state: dict[Context, float] = defaultdict(float)
            for context, mass in state.items():
                for token, p in self.distribution(context).items():
                    if token == END:
                        ended += mass * p
                    else:
                        next_state[context[1:] + (token,)] += mass * p
            state = next_state
        return ended

    def all_sentences(self, max_tokens: int = 12) -> dict[str, float]:
        """Every sentence of at most `max_tokens` words that has nonzero probability, with its probability."""
        found: dict[str, float] = {}

        def extend(context: Context, words: list[str], probability: float) -> None:
            for token, p in self.distribution(context).items():
                if token == END:
                    found[" ".join(words)] = probability * p
                elif len(words) < max_tokens:
                    extend(context[1:] + (token,), words + [token], probability * p)

        extend((START,) * self.order, [], 1.0)
        return found


class FirstOrderModel(MarkovLanguageModel):
    """The first-order model, P(X_t | X_{t-1}): the Bayesian network X1 -> X2 -> X3 -> ..."""

    def __init__(self, sentences: list[str] | None = None) -> None:
        super().__init__(1, sentences)


class SecondOrderModel(MarkovLanguageModel):
    """The second-order model, P(X_t | X_{t-2}, X_{t-1}): each X_t has two parents."""

    def __init__(self, sentences: list[str] | None = None) -> None:
        super().__init__(2, sentences)


def normalisation_totals(model: MarkovLanguageModel) -> dict[Context, float]:
    """For every context w, compute sum_v P(v | w), which should be 1."""
    return {context: sum(row.values()) for context, row in model.cpt().items()}


def is_normalised(model: MarkovLanguageModel, tolerance: float = 1e-9) -> bool:
    """Return True if every row of the CPT sums to 1 (within a tolerance)."""
    return all(abs(total - 1.0) <= tolerance for total in normalisation_totals(model).values())
