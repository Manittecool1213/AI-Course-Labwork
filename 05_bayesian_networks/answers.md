# Part I: From Probability to Language

---

### Question 1: Why is the chain-rule decomposition useful for generating text
- It turns one huge joint distribution over whole sentences into a product of small conditional distributions, P(X1) P(X2|X1) P(X3|X1,X2) ..., each of which only has to predict one word given the words before it.
- Generation then follows the same order: choose X1, then X2 given X1, then X3 given the earlier words, and so on. This works for sentences of any length, and only needs a way to estimate the next-word distribution.
- It is also exact, with no independence assumption: the same product gives the probability of any complete sentence.

---

# Part II: A Bayesian Network for Text

---

### Question 2: What independence assumption is being made by this network
- Each word depends only on the word immediately before it: P(Xt | X1, ..., Xt-1) = P(Xt | Xt-1).
- Equivalently, Xt is conditionally independent of X1, ..., Xt-2 given Xt-1 (the Markov property).

---

# Part III: Dataset

---

### Dataset and tokens
- The six sentences from the lab, converted to lower case, with each word as a token.
- `<START>` and `<END>` are added to each sentence, e.g. `<START> the cat sat on the mat <END>`.
- The vocabulary has 12 tokens: the, cat, dog, sat, ran, on, to, mat, rug, park, `<START>`, `<END>`.

---

# Part IV: The Conditional Probability Table

---

### Question 3: Construct P(next word | current word) for the, cat, dog, sat, ran
The word "the" occurs 12 times in the data (6 times at the start, and 6 times after "on" or "to"), so the probabilities after it are counts out of 12. The example numbers in the lab sheet (3/5 and 2/5) are only illustrative.

| Current word | Next word | Count | Probability |
|--------------|-----------|-------|-------------|
| the | cat | 3 | 0.250 |
| the | dog | 3 | 0.250 |
| the | mat | 2 | 0.167 |
| the | rug | 2 | 0.167 |
| the | park | 2 | 0.167 |
| cat | sat | 2 | 0.667 |
| cat | ran | 1 | 0.333 |
| dog | sat | 2 | 0.667 |
| dog | ran | 1 | 0.333 |
| sat | on | 4 | 1.000 |
| ran | to | 2 | 1.000 |

Zero-probability transitions:
- After "the": on, sat, ran, to, the, `<END>`. (A sentence cannot end with "the" here.)
- After "cat" and "dog": everything except sat and ran, e.g. P(dog|cat) = 0 and P(`<END>`|cat) = 0.
- After "sat": everything except "on", e.g. P(the|sat) = 0.
- After "ran": everything except "to", e.g. P(the|ran) = 0.
- In total, in the 11 rows of the table, 104 of the 121 entries are zero.

---

# Part V: Ask an LLM to Implement the Model

---

### Prompt used

Core Task:
Write a simple Python implementation of a first-order autoregressive language model.

Details:
- The model should take a list of tokenised sentences as training data, and add the special tokens <START> and <END> to each sentence.
- Count transitions between consecutive tokens, and construct the conditional distribution P(X_t | X_{t-1}) as P(w_j | w_i) = C(w_i, w_j) / sum_k C(w_i, w_k).
- Display the probabilities for a specified previous token, and predict the most probable next token.
- Generate a sentence by repeatedly sampling the next token, and stop when the <END> token is generated. Also support greedy generation (always the most probable token), and make sure that generation cannot loop forever.
- Do not use a machine-learning library or a pretrained language model. Use ordinary Python data structures and random sampling.
- Include a function that checks that, for every current word w, the probabilities P(v | w) sum to 1.

Programming guidelines:
- Include descriptive comments and docstrings, and use type annotations.
- Write explicit pytest tests that check the model against its probabilistic specification.

### Generated code
Present in the [src](./src/) directory.
- `language_model.py`: the model (`FirstOrderModel`, `SecondOrderModel`), the normalisation check and generation.
- `test_language_model.py`: the pytest tests (205 pass).
- `bn_experiments.py`: the experiments reported below. It also writes the generated sentences to `outputs/`.

---

# Part VI: Inspect the LLM-Generated Code

---

### Question 4: Where in the program are the transition counts stored
- In `self.counts` (line 68 of `language_model.py`), a dictionary from a context (a tuple of the previous tokens) to a `Counter` of the tokens that followed it.
- It is filled by `train` (line 85), which slides over each padded sentence.

### Question 5: Where is P(Xt | Xt-1) computed
- In `distribution` (lines 102 to 103): `count / total`, where `total` is the sum of the counts in the row of the context. It is computed when asked for, from the counts.

### Question 6: How does the program choose the next word
- In `generate` (lines 178 to 201), it depends on the mode.
- Greedy: it takes the most probable word every time (`most_probable`, line 198), i.e. the argmax.
- Sampling: it draws a word at random with `rng.choices`, with the probabilities as weights (line 201).
- The difference: argmax always returns the same word for a given context, so the whole sentence is fixed. Sampling picks word w with probability P(w | context), so likely words are chosen more often, but unlikely words still appear, and the sentences vary from run to run. Sampling reproduces the model's distribution, and argmax ignores everything except its peak.

### Question 7: What happens if the program encounters a word for which no transition has been observed
- `distribution` raises an `UnseenContextError` (line 101). I chose this so that a missing row is not silently mistaken for a probability. For example, `distribution(("bird",))` raises it.
- `sentence_probability` treats such a sentence as having probability 0.
- Generation cannot reach this situation, since every word it produces was seen in training with something after it (at least `<END>`), so its row exists.

---

# Part VII: Test the Probability Model

---

### Normalisation test
`normalisation_totals` computes sum_v P(v | w) for every context w. For both models, every total is 1.000000:

| Model | Rows checked | Result |
|-------|--------------|--------|
| First-order | 11 (`<START>`, the, cat, sat, on, mat, rug, dog, ran, to, park) | all 1.000000 |
| Second-order | 15 | all 1.000000 |

- The tests also check this on 60 random models (orders 1, 2 and 3), and that the probability of a sentence is the product of the conditionals.
- As a stronger test, the probability that a sampled sentence has ended after 60 steps, computed exactly, is 0.999969 for the first-order model and 1.000000 for the second-order model. The probability of all sentences adds up to 1.

### Question 8: If one of the totals is 0.87, what does this tell you about the implementation
- That row is not a probability distribution: 0.13 of the probability is missing. The implementation has a bug in the counting or normalisation.
- Likely causes: dividing by the wrong total (e.g. a count that is larger than the row's own total), leaving out some transitions (e.g. `<END>`), or a mismatch between the counts and the words used.
- The tests check that this check would catch such a bug. A broken model that divides by (total + 1) gives 12/13 = 0.923 for the row of "the", and is detected.

---

# Part VIII: Predicting the Next Word

---

### Next-word distributions and argmax (first-order)

| Current word | Distribution | Most probable |
|--------------|--------------|---------------|
| `<START>` | the: 1.0 | the |
| the | cat: 0.250, dog: 0.250, mat: 0.167, park: 0.167, rug: 0.167 | cat (tied with dog) |
| cat | sat: 0.667, ran: 0.333 | sat |
| dog | sat: 0.667, ran: 0.333 | sat |
| sat | on: 1.0 | on |
| on | the: 1.0 | the |
| ran | to: 1.0 | to |
| to | the: 1.0 | the |
| mat | `<END>`: 1.0 | `<END>` |

- After "the", cat and dog are tied at 0.25. The program breaks the tie by taking the word that was seen first (cat). I checked this in the code and the tests.

### Question 9: Are the most probable predictions always the same as the words that you would personally expect
- No. After "the" the model predicts "cat", but after "sat on the" I would expect "mat" or "rug". The first-order model only sees "the", so it cannot use "sat on", and its generated sentences include "the cat sat on the cat".
- The model is a summary of counts in six sentences. It does not know what a cat or a mat is, or what is grammatical. It reproduces the statistics of its data, including ties and the effects of a tiny dataset.
- Human expectations use meaning, grammar and world knowledge, and they are conditioned on far more context. A probability model can be perfectly consistent (normalised and correct given its data) without being sensible.

---

# Part IX: Generate Text

---

### Generated sentences (first-order, sampling)
20 sentences were generated with seed 0, and saved in `src/outputs/generated_first_order.txt`. Some examples:
- the dog sat on the mat
- the rug
- the cat sat on the park
- the dog sat on the dog ran to the park
- the cat sat on the dog sat on the cat sat on the dog sat on the rug
- the park

- Of the 20 sentences, 11 are distinct, and 9 are not in the training data. Many are not grammatical ("the rug", "the park"), and some are run-on sentences that follow the cycle the → cat → sat → on → the.
- The second-order sentences are saved in `src/outputs/generated_second_order.txt`.

---

# Part X: Deterministic vs Probabilistic Generation

---

### Question 10: Compare the two sets of generated sentences

| Model | Greedy (5 sentences) | Sampling (5 sentences) |
|-------|----------------------|------------------------|
| First-order | The same sentence five times: "the cat sat on the cat sat on the cat ...", which never ends and is cut off at 30 words | the dog sat on the mat / the rug / the rug / the dog sat on the dog ran to the park / the cat sat on the park |
| Second-order | The same sentence five times: "the cat sat on the mat" | the dog sat on the mat / the cat sat on the mat / the cat ran to the park / the dog ran to the park / the dog sat on the rug |

- Sampling produces more variation. Greedy generation has none: the argmax at each step is the same, so every run gives the same sentence.
- The first-order greedy path also loops (the → cat → sat → on → the → cat ...), since the most probable word after "the" is always "cat" (the model cannot see that it just followed "on"), and "the cat sat on" leads back to "the". It never reaches `<END>`, so a length limit is needed.

---

# Part XI: A Second-Order Bayesian Network

---

### Question 11: How does the second-order model differ from the first-order model
- Graph structure: in the first-order model, each Xt has one parent, Xt-1 (a chain). In the second-order model, each Xt has two parents, Xt-2 and Xt-1 (Xt-2 → Xt ← Xt-1, together with the chain edges).
- Conditional probability table: rows are indexed by a pair of words (up to |V|² of them) instead of a single word (|V| rows). Each row is still a distribution over the next word. The second-order table for this data has 15 rows, against 11.
- Context available for prediction: two previous words, not one. It can distinguish "on the" (mat or rug) from "to the" (park), while the first-order model sees just "the".
- Data needed: much more. The number of possible contexts grows as |V|^n, so each row is estimated from fewer examples, and most contexts are never seen.

---

# Part XII: Use the LLM Again

---

### Prompt used

Core Task:
Modify the existing first-order autoregressive model into a second-order model. The model should estimate P(X_t | X_{t-2}, X_{t-1}).

Details:
- What should change in the probabilistic model: the context is now the pair of the previous two tokens, so the counts are indexed by observed triples (X_{t-2}, X_{t-1}, X_t), and P(X_t | X_{t-2}, X_{t-1}) = C(X_{t-2}, X_{t-1}, X_t) / sum_w C(X_{t-2}, X_{t-1}, w).
- Each sentence is padded with two <START> tokens, so that the first word is generated from P(X_1 | <START>, <START>).
- Represent the model using counts of observed triples and use these counts to construct conditional probability distributions.
- Do not replace the model with a neural network or a pretrained language model.
- Keep the same interface, including greedy and sampling generation.

### Generated code
The first- and second-order models share one implementation, `MarkovLanguageModel(order)`, with `FirstOrderModel` and `SecondOrderModel` as the order 1 and 2 versions. The context is the tuple of the previous `order` tokens, and the padding uses `order` start tokens.

### Second-order CPT (all 15 rows)

| Context | Next word: probability |
|---------|------------------------|
| `<START>` `<START>` | the: 1.0 |
| `<START>` the | cat: 0.5, dog: 0.5 |
| the cat | sat: 0.667, ran: 0.333 |
| the dog | sat: 0.667, ran: 0.333 |
| cat sat, dog sat | on: 1.0 |
| cat ran, dog ran | to: 1.0 |
| sat on | the: 1.0 |
| ran to | the: 1.0 |
| on the | mat: 0.5, rug: 0.5 |
| to the | park: 1.0 |
| the mat, the rug, the park | `<END>`: 1.0 |

- All rows sum to 1, and the tests check the second-order probabilities against hand counts and against an independent recount of the data (for orders 1, 2 and 3).

---

# Part XIII: Comparing the Two Models

---

### Comparison

| Measure | First-order | Second-order |
|---------|-------------|--------------|
| Contexts observed (rows of the CPT) | 11 | 15 |
| Distinct parameters (nonzero CPT entries) | 17 | 19 |
| Full CPT size (\|V\|^n contexts x 11 next words) | 132 | 1584 |
| Zero entries in the observed rows | 104 | 146 |
| Contexts never seen (no row) | 0 | 106 (of 121) |
| Distinct sentences of at most 12 words with nonzero probability | 63 | 6 |
| ... of which are not in the training data | 57 | 0 |
| 20 sampled sentences: distinct / not in the training data | 11 / 9 | 6 / 0 |
| Probability of a training sentence | 0.014 to 0.028 | 0.167 (each) |

Examples:
- First-order: "the cat sat on the park" (odd), "the rug" (incomplete), "the dog sat on the dog ran to the park" (run-on). It is creative, and often incoherent.
- Second-order: "the dog sat on the rug", "the cat ran to the park". All sentences are coherent, but they are exactly the six training sentences. The model has memorised the data and generates nothing new.
- The second-order model assigns probability 0 to "the dog ran to the mat" (which is sensible), because the pair "to the" was only followed by "park". The first-order model gives it a nonzero probability.

### Question 12: Why does increasing the amount of context potentially improve prediction, and why can it make the model harder to estimate from limited data
- More context makes the prediction more specific. After "the" there are 5 possible words, but after "on the" there are only 2 (mat or rug), and after "to the" only 1 (park). The distributions are sharper, and they capture dependencies that a single word cannot.
- The conditional probability table grows exponentially with the context: |V|^n rows, each over |V| words (132 entries for order 1 and 1584 for order 2 with 12 tokens). The same amount of data is spread over many more contexts, so each row is estimated from very few examples (1 to 3 here). Most contexts are never seen (106 of 121), and probabilities are 0 for combinations that never happened but are perfectly possible.

---

# Part XIV: The Connection to Modern Language Models

---

### Relationship to modern autoregressive language models
- Both model P(Xt | X1, ..., Xt-1) and generate by sampling one token at a time.
- The difference is how the conditional distribution is represented and learned. Here it is a table of counts over a fixed window. In a modern model, a neural network with a large learned context computes the distribution, trained by gradient descent, with parameters that are learned weights and not explicit probabilities.

---

# Part XV: Reflection on the Role of the LLM

---

### Question 13: Why is Approach B preferable when constructing an intelligent system
- Specifying the intended behaviour: Approach B says exactly what the model is (P(Xt | Xt-1) from transition counts, with sampling), so the LLM implements it, rather than deciding on a model. Approach A leaves the model, the data handling and the special tokens to the LLM.
- Understanding the representation: I know that the model is a CPT of counts, so I know where to look for each part (`counts`, `distribution`, `generate`) and what each should do.
- Validating the implementation: with a specification, there is something to compare the code with, e.g. a hand count of P(cat|the) = 3/12. Without one, the only check is that the output "looks like language".
- Testing probabilistic invariants: the specification implies properties that must hold, such as rows summing to 1, sentence probabilities being products of conditionals, and the total probability of all sentences being 1. These can be tested automatically.
- Distinguishing implementation from model: bugs in the code (a wrong denominator) can be told apart from limitations of the model (the first-order model writing "the cat sat on the cat"). With Approach A, the two get confused.

### Question 14: What did thinking of the language model as a Bayesian network give you
- A representation of dependencies: the graph shows which words each word depends on (one parent or two), and thus what the CPT rows are indexed by.
- A factorisation of the joint distribution: P(X1, ..., XT) is a product of local conditionals, which gives the probability of any sentence (chain rule), and shows that the tables of the model are all that is needed.
- A principled method for generation: sampling each variable given its parents, in order (ancestral sampling), draws sentences from the model's joint distribution. Greedy decoding does not, which is why it loops.
- A way to reason about independence assumptions: the Markov assumption is stated by the graph, and I can see what is lost (e.g. "the cat sat on the cat") and how the second-order model relaxes it.
- A way to understand the effect of increasing context: adding parents multiplies the size of the CPT (132 to 1584 entries) and reduces the data per row.
- A way to test whether an implementation matches its probabilistic specification: each row must be a distribution, sentence probabilities must be products of the conditionals, the total probability of all sentences must be 1, and the frequencies of sampled sentences must match their probabilities. All of these are in the tests.

---

# Deliverables and Reflection on the Use of the LLM

---

### Deliverables
1. First-order model: `FirstOrderModel` in `src/language_model.py`.
2. Second-order model: `SecondOrderModel` in `src/language_model.py`.
3. Conditional probability tables: Parts IV, VIII and XII.
4. Generated text: Parts IX and X, and the files in `src/outputs/`.
5. Normalisation tests: Part VII, and `test_language_model.py`.
6. Answers to Questions 1 to 14: above.
7. Reflection: below.

### How the LLM was used and how I validated its output
- I specified the model (the counting, the special tokens, sampling and greedy modes, and the normalisation check) before asking for code, and the LLM produced the implementation and the tests.
- I validated it against hand counts (e.g. P(cat|the) = 3/12), an independent recount of the data, the normalisation invariant, the exact total probability of all sentences, and the match between the sampling frequencies and the probabilities.
- I inspected the code to answer Questions 4 to 7, and the outputs to answer the rest.

### An example of LLM-generated code that I inspected and corrected
- A generated test asserted that every sampled sentence has a nonzero probability. It failed for the first-order model with seed 21.
- On inspection, the sentence was 30 words long: the first-order model had cycled ("the dog sat on the dog ran to the ..."), and generation was cut off at the length limit before it produced `<END>`. Since the sentence had no `<END>`, the final factor P(`<END>` | last word) was 0, so its probability was 0. The model was behaving correctly, and the test was wrong.
- I fixed it to check that every transition in the sentence was observed in training, and that only complete sentences have nonzero probability. I then measured how often this happens: 10 of 2000 first-order samples (0.5%) reach the limit.
- Another point that I inspected: after "the", cat and dog are tied, and the argmax depends on which was seen first. I made this explicit in the code and added a test.
- A test that looked up a normalisation total by the word "the" also failed with a `KeyError`, because the rows are keyed by tuples such as `("the",)`. This was a small mistake in the test.

---
