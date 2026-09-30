# AI Course Labwork

Solutions for the five AI course lab assignments. Each lab has its own folder, with written answers in `answers.md` and code in `src/`.

## Repository structure

```
.
├── 01_neural_models/
│   ├── answers.md
│   ├── xor_points.png
│   └── src/
│       ├── xor_net.py
│       ├── xor_experiments.py
│       ├── three_class_net.py
│       └── loss_curve.png
├── 02_agents/
│   ├── answers.md
│   ├── block_diagram.png
│   └── src/
│       ├── warehouse_agent.py
│       └── test_warehouse_agent.py
├── 03_search/
│   ├── answers.md
│   └── src/
│       ├── warehouse_search.py
│       ├── test_warehouse_search.py
│       └── search_experiments.py
├── 04_logic/
│   ├── answers.md
│   └── src/
│       ├── planner.py
│       ├── test_planner.py
│       ├── logic_experiments.py
│       ├── prolog_verifier.py
│       ├── planner.pl
│       └── road.pl
├── 05_bayesian_networks/
│   ├── answers.md
│   └── src/
│       ├── language_model.py
│       ├── test_language_model.py
│       ├── bn_experiments.py
│       └── outputs/
│           ├── generated_first_order.txt
│           └── generated_second_order.txt
└── README.md
```

## Code

### 01_neural_models
- `src/xor_net.py`: the binary XOR network, a 2-2-1 network with a tanh hidden layer trained with `BCEWithLogitsLoss` (Tasks 3 and 4A).
- `src/xor_experiments.py`: gradient check, symmetry, activation and linear baseline experiments (Tasks 4B-D).
- `src/three_class_net.py`: the three-class extension with softmax and cross-entropy (Task 5).

### 02_agents
- `src/warehouse_agent.py`: goal-based warehouse navigation agent using BFS.
- `src/test_warehouse_agent.py`: pytest tests for the agent.

### 03_search
- `src/warehouse_search.py`: A* and BFS search agents for the warehouse map, with swappable heuristics.
- `src/test_warehouse_search.py`: pytest tests, including generated grids checked against an independent reference.
- `src/search_experiments.py`: the tests, BFS vs A* comparison and heuristic study reported in `answers.md`.

### 04_logic
- `src/planner.py`: a logical planner for the warehouse robot, with actions defined by preconditions and effects, BFS search, and a step-by-step plan verifier.
- `src/test_planner.py`: pytest tests, including random problems checked against an independent exhaustive search.
- `src/logic_experiments.py`: the applicability checks, hand-made plan and Tests A-C reported in `answers.md`.
- `src/planner.pl`, `src/road.pl`: the Prolog knowledge bases for the optional extension.
- `src/prolog_verifier.py`: checks the moves in a plan against `planner.pl` (needs SWI-Prolog).

### 05_bayesian_networks
- `src/language_model.py`: word-level Markov language models built from counts, viewed as Bayesian networks. `FirstOrderModel` and `SecondOrderModel` share one implementation, with the normalisation check and greedy and sampling generation.
- `src/test_language_model.py`: pytest tests against hand counts, an independent recount and the probabilistic invariants.
- `src/bn_experiments.py`: the CPTs, predictions, generation and model comparison reported in `answers.md`. It writes the generated sentences to `src/outputs/`.

## Running

```
cd 01_neural_models/src
python xor_net.py
python xor_experiments.py
python three_class_net.py

cd ../../02_agents/src
python warehouse_agent.py
pytest

cd ../../03_search/src
python warehouse_search.py
python search_experiments.py
pytest

cd ../../04_logic/src
python planner.py
python logic_experiments.py
python prolog_verifier.py
pytest

cd ../../05_bayesian_networks/src
python bn_experiments.py
pytest
```

The neural models code needs PyTorch and matplotlib, and the tests need pytest. The Prolog parts of the logic lab need SWI-Prolog (`swipl`); their tests are skipped if it is not installed.
