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
├── 04_logic/
├── 05_bayesian_networks/
└── README.md
```

`03_search`, `04_logic` and `05_bayesian_networks` are not started yet.

## Code

### 01_neural_models
- `src/xor_net.py`: the binary XOR network, a 2-2-1 network with a tanh hidden layer trained with `BCEWithLogitsLoss` (Tasks 3 and 4A).
- `src/xor_experiments.py`: gradient check, symmetry, activation and linear baseline experiments (Tasks 4B-D).
- `src/three_class_net.py`: the three-class extension with softmax and cross-entropy (Task 5).

### 02_agents
- `src/warehouse_agent.py`: goal-based warehouse navigation agent using BFS.
- `src/test_warehouse_agent.py`: pytest tests for the agent.

## Running

```
cd 01_neural_models/src
python xor_net.py
python xor_experiments.py
python three_class_net.py

cd ../../02_agents/src
python warehouse_agent.py
pytest
```

The neural models code needs PyTorch and matplotlib, and the agents tests need pytest.
