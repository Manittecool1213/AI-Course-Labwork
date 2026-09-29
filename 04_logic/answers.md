# Task 0

---

### What is the initial state I
I = {At(Robot, A), At(Package, A)}

### What is the goal G
G = {At(Package, C)}

### List the actions available to the robot
- Move(X, Y) for the connected pairs (A, B), (B, A), (B, C) and (C, B).
- PickUp(Package, L) for each location L.
- Drop(Package, L) for each location L.

### For each action, identify its preconditions and effects

| Action | Preconditions | Effects |
|--------|---------------|---------|
| Move(X, Y) | At(Robot, X) | ¬At(Robot, X), At(Robot, Y) |
| PickUp(Package, L) | At(Robot, L), At(Package, L) | ¬At(Package, L), Holding(Package) |
| Drop(Package, L) | At(Robot, L), Holding(Package) | ¬Holding(Package), At(Package, L) |

### Is the action PickUp(Package, A) applicable in the initial state
- Yes. Its preconditions are At(Robot, A) and At(Package, A), and both are in I.

### Is the action Drop(Package, C) applicable in the initial state
- No. Its preconditions are At(Robot, C) and Holding(Package), and neither is in I. The robot is at A, and it is not holding the package.

### Think About It: when is an action applicable
- An action being in the list of available actions is not enough. It is applicable only if I |= Preconditions(a), i.e. every one of its preconditions is true in the current state.

---

# Task 1

---

### Manually constructed plan
Plan: PickUp(Package, A), Move(A, B), Move(B, C), Drop(Package, C)

| State | Facts |
|-------|-------|
| S0 | At(Robot, A), At(Package, A) |
| S1 (after PickUp(Package, A)) | At(Robot, A), Holding(Package) |
| S2 (after Move(A, B)) | At(Robot, B), Holding(Package) |
| S3 (after Move(B, C)) | At(Robot, C), Holding(Package) |
| S4 (after Drop(Package, C)) | At(Robot, C), At(Package, C) |

- S4 |= G, since At(Package, C) is in S4, so this is a valid plan.
- Every action's preconditions hold in the state before it: At(Robot, A) and At(Package, A) in S0; At(Robot, A) in S1; At(Robot, B) in S2; and At(Robot, C) and Holding(Package) in S3.

### The example sequence in the lab sheet
- The sheet suggests the sequence Move(A, B), PickUp(Package, B), Move(B, C), Drop(Package, C).
- This is not a valid plan. The package is at A, not B, so PickUp(Package, B) needs At(Package, B), which is false in S1. The package has to be picked up before the robot leaves A.
- This is an example of a sequence that looks reasonable, but is not valid.

---

# Task 2

---

### Prompt used

Core Task:
I want to implement a simple planning agent in Python. Represent a state as a set of logical propositions.

Details:
- Each action should contain: a name; positive preconditions; negative preconditions; positive effects; negative effects.
- An action is applicable if all of its preconditions are satisfied by the current state. When an action is applied: (1) remove its negative effects from the state; (2) add its positive effects to the state.
- Use breadth-first search to find a sequence of actions that achieves a specified goal.
- The program should also: detect when no plan exists; print the resulting sequence of actions; print the states reached after each action.
- The warehouse problem: locations A, B and C; the robot can move between A and B, and between B and C; the robot starts at A with the package; the goal is At(Package, C). The Move, PickUp and Drop actions are as described in the table above.
- Provide a way to check a proposed plan independently of the search, by executing it step by step.

Programming guidelines:
- Explain the implementation and identify any assumptions you make.
- Include descriptive comments and docstrings, and use type annotations.
- Write explicit pytest tests, including generated problems.

### Generated code
Present in the [src](./src/) directory.
- `planner.py`: the actions, the BFS planner, and the plan verifier.
- `test_planner.py`: the pytest tests (110 pass; 10 of these use Prolog).
- `logic_experiments.py`: the experiments reported below.
- `planner.pl`, `road.pl`, `prolog_verifier.py`: the optional Prolog extension.

### Think About It: where the ideas from the specification appear in the program
All line numbers are in `planner.py`.

| Idea | Question | Where it appears |
|------|----------|------------------|
| Preconditions | When is an action applicable? | `Action.is_applicable` (line 49): the state must contain every positive precondition, and none of the negative ones. It is used at line 163. |
| Effects | How does the state change? | `Action.apply` (line 67): `(state - neg_eff) | pos_eff`, so the negative effects are removed first and the positive ones added. |
| Goal | When does planning terminate? | Line 159: `goal <= state`, i.e. the state contains all the goal propositions. If the frontier empties first, `find_plan` returns None (line 168). |
| BFS | How are alternative plans explored? | A `deque` frontier (line 154) with `popleft`, so states are expanded in order of the number of actions. States already reached are not queued again (line 165). |

- Assumptions made: a state is a set of true facts (anything not in the set is false), actions are deterministic, and a goal is satisfied by any state containing all of its propositions.

---

# Task 3

---

### Test A: solvable problem
- Initial state: {At(Robot, A), At(Package, A)}
- Goal: {At(Package, C)}
- Plan found: yes.
- Plan: PickUp(Package, A), Move(A, B), Move(B, C), Drop(Package, C). It has 4 actions, and 7 states were expanded.
- Valid: yes. I checked every action by hand against the states, and also independently with `verify_plan`, which executes the plan step by step. The states in the plan match the states computed independently.

### Test B: impossible problem
- Initial state: {At(Robot, A), At(Package, A)}
- Goal: {At(Package, C)}
- Change: the PickUp actions were removed.
- Plan found: no. The planner printed "No plan found", and did not invent an action.
- Valid: not applicable. There is no plan, and none exists: without PickUp, the package can never leave A.

### Test C: irrelevant actions
- Initial state: {At(Robot, A), At(Package, A)}

| Case | Actions | Goal | Plan found | Valid |
|------|---------|------|------------|-------|
| C1 | Only the Move actions | At(Robot, C) | Yes: Move(A, B), Move(B, C) | Yes |
| C2 | Only the Move actions | At(Package, C) | No | Not applicable |
| C3 | All actions plus Shortcut(A, C), which moves only the robot | At(Package, C) | Yes: PickUp(Package, A), Shortcut(A, C), Drop(Package, C) | Yes |

- In C1 the robot reaches C, but the package is still at A. In C2, this is not mistaken for achieving At(Package, C), and the planner reports that there is no plan.
- In C3, the planner does not use the shortcut alone. It picks up the package first, and it finds a plan that is shorter than the one in Test A (3 actions instead of 4).

### Additional tests
- 80 random planning problems (with positive and negative preconditions and effects) were compared with an independent search that expands whole layers of states. The planner agrees on whether a plan exists, and on the shortest plan length. 30 of these are solvable, and 50 are not.
- The verifier rejects a plan with any one of the four actions removed, and rejects the lab's example sequence.

---

# Task 4

---

### Complete the description

| Step |
|------|
| Current state |
| ↓ |
| Check action preconditions |
| ↓ |
| **Select the applicable actions** (those with S \|= Preconditions(a)) |
| ↓ |
| Generate successor state (S' = Apply(S, a)) |
| ↓ |
| Search over alternatives |
| ↓ |
| Goal? |

### Explain in your own words how logical reasoning and search work together to produce a plan
- Logic determines what is possible. For a state S and an action a, it decides whether S |= Preconditions(a), and if so, what the next state S' = Apply(S, a) is. This defines the successors of a state.
- Search determines what to try. There are usually many applicable actions in a state, and each leads to a different state. BFS decides the order in which these are explored, and which states have been seen already.
- Without logic, the search would not know which actions are allowed, and could return an impossible plan. Without search, logic would tell us what is possible in one step, but not which sequence achieves the goal.
- This is the same search as in the previous lab, but here the states are sets of facts and the successors come from preconditions and effects, instead of a map.

### Think About It
- Logic determines what is possible; search determines what to try. In the code, `is_applicable` (line 163) is the logic, and the queue with the record of visited states (lines 154 to 166) is the search.

---

# Task 5

---

### Can the LLM verify its own plan
The LLM was asked: "For every action in the plan, identify its preconditions and show that those preconditions are satisfied in the state in which the action is executed."

Its explanation:

| Action | Preconditions | Why they hold |
|--------|---------------|---------------|
| PickUp(Package, A) | At(Robot, A), At(Package, A) | Both are in the initial state |
| Move(A, B) | At(Robot, A) | PickUp does not delete it |
| Move(B, C) | At(Robot, B) | Move(A, B) added it |
| Drop(Package, C) | At(Robot, C), Holding(Package) | Move(B, C) added the first, and PickUp added the second, which nothing has deleted |

- I compared this with the states computed by the Python program. The two agree: the program reports no unmet preconditions at any step, and the states are S1 = {At(Robot, A), Holding(Package)}, S2 = {At(Robot, B), Holding(Package)}, S3 = {At(Robot, C), Holding(Package)}, S4 = {At(Robot, C), At(Package, C)}.
- I also tested the explanation on the sequence from the lab sheet, which is invalid. A fluent explanation of it could easily be written ("the robot moves to B, picks up the package, ..."), but executing it shows that step 2 has an unmet precondition, At(Package, B).

### Which should you trust more: (a) the LLM's explanation or (b) the independently executed state transitions
- (b), the independently executed state transitions.
- The explanation is text that was generated to sound plausible, and it is only correct if the model reasoned correctly. The execution actually computes each state from the preconditions and effects, so it cannot state a precondition that does not hold. It is also repeatable.
- The explanation is still useful, since it says what to check, but it is not a verification. A generated explanation is not the same as an independent verification.

---

# Submission notes

---

### Which parts of the program were generated or modified with the assistance of an LLM
- Generated: the structure of the planner (`Action`, `find_plan`), the printing of the plan and the states, the tests, the Prolog files, and the code that calls Prolog.
- Written or decided by me: the planning problem and the action definitions (Tasks 0 and 1), the choice of BFS, the three tests (A, B, C), and the decision to add an independent verifier and a comparison with an exhaustive search.
- Modified after generation: the random test generator was too easy (36 of its 52 solvable problems needed no actions), so I changed it so that the goal is not true in the initial state. After this, all 30 solvable problems need at least one action (up to 4).

---

# Reflection Questions

---

### Why is it useful to specify action preconditions and effects before asking an LLM to write the planner
- They are the precise specification of the problem. The planner is just a general procedure that applies them, so the LLM only needs to implement it, and does not have to decide what an action means.
- With them written down, I can check the generated code against the specification, for example that PickUp requires both At(Robot, L) and At(Package, L). Without them, I would not know what the program should do, so I could not tell whether it is right.

### Give an example of an error that could occur if the planner failed to check an action's preconditions
- The planner could apply Drop(Package, C) in the initial state, giving the "plan" Drop(Package, C) of length 1: the package appears at C even though the robot is at A and is not holding it.
- Or it could apply Move(B, C) when the robot is at A, so the robot appears to jump from A to C.

### Why is a plan that "looks reasonable" not necessarily a valid plan
- Validity depends on the state at every step, and this is not visible in the list of actions. The sequence Move(A, B), PickUp(Package, B), Move(B, C), Drop(Package, C) from the lab sheet looks reasonable, but the package is at A, so step 2 is impossible.

### What did the LLM contribute to the implementation
- The code for the planner: the data structures, the BFS with a set of visited states, printing the plan and states, and the tests. It saved time on boilerplate that would have taken longer to write by hand.

### What did you have to verify independently
- That every action in the plan satisfies its preconditions in the state where it is executed (by hand and with `verify_plan`).
- That the planner reports "No plan found" when none exists, and does not confuse the robot's location with the package's.
- That the plans are shortest, by comparison with an independent exhaustive search on random problems.
- That the moves are supported by the warehouse knowledge, using Prolog.

### In this laboratory, where is logical reasoning being used
- In deciding whether an action is applicable (S |= Preconditions(a)), in computing the new state from the effects, and in the goal test (S |= G). It is also used in the verifier, and in Prolog, where it is done by inference from facts and rules.

### How is planning related to the search algorithms studied in the previous module
- A planning problem is a search problem: the states are sets of facts, the actions are the operators, the transition function comes from the preconditions and effects, the initial state is I, the goal test is S |= G, and each action costs 1.
- BFS here is the same algorithm as in the search lab, with the same properties: it is complete, and finds a plan with the fewest actions. The heuristic search from the previous module (A*) could also be applied, with a heuristic such as the number of goal facts that are not yet true.

---

# Optional: Prolog as a Logical Verifier

---

### Task 6: Prolog as a plan verifier
The file `planner.pl` contains the `connected` facts and the rule `can_move(X,Y) :- connected(X,Y).`

| Query | Result |
|-------|--------|
| `?- can_move(a,b).` | true |
| `?- can_move(a,c).` | false |

### Why does Prolog return true for can_move(a,b)
- The fact `connected(a,b)` is in the program. The rule says that `can_move(X,Y)` holds if `connected(X,Y)` does, so with X = a and Y = b, the body of the rule is a fact, and the head follows.

### Why does it not establish can_move(a,c)
- There is no fact `connected(a,c)`, and no other rule that could give `can_move(a,c)`. Prolog treats what it cannot prove as false (the closed-world assumption). The robot could go from a to c through b, but `can_move` only covers one step.

### What is the relationship between the Prolog rule can_move(X,Y) and the logical implication Connected(X, Y) → CanMove(X, Y)
- The rule is that implication, written backwards: `head :- body` means body → head. X and Y are universally quantified variables, and the rule reads "for all X and Y, if X is connected to Y, then the robot can move from X to Y".

### Task 7: using Prolog to check a proposed plan
The plan from the Python planner contains Move(a, b) and Move(b, c). With `valid_move(X,Y) :- connected(X,Y).` added:

| Query | Result |
|-------|--------|
| `?- valid_move(a,b).` | true |
| `?- valid_move(b,c).` | true |
| `?- valid_move(a,c).` | false |

- The Python planner's moves were also checked by running Prolog from Python (`prolog_verifier.py`). Both moves in the plan, (a, b) and (b, c), are supported.

### Challenge: the Python planner proposes Move(a, c)
- `valid_move(a,c)` fails. The action is not supported by the warehouse knowledge, so it should be rejected. Prolog is used here to check that the Python program's proposal is consistent with the logical description of the warehouse, independently of how the proposal was generated.
- The Python planner never proposes this action, because Move(A, C) is not among its actions.
- I also added a rule `valid_route`, which checks a whole route one move at a time. `valid_route([a,b,c])` succeeds and `valid_route([a,c])` fails.

### Task 8: connect Prolog to logical reasoning
The program `road.pl` has the fact `wet_road.` and the rules `slippery :- wet_road.` and `reduce_speed :- slippery.`

- `?- reduce_speed.` succeeds. To prove `reduce_speed`, Prolog needs `slippery`, and to prove that it needs `wet_road`, which is a fact.
- As a chain of implications: wet_road (fact) ⇒ (wet_road → slippery) ⇒ slippery ⇒ (slippery → reduce_speed) ⇒ reduce_speed.
- In the required form: Fact ⇒ Rule ⇒ Rule ⇒ Conclusion, i.e. wet_road ⇒ slippery ⇒ reduce_speed.

### What is the difference between a Prolog fact and a Prolog rule
- A fact is unconditionally true (`connected(a,b).`). A rule is conditionally true: its head holds if its body can be proved (`can_move(X,Y) :- connected(X,Y).`).

### How does a Prolog query correspond to asking whether something follows from a knowledge base
- The facts and rules are the knowledge base, and a query is a statement to prove from it. If Prolog can derive the query by chaining rules and facts, it answers true, which corresponds to the query being entailed by the knowledge base. If it cannot, it answers false, which only means "not provable" (with variables, it also returns the values that make it true).

### Why might it be useful to use a Prolog program to verify a plan generated by a Python program
- It checks the plan against the knowledge, not against the code that produced it. If the Python planner has a bug, the Prolog check is unlikely to have the same one.

### What advantage does an independent verifier provide when the original plan was generated with the help of an LLM
- The LLM's output could be plausible but wrong, and the LLM cannot be relied on to notice its own mistakes (Task 5). An independent verifier has a simple, precise specification that I can read and trust, so it provides evidence that does not depend on trusting the generator: generate, then independently verify.
- The limit is that the verifier is only as correct as its knowledge base, which is also written by someone, and a verifier that only checks moves does not check picking up and dropping the package.

---
