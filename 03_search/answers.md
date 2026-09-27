# Task 0

---

### Formulation of the warehouse as a search problem

| Component | Specification |
|-----------|---------------|
| State S | The robot's position (row, column) on the map. The 64 non-obstacle cells are the possible states. |
| Actions A | Up, Down, Left, Right. |
| Transition T | T(s, a) = the neighbouring cell in direction a. It is defined only if that cell is inside the map and not `#`. |
| Initial state s0 | The cell marked `S`, i.e. (1, 1). |
| Goal G | The cell marked `G`, i.e. {(7, 15)}. |
| Cost c | 1 for every move, so the cost of a path is its number of moves. |

### What information is necessary to specify a state
- Only the robot's position (row, column).
- The map is fixed and never changes, so it is part of the problem definition, not the state.

### What makes an action invalid
- Moving into a cell containing `#`.
- Moving outside the map. (The border is all `#`, but the code checks the bounds explicitly so that it works for any map.)

### Is this a deterministic search problem
- Yes. Each valid action leads to exactly one successor state, with no randomness or uncertainty.
- It is also fully observable, static, and has a uniform step cost.

### What would constitute a solution
- A sequence of valid actions that takes the robot from S to G.
- An optimal solution is one with the fewest moves, since each move costs 1.

---

# Task 1

---

### Design of the agent
- State: a `(row, column)` tuple of ints.
- Warehouse: the ASCII map as a list of strings, wrapped in a `WarehouseProblem` class that finds S and G, and rejects invalid maps (ragged, unknown characters, not exactly one S and one G).
- Valid actions: for each of the four offsets, the neighbour must be inside the map and not `#`.
- Goal test: `state == goal`, applied when a state is removed from the frontier, not when it is generated.
- Frontier: a priority queue (heap) of entries (f, tie-breaker, g, state). f = g + h decides the order, and the tie-breaker (an increasing counter) makes states with equal f come out first-in first-out.
- Path reconstruction: a `came_from` dictionary maps each state to the state it was reached from. Walk back from G to S, then reverse.
- Reported on termination: whether a solution was found, the path, the path length, and the number of states expanded (the goal itself is not counted, since it is never expanded).

---

# Task 2

---

### Prompt used

Core Task:
I am implementing a simple goal-based search agent in Python. The environment is a grid represented by an ASCII map. The agent starts at S and must reach G. The symbols # represent obstacles and . represents free cells. The agent can move up, down, left, or right, and every movement has cost 1. Implement A* search.

Details:
- Use Manhattan distance as the heuristic, h(n) = |x - x_G| + |y - y_G|.
- Represent grid positions as (row, column) tuples, and the warehouse as a list of strings. The program must work for any map of this form, not just the one provided. Reject invalid maps.
- Maintain a priority queue as the frontier, and calculate g(n), h(n) and f(n) explicitly.
- Avoid repeatedly expanding the same state. Break ties in f first-in first-out.
- Reconstruct the path when the goal is reached.
- Report whether a solution was found, the path, its length, and the number of states expanded.
- Also provide a BFS version of the same agent, using the same goal test so that the expansion counts are comparable, and make the heuristic a parameter so that it can be replaced.

Programming guidelines:
- Keep the implementation simple and explain the main components of the code.
- Include descriptive comments and docstrings.
- Use type annotations.
- Write explicit pytest tests, which should include other generated grids.

### Generated code
Present in the [src](./src/) directory.
- `warehouse_search.py`: the problem, A*, BFS and the heuristics.
- `test_warehouse_search.py`: the pytest tests (137 pass).
- `search_experiments.py`: the experiments reported below.

---

# Task 3

---

### Test 1: original warehouse
- Path found: yes.
- Path length: 40 moves.
- States expanded: 63.
- Path: Right x4, Down x4, Right x8, Up x2, Left x6, Up x2, Right x8, Down x6.

```
#################
#S****#*********#
#.###*#*#######*#
#...#*#*******#*#
###.#*#######*#*#
#...#*********#*#
#.###########.#*#
#.............#G#
#################
```

- I also checked this against an independent reference. It computes shortest distances by repeated relaxation and shares no code with the searches. It agrees that the shortest distance is 40.
- There are exactly two routes without repeated cells, of lengths 40 and 48. A* returned the shorter.

### Test 2: trivial case
- Map: `#####` / `#SG##` / `#####`.
- Path found: yes, in 1 move (Right). 1 state expanded (only S).

### Test 3: no solution
- Map: the one given in the lab.
- Path found: no. It reported failure after 9 states expanded (every cell reachable from S) and then stopped, with no infinite loop.
- The same holds for BFS.

### Test 4: alternative paths
- Map: a small room with a direct top route (6 moves) and a bottom detour (10 moves).
- A* returned the 6-move top route, which matches the shortest path length found by BFS and by the reference.
- 6 states expanded, against 12 for BFS.

### Additional tests
- 60 random grids: A* with each admissible heuristic, and BFS, always agree with the reference on whether a path exists and on its length.
- Admissible heuristics never overestimate the true remaining cost, and never expand a state twice.
- Invalid maps (empty, ragged, bad character, missing or duplicate S/G) raise errors.

---

# Task 4

---

### Where each concept appears in the program
All line numbers are in `warehouse_search.py`.

| Concept | Where it appears |
|---------|------------------|
| State | A `Position = tuple[int, int]` (line 30), the robot's (row, column) |
| Action | The `Action` enum (line 54), with a (row, column) offset per move |
| Transition | `WarehouseProblem.successors` (line 138), which applies each offset and keeps the valid neighbours |
| Goal test | `WarehouseProblem.is_goal` (line 148), called at line 211 when a state is popped |
| g(n) | `best_g` (line 200), updated at lines 217 to 219 as `new_g = g + STEP_COST` |
| h(n) | The `heuristic` argument, e.g. `manhattan` (line 153), called at line 221 |
| f(n) | `f = new_g + heuristic(...)` at line 221 (and h(start) at line 203, since g = 0) |
| Frontier | The `frontier` heap (line 202), with `heappop` at line 208 and `heappush` at line 222 |
| Visited states | `best_g` and `came_from` (lines 200 and 201), which record every state reached |
| Path reconstruction | `reconstruct_path` (line 177), which follows `came_from` back from G |

### What data structure is used for the A* frontier
- A priority queue implemented as a binary heap (`heapq`), holding tuples (f, tie-breaker, g, state).

### How does the program select the next state to expand
- It pops the entry with the smallest f from the heap. Ties in f go to the entry that was pushed first, via the tie-breaker counter.

### Where is the heuristic calculated
- Only when a successor is added to the frontier (line 221), plus once for the start state (line 203). It is called as `heuristic(neighbour, problem.goal)`.

### Does the program explicitly calculate f(n) = g(n) + h(n)
- Yes, at line 221: `f = new_g + heuristic(neighbour, problem.goal)`. It is stored as the first element of the heap entry, so the heap orders by it.

### How does the program prevent unnecessary repeated exploration
- A state is only pushed again if a strictly cheaper route to it has been found (`new_g < best_g[neighbour]`).
- When an old, more expensive entry is popped later, it is skipped (line 209).
- As a result, with a consistent heuristic (Manhattan, Euclidean, zero), no state is ever expanded twice. The tests check this on 30 random grids.

---

# Task 5

---

### Comparison of BFS and A* on the original warehouse

| Measure | BFS | A* |
|---------|-----|----|
| Solution found | Yes | Yes |
| Path length | 40 | 40 |
| States expanded | 63 | 63 |

### Did both algorithms find a solution
Yes, both did.

### Did they find paths of the same length
Yes, both found a path of length 40. Both are optimal here, since BFS is optimal for equal step costs and Manhattan is admissible.

### Which algorithm expanded fewer states
- Neither: both expanded 63 states. This is every non-goal cell (the map has 64 non-obstacle cells).

### Why might A* expand fewer states
- In general, A* expands only states whose f = g + h does not exceed the optimal cost, so states that look far from the goal are ignored, while BFS expands everything closer to S than G is.
- It did not help here because this map is a set of winding corridors. Every cell has f <= 40, the optimal cost. The goal is in the bottom-right, but the route has to go up and around, so Manhattan distance is misleading and cannot prune anything.
- The heuristic does help on open maps. On a 15x20 empty grid with S and G on the same row, A* expanded 15 states and BFS 213, with the same path length 15.
- However, with S and G in opposite corners of an empty grid, every cell lies on a shortest path, so every cell has the same f and A* expands as many states as BFS (299 each).

### Think About It: what information does the algorithm use to decide where to search next
- BFS uses only the distance from the start, g(n). A* also uses an estimate of the distance remaining to the goal, h(n).
- A* is therefore only better when h(n) is informative. Its advantage depends on how well the heuristic matches the geometry of the map, and here it gives no advantage.

---

# Task 6

---

### Why is the Manhattan heuristic appropriate for the warehouse
- With moves limited to Up, Down, Left and Right, and no obstacles, the shortest path costs exactly |x - x_G| + |y - y_G|.
- Obstacles can only make the true cost larger, so h(n) <= h*(n). Manhattan distance is admissible, and A* returns an optimal path.
- It is also consistent, since one move changes h by exactly 1, which is not more than the step cost of 1.

### Results of the heuristic investigation
Columns: solution found, path length, number of states expanded (in brackets, the number of distinct states among them).

Original warehouse (BFS: length 40, 63 expanded):

| Heuristic | Found | Length | Expanded |
|-----------|-------|--------|----------|
| Manhattan | Yes | 40 | 63 (63) |
| h(n) = 0 | Yes | 40 | 63 (63) |
| Euclidean | Yes | 40 | 63 (63) |
| 2 * Manhattan | Yes | 40 | 66 (63) |

To see the effect on maps where the heuristic matters, I also ran three other maps.

Open 15x20 grid, S and G on one row (BFS: length 15, 213 expanded):

| Heuristic | Found | Length | Expanded |
|-----------|-------|--------|----------|
| Manhattan | Yes | 15 | 15 |
| h(n) = 0 | Yes | 15 | 213 |
| Euclidean | Yes | 15 | 15 |
| 2 * Manhattan | Yes | 15 | 15 |

Open 15x20 grid, S and G in opposite corners (BFS: length 33, 299 expanded):

| Heuristic | Found | Length | Expanded |
|-----------|-------|--------|----------|
| Manhattan | Yes | 33 | 299 |
| h(n) = 0 | Yes | 33 | 299 |
| Euclidean | Yes | 33 | 299 |
| 2 * Manhattan | Yes | 33 | 33 |

A 5x7 map with obstacles, in which the shortest path has 7 moves (BFS: 23 expanded):

| Heuristic | Found | Length | Expanded |
|-----------|-------|--------|----------|
| Manhattan | Yes | 7 | 14 |
| h(n) = 0 | Yes | 7 | 23 |
| Euclidean | Yes | 7 | 15 |
| 2 * Manhattan | Yes | **9** | 9 |

### What happens if the heuristic is replaced by h(n) = 0
- A* still finds an optimal path, but it now orders states only by g, so it behaves like BFS (uniform-cost search) and expands as many states as BFS (63, 213, 299 and 23).
- It never overestimates, so it stays admissible.

### What happens if the heuristic is replaced by Euclidean distance
- It is still admissible (straight-line distance <= Manhattan distance <= true cost), so the path is optimal in every case.
- It is weaker than Manhattan, being smaller or equal. It expanded the same number of states except on the last map (15 against 14).
- The less informed the heuristic, the more states it tends to expand.

### What happens if the heuristic is multiplied by 2
- It is no longer admissible. On an open grid, 2 * Manhattan is twice the true cost.
- On the original warehouse the path was still optimal (40), but 66 states were expanded although only 63 are distinct. Three states were expanded twice. 2 * Manhattan is not consistent, so a state can be reached again via a cheaper route after it was expanded, and it has to be re-opened.
- On the last map, it returned a path of length 9 instead of 7, so it lost the optimality guarantee.
- It expanded far fewer states in the open corner case (33 against 299), so the speed-up is real, but it comes at the risk of a longer path.

### Think About It: what happens to A* when the heuristic becomes too optimistic or too aggressive
- Too optimistic (underestimating, like h = 0 or Euclidean): the path stays optimal, but the search is less focused and expands more states.
- Too aggressive (overestimating, like 2 * Manhattan): the search is drawn strongly towards the goal, and expands fewer states. It can commit to a route that looks promising and never revisit a cheaper one, so the optimality guarantee is lost. It behaves more like greedy search.
- The best heuristic is one that is admissible and as close to the true cost as possible.

---

# Task 7

---

### What parts of the generated code were correct immediately
- BFS, the map parsing and validation, the successor function, path reconstruction, and the main A* loop all behaved correctly on the first run, on all four tests and against the independent reference.

### Did you find any bugs or design problems
- I found no bugs in the search code. The problems I found were in my own test expectations. I expected 8 states to be expanded in the no-solution map when 9 cells are reachable, and I expected A* to beat BFS on an empty grid with S and G in opposite corners, when in fact every cell has the same f.
- One design issue was subtle: the 2 * Manhattan run expanded 66 states while only 63 are distinct. It shows that re-opening states is needed for inconsistent heuristics.

### How did you discover those problems
- Through failing pytest cases, followed by investigating the numbers (counting the reachable cells, computing f for every cell) instead of just changing the expected values.
- The re-expansion was found by recording the order of expansion and comparing its length to the number of distinct states.

### Did the LLM use terminology or data structures that you did not understand
- The tie-breaker counter in the heap entries, and skipping stale entries when they are popped instead of updating them in the heap ("lazy deletion"). Both are needed because `heapq` cannot change the priority of an entry that is already in the heap.

### Did you modify the LLM-generated code
- Yes. I made the heuristic a parameter, added the record of the expansion order, and changed the test expectations described above. I did not change the core algorithm.

### Which tests were most useful
- The comparison against the independent reference on random grids, since it checks optimality and not just that a plausible path is returned.
- The heuristic tests on maps where the answer is known, such as the corner-to-corner grid, which showed that "A* expands fewer states" is not always true.

### Could you have trusted the program without testing it
- No. Every output on the original warehouse looked plausible, and A* and BFS gave the same numbers. Testing was needed to know that the numbers were right, and the extra maps were needed to see where the two algorithms differ.

### What did you understand about A* that you did not understand before implementing it
- That A* is only as good as its heuristic. On this warehouse it gives no advantage over BFS.
- That admissibility guarantees the optimal path, but that a consistent heuristic is what avoids re-expanding states.
- That the goal must be tested when a state is popped, not when it is generated, or optimality can be lost.

### What I designed, what the LLM suggested, what I accepted, what I changed and what I tested
- Designed by me: the problem formulation (Task 0), the design of the agent (Task 1), and the choice of tests and heuristics to compare.
- Suggested by the LLM: the code structure, the tie-breaker and stale-entry handling, and the extra test maps.
- Accepted: the search algorithms and the parsing.
- Changed: see above.
- Tested: all four required tests, 60 random grids against an independent reference, admissibility, repeated expansion, and invalid maps.

---

# Final Reflection

---

### Why is it important to formulate the search problem before writing the search algorithm
- The algorithm is generic. It needs the states, actions, transitions, goal and costs to be defined before it can be applied, and it can only be as correct as this definition.
- Writing it down first also exposes decisions such as what counts as a state, what makes a move invalid, and what to do when there is no solution, which would otherwise be hidden in code.

### In what sense is A* an informed search algorithm
- BFS uses only what it has already seen, the cost from the start, g(n). A* also uses domain knowledge about the goal, through h(n), an estimate of the remaining cost.
- This lets it prefer states that look closer to the goal. It is informed only to the extent that h(n) is accurate, as the original warehouse shows.

### Why does the choice of heuristic matter
- It decides both the correctness and the efficiency of A*. An admissible heuristic guarantees an optimal path, and an overestimating one does not (length 9 instead of 7).
- A more accurate one prunes more states (Manhattan against h = 0 on the open grid: 15 against 213), and one that misleads (like Manhattan on the winding warehouse) prunes nothing.

### What did the LLM contribute to the engineering process
- It turned my design into working, typed and documented code and a set of tests quickly, and suggested implementation details such as the tie-breaker and lazy deletion.
- It did not decide what the correct behaviour was. That came from the problem formulation and from the independent reference used for testing.

### What could go wrong if an engineer simply accepted LLM-generated code without testing it
- The code may look right but be subtly wrong, for example testing the goal too early and returning non-optimal paths, or expanding states repeatedly, and neither would be visible on one map.
- On the original warehouse, all four heuristics return a path of length 40, which hides the fact that 2 * Manhattan is inadmissible. Only tests on other maps expose it.
- Claims made about the result (e.g. "A* is faster") could also be wrong, as in this warehouse, where it is not.

---
