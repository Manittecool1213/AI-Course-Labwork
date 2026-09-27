"""Experiments for the search lab: the four tests, BFS vs A*, and the heuristic study.

Run as a script:

    python search_experiments.py
"""

from __future__ import annotations

from test_warehouse_search import (
    ALTERNATIVE_MAP,
    GREEDY_TRAP_MAP,
    NO_SOLUTION_MAP,
    TRIVIAL_MAP,
    open_grid,
)
from warehouse_search import (
    WAREHOUSE_MAP,
    Heuristic,
    WarehouseProblem,
    a_star,
    bfs,
    euclidean,
    manhattan,
    render_path,
    scaled,
    zero,
)

HEURISTICS: dict[str, Heuristic] = {
    "Manhattan": manhattan,
    "h(n) = 0": zero,
    "Euclidean": euclidean,
    "2 * Manhattan": scaled(manhattan, 2),
}


def row(label: str, problem: WarehouseProblem, heuristic: Heuristic) -> str:
    """Format one A* run as a table row."""
    result = a_star(problem, heuristic)
    distinct = len(set(result.expansion_order))
    return f"| {label} | {result.found} | {result.length} | {result.expanded} | {distinct} |"


def main() -> None:
    print("=== Task 3: tests (A*, Manhattan) ===")
    for name, rows in [
        ("Test 1: original warehouse", WAREHOUSE_MAP),
        ("Test 2: trivial", TRIVIAL_MAP),
        ("Test 3: no solution", NO_SOLUTION_MAP),
        ("Test 4: alternative paths", ALTERNATIVE_MAP),
    ]:
        problem = WarehouseProblem(rows)
        result = a_star(problem)
        print(f"{name}: found={result.found}, length={result.length}, expanded={result.expanded}")
        if result.found and len(rows) < 6:
            print(render_path(problem, result))
        if result.found:
            print("  path:", " ".join(str(p) for p in result.path))
    print("Test 4 BFS length:", bfs(WarehouseProblem(ALTERNATIVE_MAP)).length)

    print("\n=== Task 5: BFS vs A* on the original warehouse ===")
    problem = WarehouseProblem(WAREHOUSE_MAP)
    for name, result in [("BFS", bfs(problem)), ("A*", a_star(problem))]:
        print(f"{name}: found={result.found}, length={result.length}, expanded={result.expanded}")

    print("\n=== Task 6: heuristics (A*), columns: found, length, expanded, distinct expanded ===")
    maps = {
        "Original warehouse": WarehouseProblem(WAREHOUSE_MAP),
        "Open grid, S and G on one row": open_grid(15, 20, (7, 2), (7, 17)),
        "Open grid, S and G in corners": open_grid(15, 20, (0, 0), (14, 19)),
        "Alternative paths map": WarehouseProblem(ALTERNATIVE_MAP),
        "Greedy trap map": WarehouseProblem(GREEDY_TRAP_MAP),
    }
    for map_name, problem in maps.items():
        print(f"\n{map_name}")
        print(f"BFS reference: length {bfs(problem).length}, expanded {bfs(problem).expanded}")
        for name, heuristic in HEURISTICS.items():
            print(row(name, problem, heuristic))


if __name__ == "__main__":
    main()
