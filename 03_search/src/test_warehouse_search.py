"""Tests for the warehouse search agents. Run with: pytest"""

from __future__ import annotations

import random

import pytest

from warehouse_search import (
    OBSTACLE,
    WAREHOUSE_MAP,
    Action,
    Position,
    SearchResult,
    WarehouseProblem,
    a_star,
    bfs,
    euclidean,
    manhattan,
    scaled,
    zero,
)

TRIVIAL_MAP = [
    "#####",
    "#SG##",
    "#####",
]

NO_SOLUTION_MAP = [
    "#######",
    "#S....#",
    "###.###",
    "#...#G#",
    "#######",
]

# Two routes from S to G: 6 moves along the top, or 10 moves around the bottom.
ALTERNATIVE_MAP = [
    "#########",
    "#S.....G#",
    "#.#####.#",
    "#.......#",
    "#########",
]

# On this map an over-estimating heuristic (2 * Manhattan) returns a 9-move path
# although the shortest path has 7 moves.
GREEDY_TRAP_MAP = [
    ".##..",
    "#....",
    "....G",
    "....#",
    "#.#..",
    "..S#.",
    ".....",
]

ADMISSIBLE_HEURISTICS = [manhattan, euclidean, zero]


def assert_valid_path(problem: WarehouseProblem, result: SearchResult) -> None:
    """Check that the path starts at S, ends at G and moves one free cell at a time."""
    assert result.path[0] == problem.start
    assert result.path[-1] == problem.goal
    for (r, c), (nr, nc) in zip(result.path, result.path[1:]):
        assert abs(r - nr) + abs(c - nc) == 1, "each step must move one cell"
        assert problem.is_free((nr, nc)), f"path enters an obstacle at {(nr, nc)}"


def reference_distances(problem: WarehouseProblem) -> dict[Position, int]:
    """Shortest distance from every reachable cell to G, by repeated relaxation.

    This shares no code with the searches under test, and it also gives the true
    remaining cost h*(n) used to check admissibility.
    """
    dist = {problem.goal: 0}
    changed = True
    while changed:
        changed = False
        for state, d in list(dist.items()):
            for _, neighbour in problem.successors(state):
                if dist.get(neighbour, d + 2) > d + 1:
                    dist[neighbour] = d + 1
                    changed = True
    return dist


def random_grid(rng: random.Random, height: int, width: int, density: float) -> list[str]:
    """Generate a random warehouse with one S and one G."""
    cells = [[OBSTACLE if rng.random() < density else "." for _ in range(width)] for _ in range(height)]
    free = [(r, c) for r in range(height) for c in range(width)]
    (sr, sc), (gr, gc) = rng.sample(free, 2)
    cells[sr][sc] = "S"
    cells[gr][gc] = "G"
    return ["".join(row) for row in cells]


# ---------------------------------------------------------------------------
# The four tests required by the lab
# ---------------------------------------------------------------------------
def test_original_warehouse() -> None:
    problem = WarehouseProblem(WAREHOUSE_MAP)
    result = a_star(problem)
    assert result.found
    assert_valid_path(problem, result)
    assert result.length == reference_distances(problem)[problem.start] == 40
    assert result.expanded == 63


def test_trivial_map_one_step() -> None:
    problem = WarehouseProblem(TRIVIAL_MAP)
    result = a_star(problem)
    assert result.found
    assert result.length == 1
    assert result.actions == [Action.RIGHT]


def test_no_solution_terminates_with_failure() -> None:
    problem = WarehouseProblem(NO_SOLUTION_MAP)
    for result in (a_star(problem), bfs(problem)):
        assert not result.found
        assert result.path == []
        assert result.length is None
        assert result.expanded == 9  # All 9 cells reachable from S, and then it stops.


def test_alternative_paths_returns_shortest() -> None:
    problem = WarehouseProblem(ALTERNATIVE_MAP)
    result = a_star(problem)
    assert result.length == 6
    assert_valid_path(problem, result)


# ---------------------------------------------------------------------------
# Correctness on generated grids
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", range(60))
def test_random_grids_match_reference(seed: int) -> None:
    """On random grids every admissible heuristic and BFS find a shortest path."""
    rng = random.Random(seed)
    problem = WarehouseProblem(random_grid(rng, rng.randint(2, 12), rng.randint(2, 12), rng.choice([0.1, 0.3, 0.5])))
    expected = reference_distances(problem).get(problem.start)
    for result in [bfs(problem)] + [a_star(problem, h) for h in ADMISSIBLE_HEURISTICS]:
        assert result.found == (expected is not None)
        if expected is not None:
            assert_valid_path(problem, result)
            assert result.length == expected


@pytest.mark.parametrize("seed", range(30))
def test_consistent_heuristics_never_expand_a_state_twice(seed: int) -> None:
    rng = random.Random(seed)
    problem = WarehouseProblem(random_grid(rng, rng.randint(3, 12), rng.randint(3, 12), 0.25))
    for heuristic in ADMISSIBLE_HEURISTICS:
        order = a_star(problem, heuristic).expansion_order
        assert len(order) == len(set(order))


def open_grid(height: int, width: int, start: Position, goal: Position) -> WarehouseProblem:
    """An obstacle-free grid with S and G at the given positions."""
    grid = [["."] * width for _ in range(height)]
    grid[start[0]][start[1]], grid[goal[0]][goal[1]] = "S", "G"
    return WarehouseProblem(["".join(row) for row in grid])


def test_open_grid_a_star_expands_fewer_states_than_bfs() -> None:
    """With S and G on one row, only the cells on that row have the smallest f."""
    problem = open_grid(15, 20, (7, 2), (7, 17))
    astar_result, bfs_result = a_star(problem), bfs(problem)
    assert astar_result.length == bfs_result.length == 15
    assert astar_result.expanded == 15
    assert bfs_result.expanded == 213


def test_open_grid_corner_to_corner_gives_a_star_no_advantage() -> None:
    """Every cell of the grid lies on a shortest path, so every cell has the same f."""
    problem = open_grid(15, 20, (0, 0), (14, 19))
    astar_result, bfs_result = a_star(problem), bfs(problem)
    assert astar_result.length == bfs_result.length == 14 + 19
    assert astar_result.expanded == bfs_result.expanded == 15 * 20 - 1


# ---------------------------------------------------------------------------
# Heuristics
# ---------------------------------------------------------------------------
def test_heuristic_values() -> None:
    assert manhattan((0, 0), (3, 4)) == 7
    assert euclidean((0, 0), (3, 4)) == 5
    assert zero((0, 0), (3, 4)) == 0
    assert scaled(manhattan, 2)((0, 0), (3, 4)) == 14


@pytest.mark.parametrize("seed", range(30))
def test_admissible_heuristics_never_overestimate(seed: int) -> None:
    rng = random.Random(seed)
    problem = WarehouseProblem(random_grid(rng, rng.randint(3, 10), rng.randint(3, 10), 0.3))
    for state, true_cost in reference_distances(problem).items():
        for heuristic in ADMISSIBLE_HEURISTICS:
            assert heuristic(state, problem.goal) <= true_cost


def test_doubled_manhattan_overestimates() -> None:
    problem = WarehouseProblem(GREEDY_TRAP_MAP)
    doubled = scaled(manhattan, 2)
    assert any(
        doubled(state, problem.goal) > true_cost
        for state, true_cost in reference_distances(problem).items()
    )


def test_overestimating_heuristic_can_return_a_longer_path() -> None:
    problem = WarehouseProblem(GREEDY_TRAP_MAP)
    optimal = a_star(problem)
    trapped = a_star(problem, scaled(manhattan, 2))
    assert optimal.length == 7
    assert trapped.found
    assert_valid_path(problem, trapped)
    assert trapped.length == 9


# ---------------------------------------------------------------------------
# Map validation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "grid",
    [
        [],
        [""],
        ["S.", "G"],  # ragged
        ["S.X", "..G"],  # invalid character
        ["...", "..G"],  # no start
        ["S..", "..."],  # no goal
        ["SS.", "..G"],  # two starts
        ["S..", ".GG"],  # two goals
    ],
)
def test_invalid_maps_raise(grid: list[str]) -> None:
    with pytest.raises(ValueError):
        WarehouseProblem(grid)
