"""Tests for the warehouse navigation agent. Run with: pytest"""

from __future__ import annotations

import random

import pytest

from warehouse_agent import (
    OBSTACLE,
    WAREHOUSE_MAP,
    Action,
    describe_plan,
    find_actions,
    is_free,
    move,
    parse_grid,
)


def assert_valid_plan(rows: list[str], actions: list[Action]) -> None:
    """Check that following the actions from S stays on free cells and ends at G."""
    grid, start, goal = parse_grid(rows)
    position = start
    for action in actions:
        position = move(position, action)
        assert is_free(grid, position), f"Plan enters an obstacle or leaves the grid at {position}"
    assert position == goal


def reference_distance(rows: list[str]) -> int | None:
    """Shortest S-to-G distance by repeated relaxation (independent of BFS)."""
    grid, start, goal = parse_grid(rows)
    dist = {start: 0}
    changed = True
    while changed:
        changed = False
        for (r, c), d in list(dist.items()):
            for action in Action:
                nxt = move((r, c), action)
                if is_free(grid, nxt) and dist.get(nxt, d + 2) > d + 1:
                    dist[nxt] = d + 1
                    changed = True
    return dist.get(goal)


def random_grid(rng: random.Random, height: int, width: int, density: float) -> list[str]:
    """Generate a random walled warehouse with one S and one G."""
    cells = [
        [OBSTACLE if rng.random() < density else "." for _ in range(width)]
        for _ in range(height)
    ]
    free = [(r, c) for r in range(height) for c in range(width)]
    (sr, sc), (gr, gc) = rng.sample(free, 2)
    cells[sr][sc] = "S"
    cells[gr][gc] = "G"
    return ["".join(row) for row in cells]


# ---------------------------------------------------------------------------
# The provided warehouse
# ---------------------------------------------------------------------------
def test_default_warehouse_has_valid_shortest_path() -> None:
    actions = find_actions(WAREHOUSE_MAP)
    assert actions is not None
    assert_valid_plan(WAREHOUSE_MAP, actions)
    assert len(actions) == reference_distance(WAREHOUSE_MAP)


# ---------------------------------------------------------------------------
# Hand-made grids
# ---------------------------------------------------------------------------
def test_straight_corridor() -> None:
    assert find_actions(["S..G"]) == [Action.RIGHT] * 3


def test_goal_left_and_vertical_moves() -> None:
    assert find_actions(["G", ".", "S"]) == [Action.UP, Action.UP]
    assert find_actions(["G.S"]) == [Action.LEFT, Action.LEFT]
    assert find_actions(["S", ".", "G"]) == [Action.DOWN, Action.DOWN]


def test_adjacent_start_and_goal() -> None:
    assert find_actions(["SG"]) == [Action.RIGHT]


def test_detour_around_wall() -> None:
    grid = [
        "S.#..",
        "..#.G",
        ".....",
    ]
    actions = find_actions(grid)
    assert actions is not None
    assert_valid_plan(grid, actions)
    assert len(actions) == reference_distance(grid) == 7


def test_chooses_shorter_of_two_routes() -> None:
    grid = [
        "S.....",
        ".####.",
        "..G...",
    ]
    actions = find_actions(grid)
    assert actions is not None
    assert len(actions) == 4  # down, down, right, right
    assert_valid_plan(grid, actions)


def test_maze_with_single_winding_route() -> None:
    grid = [
        "S.#....",
        "#.#.##.",
        "#.#.#..",
        "#...#.#",
        "#####G#",
    ]
    actions = find_actions(grid)
    assert actions is not None
    assert_valid_plan(grid, actions)
    assert len(actions) == reference_distance(grid)


# ---------------------------------------------------------------------------
# Unreachable goals
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "grid",
    [
        ["S#G"],
        ["S.#", "##.", "..G"],
        ["#####", "#S#G#", "#####"],
        ["S..", "###", "..G"],
    ],
)
def test_no_path_returns_none(grid: list[str]) -> None:
    assert find_actions(grid) is None


def test_no_path_message() -> None:
    assert "No path" in describe_plan(find_actions(["S#G"]))


# ---------------------------------------------------------------------------
# Generated grids
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", range(60))
def test_random_grids_match_reference(seed: int) -> None:
    """On random grids, the result is valid and as short as an independent reference."""
    rng = random.Random(seed)
    grid = random_grid(rng, rng.randint(2, 12), rng.randint(2, 12), rng.choice([0.1, 0.3, 0.5]))
    actions = find_actions(grid)
    expected = reference_distance(grid)
    if expected is None:
        assert actions is None
    else:
        assert actions is not None
        assert_valid_plan(grid, actions)
        assert len(actions) == expected


def test_open_grid_length_is_manhattan_distance() -> None:
    height, width = 15, 20
    grid = [["."] * width for _ in range(height)]
    grid[0][0], grid[height - 1][width - 1] = "S", "G"
    actions = find_actions(["".join(row) for row in grid])
    assert actions is not None
    assert len(actions) == (height - 1) + (width - 1)


def test_large_grid_completes() -> None:
    grid = [["."] * 200 for _ in range(200)]
    grid[0][0], grid[199][199] = "S", "G"
    actions = find_actions(["".join(row) for row in grid])
    assert actions is not None
    assert len(actions) == 398


# ---------------------------------------------------------------------------
# Output and validation
# ---------------------------------------------------------------------------
def test_describe_plan_lists_actions() -> None:
    text = describe_plan([Action.RIGHT, Action.DOWN])
    assert "2 moves" in text
    assert "Right -> Down" in text


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
        find_actions(grid)
