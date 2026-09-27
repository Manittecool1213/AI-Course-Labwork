"""A* and BFS search agents for the warehouse robot navigation problem.

The warehouse is an ASCII map:

    S  start position (exactly one)
    G  goal position (exactly one)
    #  obstacle (cannot be entered)
    .  free cell

The robot moves Up, Down, Left or Right, one cell at a time, and every move
costs 1. Both agents report whether a solution was found, the path, and the
number of states expanded.

Run as a script to solve the default warehouse with A* and BFS:

    python warehouse_search.py
"""

from __future__ import annotations

import heapq
import itertools
import math
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

# A grid position as (row, column), with (0, 0) in the top-left corner.
Position = tuple[int, int]
Heuristic = Callable[[Position, Position], float]

START = "S"
GOAL = "G"
OBSTACLE = "#"
FREE = "."
VALID_CELLS = {START, GOAL, OBSTACLE, FREE}
STEP_COST = 1

# The warehouse from the lab sheet.
WAREHOUSE_MAP: list[str] = [
    "#################",
    "#S....#.........#",
    "#.###.#.#######.#",
    "#...#.#.......#.#",
    "###.#.#######.#.#",
    "#...#.........#.#",
    "#.###########.#.#",
    "#.............#G#",
    "#################",
]


class Action(Enum):
    """The moves available to the robot, each with its (row, column) offset."""

    UP = (-1, 0)
    DOWN = (1, 0)
    LEFT = (0, -1)
    RIGHT = (0, 1)


@dataclass(frozen=True)
class SearchResult:
    """What a search agent reports when it terminates.

    Attributes:
        found: Whether a path from S to G was found.
        path: The positions from S to G inclusive (empty if no path exists).
        expanded: The number of states expanded. A state counts each time its
            successors are generated; the goal state is never expanded.
        expansion_order: The states in the order they were expanded.
    """

    found: bool
    path: list[Position]
    expanded: int
    expansion_order: list[Position] = field(default_factory=list)

    @property
    def length(self) -> int | None:
        """The path length in moves (each move costs 1), or None if no path was found."""
        return len(self.path) - 1 if self.found else None

    @property
    def actions(self) -> list[Action]:
        """The actions the robot takes along the path."""
        offsets = {action.value: action for action in Action}
        return [
            offsets[(nr - r, nc - c)] for (r, c), (nr, nc) in zip(self.path, self.path[1:])
        ]


class WarehouseProblem:
    """The warehouse as a search problem P = (S, A, T, s0, G, c).

    A state is the robot's (row, column) position. The map never changes, so
    the position is all that is needed to describe the state.
    """

    def __init__(self, rows: list[str]) -> None:
        """Validate the map and locate the start and goal.

        Raises:
            ValueError: If the map is empty, ragged, contains characters other
                than S, G, # and ., or does not have exactly one S and one G.
        """
        if not rows or not rows[0]:
            raise ValueError("The warehouse map is empty.")
        width = len(rows[0])
        if any(len(row) != width for row in rows):
            raise ValueError("All rows of the warehouse map must have the same length.")

        starts: list[Position] = []
        goals: list[Position] = []
        for r, row in enumerate(rows):
            for c, cell in enumerate(row):
                if cell not in VALID_CELLS:
                    raise ValueError(f"Invalid character {cell!r} at row {r}, column {c}.")
                if cell == START:
                    starts.append((r, c))
                elif cell == GOAL:
                    goals.append((r, c))
        if len(starts) != 1:
            raise ValueError(f"Expected exactly one start '{START}', found {len(starts)}.")
        if len(goals) != 1:
            raise ValueError(f"Expected exactly one goal '{GOAL}', found {len(goals)}.")

        self.grid = list(rows)
        self.start = starts[0]
        self.goal = goals[0]

    def is_free(self, position: Position) -> bool:
        """Return True if the position is inside the map and not an obstacle."""
        r, c = position
        return 0 <= r < len(self.grid) and 0 <= c < len(self.grid[0]) and self.grid[r][c] != OBSTACLE

    def successors(self, state: Position) -> list[tuple[Action, Position]]:
        """Return each valid action from a state with the state it leads to."""
        result = []
        for action in Action:
            dr, dc = action.value
            neighbour = (state[0] + dr, state[1] + dc)
            if self.is_free(neighbour):
                result.append((action, neighbour))
        return result

    def is_goal(self, state: Position) -> bool:
        """Return True if the state is the goal."""
        return state == self.goal


def manhattan(state: Position, goal: Position) -> float:
    """h(n) = |x - x_G| + |y - y_G|, the cost if no obstacles were in the way."""
    return abs(state[0] - goal[0]) + abs(state[1] - goal[1])


def euclidean(state: Position, goal: Position) -> float:
    """h(n) = straight-line distance to the goal."""
    return math.hypot(state[0] - goal[0], state[1] - goal[1])


def zero(state: Position, goal: Position) -> float:
    """h(n) = 0, which turns A* into uniform-cost search."""
    return 0.0


def scaled(heuristic: Heuristic, factor: float) -> Heuristic:
    """Return a heuristic that multiplies another heuristic by a constant factor."""

    def scaled_heuristic(state: Position, goal: Position) -> float:
        return factor * heuristic(state, goal)

    return scaled_heuristic


def reconstruct_path(came_from: dict[Position, Position | None], goal: Position) -> list[Position]:
    """Walk the parent links back from the goal to the start."""
    path: list[Position] = []
    node: Position | None = goal
    while node is not None:
        path.append(node)
        node = came_from[node]
    path.reverse()
    return path


def a_star(problem: WarehouseProblem, heuristic: Heuristic = manhattan) -> SearchResult:
    """Search with A*, always expanding the frontier state with the smallest f = g + h.

    The frontier is a priority queue of (f, tie-breaker, g, state) entries. A
    state can be pushed again if a cheaper route to it is found, so entries
    whose g is worse than the best known g are skipped when they are popped.
    Ties on f are broken first-in first-out.

    Returns:
        The search result. With an admissible heuristic the path is optimal.
    """
    tie_breaker = itertools.count()
    best_g: dict[Position, float] = {problem.start: 0}
    came_from: dict[Position, Position | None] = {problem.start: None}
    frontier: list[tuple[float, int, float, Position]] = [
        (heuristic(problem.start, problem.goal), next(tie_breaker), 0, problem.start)
    ]
    expansion_order: list[Position] = []

    while frontier:
        _, _, g, state = heapq.heappop(frontier)
        if g > best_g[state]:
            continue  # A cheaper route to this state was found after this entry was pushed.
        if problem.is_goal(state):
            path = reconstruct_path(came_from, state)
            return SearchResult(True, path, len(expansion_order), expansion_order)

        expansion_order.append(state)
        for _, neighbour in problem.successors(state):
            new_g = g + STEP_COST
            if neighbour not in best_g or new_g < best_g[neighbour]:
                best_g[neighbour] = new_g
                came_from[neighbour] = state
                f = new_g + heuristic(neighbour, problem.goal)
                heapq.heappush(frontier, (f, next(tie_breaker), new_g, neighbour))

    return SearchResult(False, [], len(expansion_order), expansion_order)


def bfs(problem: WarehouseProblem) -> SearchResult:
    """Search breadth-first, expanding states in order of their distance from S.

    The goal test is done when a state is removed from the frontier, exactly as
    in `a_star`, so the two expansion counts are comparable.
    """
    came_from: dict[Position, Position | None] = {problem.start: None}
    frontier: deque[Position] = deque([problem.start])
    expansion_order: list[Position] = []

    while frontier:
        state = frontier.popleft()
        if problem.is_goal(state):
            path = reconstruct_path(came_from, state)
            return SearchResult(True, path, len(expansion_order), expansion_order)

        expansion_order.append(state)
        for _, neighbour in problem.successors(state):
            if neighbour not in came_from:
                came_from[neighbour] = state
                frontier.append(neighbour)

    return SearchResult(False, [], len(expansion_order), expansion_order)


def render_path(problem: WarehouseProblem, result: SearchResult) -> str:
    """Draw the map with the path marked by '*'."""
    cells = [list(row) for row in problem.grid]
    for r, c in result.path:
        if cells[r][c] == FREE:
            cells[r][c] = "*"
    return "\n".join("".join(row) for row in cells)


def describe(name: str, result: SearchResult) -> str:
    """Format a search result for display."""
    if not result.found:
        return f"{name}: no solution found ({result.expanded} states expanded)"
    return f"{name}: solution found, length {result.length}, {result.expanded} states expanded"


def main() -> None:
    """Solve the default warehouse with A* and BFS and print both results."""
    problem = WarehouseProblem(WAREHOUSE_MAP)
    astar_result = a_star(problem)
    print(describe("A*", astar_result))
    print(describe("BFS", bfs(problem)))
    print("\nA* path (actions):")
    print(" -> ".join(action.name.capitalize() for action in astar_result.actions))
    print("\nA* path on the map:")
    print(render_path(problem, astar_result))


if __name__ == "__main__":
    main()
