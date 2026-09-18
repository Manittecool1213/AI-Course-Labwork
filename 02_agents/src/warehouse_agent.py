"""Goal-based agent for the warehouse navigation problem.

The warehouse is a rectangular grid of characters:

    S  start position (exactly one)
    G  goal position (exactly one)
    #  obstacle (cannot be entered)
    .  free space

The agent may move Up, Down, Left or Right, one square at a time. Given a
map, it searches for a collision-free route from S to G and reports it as the
sequence of actions the vehicle must take.

Run as a script to solve the default warehouse:

    python warehouse_agent.py
"""

from __future__ import annotations

from collections import deque
from enum import Enum

# A grid position as (row, column), with (0, 0) in the top-left corner.
Position = tuple[int, int]
Grid = list[str]

START = "S"
GOAL = "G"
OBSTACLE = "#"
FREE = "."
VALID_CELLS = {START, GOAL, OBSTACLE, FREE}

# The warehouse from the lab sheet.
WAREHOUSE_MAP: Grid = [
    "#####################",
    "#S....#............G#",
    "#.##....##########..#",
    "#....##.............#",
    "#.######.###.#.###..#",
    "#........#..........#",
    "#####################",
]


class Action(Enum):
    """The moves available to the agent, each with its (row, column) offset."""

    UP = (-1, 0)
    DOWN = (1, 0)
    LEFT = (0, -1)
    RIGHT = (0, 1)


def parse_grid(rows: list[str]) -> tuple[Grid, Position, Position]:
    """Validate a warehouse map and locate the start and goal.

    Args:
        rows: The map, one string per row.

    Returns:
        The grid, the start position and the goal position.

    Raises:
        ValueError: If the map is empty, ragged, contains characters other than
            S, G, # and ., or does not have exactly one S and one G.
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
    return list(rows), starts[0], goals[0]


def is_free(grid: Grid, position: Position) -> bool:
    """Return True if the position is inside the grid and not an obstacle."""
    r, c = position
    return 0 <= r < len(grid) and 0 <= c < len(grid[0]) and grid[r][c] != OBSTACLE


def move(position: Position, action: Action) -> Position:
    """Return the position reached by applying an action (no bounds checking)."""
    dr, dc = action.value
    return position[0] + dr, position[1] + dc


# ---------------------------------------------------------------------------
# Choice of search algorithm: Breadth-First Search (BFS)
#
# Every move costs the same (one grid square), so the best path is simply the
# one with the fewest moves. BFS expands positions in order of their distance
# from S, so the first time it reaches G it has found a shortest path. It is
# also complete: on a finite grid it either finds a path or exhausts every
# reachable cell, which lets it report reliably that no path exists.
#
# The alternatives are less suitable here:
#   - DFS is simple and light on memory, but its path is not guaranteed to be
#     shortest and can wander through most of the warehouse.
#   - Greedy best-first search follows a heuristic towards G and can be led
#     into dead ends by shelves, again without guaranteeing the shortest path.
#   - Dijkstra's algorithm and A* handle varying costs or use heuristics to
#     expand fewer cells. With uniform costs Dijkstra reduces to BFS with extra
#     overhead, and A* only pays off on much larger maps than this one.
# ---------------------------------------------------------------------------
def find_actions(rows: list[str]) -> list[Action] | None:
    """Find a shortest collision-free sequence of actions from S to G using BFS.

    Args:
        rows: The warehouse map, one string per row.

    Returns:
        The actions to take, in order, or None if G is unreachable from S.

    Raises:
        ValueError: If the map is invalid (see `parse_grid`).
    """
    grid, start, goal = parse_grid(rows)

    # For each visited position, the position and action we arrived by.
    # Doubles as the visited set, and lets us rebuild the path at the end.
    came_from: dict[Position, tuple[Position, Action] | None] = {start: None}
    frontier: deque[Position] = deque([start])

    while frontier:
        current = frontier.popleft()
        if current == goal:
            break
        for action in Action:
            neighbour = move(current, action)
            if is_free(grid, neighbour) and neighbour not in came_from:
                came_from[neighbour] = (current, action)
                frontier.append(neighbour)
    else:
        return None

    # Walk back from the goal to the start to recover the actions.
    actions: list[Action] = []
    step = came_from[goal]
    while step is not None:
        previous, action = step
        actions.append(action)
        step = came_from[previous]
    actions.reverse()
    return actions


def describe_plan(actions: list[Action] | None) -> str:
    """Format the agent's plan for display, or a message if there is none."""
    if actions is None:
        return "No path exists from S to G."
    names = " -> ".join(action.name.capitalize() for action in actions)
    return f"Path found in {len(actions)} moves:\n{names}"


def main() -> None:
    """Solve the default warehouse and print the agent's actions."""
    print(describe_plan(find_actions(WAREHOUSE_MAP)))


if __name__ == "__main__":
    main()
