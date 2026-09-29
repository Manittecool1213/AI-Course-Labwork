"""Use Prolog as an independent verifier for the moves in a plan (Tasks 6 and 7).

The Python planner generates a candidate plan. Each Move(X, Y) in it is then
checked against the Prolog knowledge base in planner.pl, which knows only which
locations are connected. Requires SWI-Prolog (`swipl`) to be installed.

Run as a script:

    python prolog_verifier.py
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

from planner import GOAL, INITIAL_STATE, Action, find_plan, warehouse_actions

KNOWLEDGE_BASE = Path(__file__).with_name("planner.pl")
MOVE_NAME = re.compile(r"Move\((\w+), (\w+)\)")


def prolog_available() -> bool:
    """Return True if SWI-Prolog is installed."""
    return shutil.which("swipl") is not None


def query(goal: str, program: Path = KNOWLEDGE_BASE) -> bool:
    """Run a Prolog goal against a program file and return whether it succeeds.

    Args:
        goal: A Prolog goal without the trailing full stop, e.g. "valid_move(a,b)".
        program: The Prolog file to load.
    """
    command = ["swipl", "-q", "-s", str(program), "-g", f"({goal} -> halt(0) ; halt(1))"]
    return subprocess.run(command, capture_output=True, check=False).returncode == 0


def moves_in(actions: list[Action]) -> list[tuple[str, str]]:
    """Extract the (from, to) pairs of the Move actions, as lower-case Prolog atoms."""
    moves = []
    for action in actions:
        match = MOVE_NAME.fullmatch(action.name)
        if match:
            moves.append((match.group(1).lower(), match.group(2).lower()))
    return moves


def verify_moves(moves: list[tuple[str, str]]) -> list[bool]:
    """Ask Prolog whether each proposed move is supported by the warehouse knowledge."""
    return [query(f"valid_move({a},{b})") for a, b in moves]


def main() -> None:
    """Verify the moves in the planner's plan, and one unsupported move."""
    if not prolog_available():
        print("SWI-Prolog (swipl) is not installed.")
        return
    plan = find_plan(INITIAL_STATE, GOAL, warehouse_actions())
    assert plan is not None
    moves = moves_in(plan.actions)
    print("Moves in the Python plan:", moves)
    print("Prolog verdicts:", dict(zip(moves, verify_moves(moves))))
    print("Proposed Move(a, c):", "supported" if query("valid_move(a,c)") else "not supported")


if __name__ == "__main__":
    main()
