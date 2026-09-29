"""Experiments for the logic lab: applicability, the hand-made plan, Tests A-C and plan checking.

Run as a script:

    python logic_experiments.py
"""

from __future__ import annotations

from planner import (
    GOAL,
    INITIAL_STATE,
    Action,
    at,
    describe_plan,
    find_plan,
    format_state,
    make_action,
    verify_plan,
    warehouse_actions,
)

ACTIONS = {action.name: action for action in warehouse_actions()}


def show_transitions(names: list[str]) -> None:
    """Print, for each action, the preconditions, whether they hold, and the resulting state."""
    state = INITIAL_STATE
    print(f"S0: {format_state(state)}")
    for i, name in enumerate(names, start=1):
        action = ACTIONS[name]
        print(f"{i}. {name}: preconditions {sorted(action.pos_pre)}, unmet {action.unmet_preconditions(state)}")
        if not action.is_applicable(state):
            print("   -> not applicable, stop")
            return
        state = action.apply(state)
        print(f"S{i}: {format_state(state)}")
    print(f"Goal {sorted(GOAL)} satisfied: {GOAL <= state}")


def report(title: str, initial: frozenset[str], goal: frozenset[str], actions: list[Action]) -> None:
    """Run the planner and independently verify the plan it returns."""
    plan = find_plan(initial, goal, actions)
    print(f"\n{title}\ninitial: {sorted(initial)}\ngoal: {sorted(goal)}")
    print(describe_plan(initial, plan))
    if plan is not None:
        print("verification problems:", verify_plan(initial, goal, plan.actions))


def main() -> None:
    print("=== Task 0: applicability in the initial state ===")
    for name in ["PickUp(Package, A)", "Drop(Package, C)", "Move(A, B)", "Move(B, C)"]:
        action = ACTIONS[name]
        print(f"{name}: applicable={action.is_applicable(INITIAL_STATE)}, unmet={action.unmet_preconditions(INITIAL_STATE)}")

    print("\n=== Task 1: hand-made plan, state after every action ===")
    show_transitions(["PickUp(Package, A)", "Move(A, B)", "Move(B, C)", "Drop(Package, C)"])

    print("\n=== Task 1: the example sequence from the lab sheet ===")
    show_transitions(["Move(A, B)", "PickUp(Package, B)", "Move(B, C)", "Drop(Package, C)"])

    print("\n=== Task 3 ===")
    report("Test A: solvable", INITIAL_STATE, GOAL, warehouse_actions())
    report("Test B: PickUp removed", INITIAL_STATE, GOAL, warehouse_actions(include_pickup=False))

    moves_only = [a for a in warehouse_actions() if a.name.startswith("Move")]
    report("Test C1: only robot moves, goal At(Robot, C)", INITIAL_STATE, frozenset({at("Robot", "C")}), moves_only)
    report("Test C2: only robot moves, goal At(Package, C)", INITIAL_STATE, GOAL, moves_only)
    shortcut = make_action("Shortcut(A, C)", pre=[at("Robot", "A")], add=[at("Robot", "C")], delete=[at("Robot", "A")])
    report("Test C3: extra robot-only action Shortcut(A, C)", INITIAL_STATE, GOAL, warehouse_actions() + [shortcut])

    print("\n=== Task 5: an invalid but plausible plan is caught by independent execution ===")
    names = ["Move(A, B)", "PickUp(Package, B)", "Move(B, C)", "Drop(Package, C)"]
    print(verify_plan(INITIAL_STATE, GOAL, [ACTIONS[n] for n in names]))


if __name__ == "__main__":
    main()
