"""A simple logical planner for the warehouse robot (Logic + Search = Planning).

A state is a set of propositions such as "At(Robot, A)". Each action has a
name, positive and negative preconditions, and positive and negative effects.

- Logic decides what is possible: an action is applicable in a state S if S
  satisfies its preconditions, and applying it removes its negative effects
  and then adds its positive effects.
- Search decides what to try: breadth-first search explores sequences of
  applicable actions until a state satisfying the goal is reached.

Run as a script to plan for the warehouse problem:

    python planner.py
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass

State = frozenset[str]

LOCATIONS = ("A", "B", "C")
# Directed connections between locations: the robot can go A <-> B and B <-> C.
CONNECTIONS = (("A", "B"), ("B", "A"), ("B", "C"), ("C", "B"))
HOLDING = "Holding(Package)"


@dataclass(frozen=True)
class Action:
    """A planning action, defined by its preconditions and effects.

    Attributes:
        name: A readable name such as "Move(A, B)".
        pos_pre: Propositions that must be true before the action.
        neg_pre: Propositions that must be false before the action.
        pos_eff: Propositions that become true after the action.
        neg_eff: Propositions that become false after the action.
    """

    name: str
    pos_pre: frozenset[str] = frozenset()
    neg_pre: frozenset[str] = frozenset()
    pos_eff: frozenset[str] = frozenset()
    neg_eff: frozenset[str] = frozenset()

    def is_applicable(self, state: State) -> bool:
        """Return True if the state satisfies every precondition (S |= Preconditions(a))."""
        return self.pos_pre <= state and self.neg_pre.isdisjoint(state)

    def unmet_preconditions(self, state: State) -> list[str]:
        """List the preconditions that the state does not satisfy."""
        missing = sorted(self.pos_pre - state)
        violated = sorted(f"not {p}" for p in self.neg_pre & state)
        return missing + violated

    def apply(self, state: State) -> State:
        """Return the successor state: remove the negative effects, then add the positive ones.

        Raises:
            ValueError: If the action is not applicable in the state.
        """
        if not self.is_applicable(state):
            raise ValueError(f"{self.name} is not applicable; unmet: {self.unmet_preconditions(state)}")
        return (state - self.neg_eff) | self.pos_eff


@dataclass(frozen=True)
class Plan:
    """A sequence of actions with the states it passes through.

    Attributes:
        actions: The actions a1, ..., an.
        states: The states S0, ..., Sn, so states[i + 1] results from actions[i] in states[i].
        expanded: The number of states expanded by the search that found the plan.
    """

    actions: list[Action]
    states: list[State]
    expanded: int = 0


def make_action(
    name: str,
    pre: Iterable[str] = (),
    neg_pre: Iterable[str] = (),
    add: Iterable[str] = (),
    delete: Iterable[str] = (),
) -> Action:
    """Build an Action from plain iterables of propositions."""
    return Action(name, frozenset(pre), frozenset(neg_pre), frozenset(add), frozenset(delete))


def at(thing: str, place: str) -> str:
    """The proposition At(thing, place)."""
    return f"At({thing}, {place})"


def warehouse_actions(
    locations: Iterable[str] = LOCATIONS,
    connections: Iterable[tuple[str, str]] = CONNECTIONS,
    include_pickup: bool = True,
) -> list[Action]:
    """The Move, PickUp and Drop actions of the warehouse scenario.

    Args:
        locations: The locations of the warehouse.
        connections: The (from, to) pairs between which the robot can move.
        include_pickup: If False, leave out the PickUp actions.
    """
    actions = [
        make_action(f"Move({a}, {b})", pre=[at("Robot", a)], add=[at("Robot", b)], delete=[at("Robot", a)])
        for a, b in connections
    ]
    for place in locations:
        if include_pickup:
            actions.append(
                make_action(
                    f"PickUp(Package, {place})",
                    pre=[at("Robot", place), at("Package", place)],
                    add=[HOLDING],
                    delete=[at("Package", place)],
                )
            )
        actions.append(
            make_action(
                f"Drop(Package, {place})",
                pre=[at("Robot", place), HOLDING],
                add=[at("Package", place)],
                delete=[HOLDING],
            )
        )
    return actions


INITIAL_STATE: State = frozenset({at("Robot", "A"), at("Package", "A")})
GOAL: frozenset[str] = frozenset({at("Package", "C")})


def find_plan(initial: State, goal: frozenset[str], actions: Iterable[Action]) -> Plan | None:
    """Find a shortest plan from the initial state to a state satisfying the goal, using BFS.

    Each search node is a state. The goal is satisfied by any state that
    contains all the goal propositions. Visited states are remembered, so the
    search terminates even when no plan exists.

    Returns:
        The plan, or None if no plan exists.
    """
    actions = list(actions)
    came_from: dict[State, tuple[State, Action] | None] = {initial: None}
    frontier: deque[State] = deque([initial])
    expanded = 0

    while frontier:
        state = frontier.popleft()
        if goal <= state:
            return _build_plan(came_from, state, expanded)
        expanded += 1
        for action in actions:
            if action.is_applicable(state):  # Logic: what is possible.
                successor = action.apply(state)
                if successor not in came_from:  # Search: what to try, without repeats.
                    came_from[successor] = (state, action)
                    frontier.append(successor)
    return None


def _build_plan(came_from: dict[State, tuple[State, Action] | None], goal_state: State, expanded: int) -> Plan:
    """Walk the parent links back from the goal state to the initial state."""
    actions: list[Action] = []
    states: list[State] = [goal_state]
    step = came_from[goal_state]
    while step is not None:
        previous, action = step
        actions.append(action)
        states.append(previous)
        step = came_from[previous]
    actions.reverse()
    states.reverse()
    return Plan(actions, states, expanded)


def verify_plan(initial: State, goal: frozenset[str], actions: Iterable[Action]) -> list[str]:
    """Independently execute a proposed action sequence and report every problem.

    This does not use the search. It checks, step by step, that each action's
    preconditions hold in the state where it is executed, and finally that the
    goal holds.

    Returns:
        A list of problems. An empty list means the plan is valid.
    """
    problems: list[str] = []
    state = initial
    for step, action in enumerate(actions, start=1):
        unmet = action.unmet_preconditions(state)
        if unmet:
            problems.append(f"step {step}, {action.name}: unmet preconditions {unmet}")
            return problems
        state = action.apply(state)
    if not goal <= state:
        problems.append(f"goal not reached; missing {sorted(goal - state)}")
    return problems


def format_state(state: State) -> str:
    """Format a state as a sorted, comma-separated list of facts."""
    return ", ".join(sorted(state))


def describe_plan(initial: State, plan: Plan | None) -> str:
    """Format a plan with the state reached after each action, or a failure message."""
    if plan is None:
        return "No plan found"
    lines = [f"Plan found ({len(plan.actions)} actions, {plan.expanded} states expanded):"]
    lines.append(f"  S0: {format_state(initial)}")
    for i, (action, state) in enumerate(zip(plan.actions, plan.states[1:]), start=1):
        lines.append(f"  {i}. {action.name}")
        lines.append(f"  S{i}: {format_state(state)}")
    return "\n".join(lines)


def main() -> None:
    """Plan for the warehouse problem and print the plan and the states reached."""
    plan = find_plan(INITIAL_STATE, GOAL, warehouse_actions())
    print(describe_plan(INITIAL_STATE, plan))
    if plan is not None:
        problems = verify_plan(INITIAL_STATE, GOAL, plan.actions)
        print("Independent verification:", "valid" if not problems else problems)


if __name__ == "__main__":
    main()
