"""Tests for the logical planner and the Prolog verifier. Run with: pytest"""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from planner import (
    GOAL,
    INITIAL_STATE,
    Action,
    State,
    at,
    describe_plan,
    find_plan,
    make_action,
    verify_plan,
    warehouse_actions,
)
from prolog_verifier import moves_in, prolog_available, query, verify_moves

ACTIONS = {action.name: action for action in warehouse_actions()}
NEEDS_PROLOG = pytest.mark.skipif(not prolog_available(), reason="SWI-Prolog is not installed")


def run(names: list[str], state: State = INITIAL_STATE) -> State:
    """Apply the named warehouse actions in order."""
    for name in names:
        state = ACTIONS[name].apply(state)
    return state


# ---------------------------------------------------------------------------
# Action semantics (Task 0)
# ---------------------------------------------------------------------------
def test_pickup_is_applicable_initially() -> None:
    assert ACTIONS["PickUp(Package, A)"].is_applicable(INITIAL_STATE)


def test_drop_is_not_applicable_initially() -> None:
    drop = ACTIONS["Drop(Package, C)"]
    assert not drop.is_applicable(INITIAL_STATE)
    assert drop.unmet_preconditions(INITIAL_STATE) == ["At(Robot, C)", "Holding(Package)"]


def test_action_in_the_list_is_not_applicable_unless_preconditions_hold() -> None:
    assert not ACTIONS["Move(B, C)"].is_applicable(INITIAL_STATE)
    assert not ACTIONS["PickUp(Package, B)"].is_applicable(INITIAL_STATE)


def test_apply_removes_negative_effects_and_adds_positive_effects() -> None:
    state = ACTIONS["Move(A, B)"].apply(INITIAL_STATE)
    assert state == {at("Robot", "B"), at("Package", "A")}
    assert INITIAL_STATE == {at("Robot", "A"), at("Package", "A")}  # The original state is unchanged.


def test_apply_raises_when_not_applicable() -> None:
    with pytest.raises(ValueError):
        ACTIONS["Drop(Package, C)"].apply(INITIAL_STATE)


def test_negative_preconditions_and_effects() -> None:
    lock = make_action("Lock", pre=["Closed"], neg_pre=["Locked"], add=["Locked"])
    unlock = make_action("Unlock", pre=["Locked"], delete=["Locked"])
    assert lock.is_applicable(frozenset({"Closed"}))
    assert not lock.is_applicable(frozenset({"Closed", "Locked"}))
    assert unlock.apply(frozenset({"Locked", "Closed"})) == {"Closed"}


def test_positive_effect_wins_if_a_fact_is_both_deleted_and_added() -> None:
    action = make_action("Refresh", add=["X"], delete=["X"])
    assert action.apply(frozenset()) == {"X"}


# ---------------------------------------------------------------------------
# The three tests required by the lab
# ---------------------------------------------------------------------------
def test_a_solvable_problem() -> None:
    plan = find_plan(INITIAL_STATE, GOAL, warehouse_actions())
    assert plan is not None
    assert [a.name for a in plan.actions] == [
        "PickUp(Package, A)",
        "Move(A, B)",
        "Move(B, C)",
        "Drop(Package, C)",
    ]
    assert plan.states[0] == INITIAL_STATE
    assert GOAL <= plan.states[-1]
    assert verify_plan(INITIAL_STATE, GOAL, plan.actions) == []


def test_a_states_in_the_plan_match_independent_execution() -> None:
    plan = find_plan(INITIAL_STATE, GOAL, warehouse_actions())
    assert plan is not None
    state = INITIAL_STATE
    for action, reported in zip(plan.actions, plan.states[1:]):
        state = action.apply(state)
        assert state == reported


def test_b_impossible_problem_when_pickup_is_removed() -> None:
    assert find_plan(INITIAL_STATE, GOAL, warehouse_actions(include_pickup=False)) is None
    assert describe_plan(INITIAL_STATE, None) == "No plan found"


def test_c_moving_the_robot_is_not_moving_the_package() -> None:
    moves_only = [a for a in warehouse_actions() if a.name.startswith("Move")]
    # The robot can reach C, but the package never gets there.
    assert find_plan(INITIAL_STATE, frozenset({at("Robot", "C")}), moves_only) is not None
    assert find_plan(INITIAL_STATE, GOAL, moves_only) is None
    assert at("Package", "C") not in run(["Move(A, B)", "Move(B, C)"])


def test_c_extra_action_that_moves_only_the_robot() -> None:
    shortcut = make_action("Shortcut(A, C)", pre=[at("Robot", "A")], add=[at("Robot", "C")], delete=[at("Robot", "A")])
    with_shortcut = warehouse_actions() + [shortcut]
    assert find_plan(INITIAL_STATE, GOAL, warehouse_actions(include_pickup=False) + [shortcut]) is None
    plan = find_plan(INITIAL_STATE, GOAL, with_shortcut)
    assert plan is not None
    assert [a.name for a in plan.actions] == ["PickUp(Package, A)", "Shortcut(A, C)", "Drop(Package, C)"]
    assert verify_plan(INITIAL_STATE, GOAL, plan.actions) == []


# ---------------------------------------------------------------------------
# Hand-made plans and the verifier
# ---------------------------------------------------------------------------
def test_verify_accepts_the_hand_made_plan() -> None:
    names = ["PickUp(Package, A)", "Move(A, B)", "Move(B, C)", "Drop(Package, C)"]
    assert verify_plan(INITIAL_STATE, GOAL, [ACTIONS[n] for n in names]) == []


def test_verify_rejects_the_example_sequence_from_the_lab_sheet() -> None:
    """The package is at A, so PickUp(Package, B) is not applicable at step 2."""
    names = ["Move(A, B)", "PickUp(Package, B)", "Move(B, C)", "Drop(Package, C)"]
    problems = verify_plan(INITIAL_STATE, GOAL, [ACTIONS[n] for n in names])
    assert problems == ["step 2, PickUp(Package, B): unmet preconditions ['At(Package, B)']"]


def test_verify_rejects_a_plan_that_never_reaches_the_goal() -> None:
    problems = verify_plan(INITIAL_STATE, GOAL, [ACTIONS["Move(A, B)"], ACTIONS["Move(B, C)"]])
    assert problems == ["goal not reached; missing ['At(Package, C)']"]


@pytest.mark.parametrize("dropped", range(4))
def test_verify_rejects_a_plan_with_one_action_removed(dropped: int) -> None:
    plan = find_plan(INITIAL_STATE, GOAL, warehouse_actions())
    assert plan is not None
    mutated = plan.actions[:dropped] + plan.actions[dropped + 1 :]
    assert verify_plan(INITIAL_STATE, GOAL, mutated) != []


def test_goal_already_true_gives_an_empty_plan() -> None:
    plan = find_plan(INITIAL_STATE, frozenset({at("Robot", "A")}), warehouse_actions())
    assert plan is not None
    assert plan.actions == []
    assert plan.states == [INITIAL_STATE]


# ---------------------------------------------------------------------------
# Generated problems, checked against an independent exhaustive search
# ---------------------------------------------------------------------------
def random_problem(rng: random.Random) -> tuple[State, frozenset[str], list[Action]]:
    """A random small propositional planning problem whose goal is not true initially."""
    props = [f"p{i}" for i in range(rng.randint(4, 7))]
    pick = lambda k: rng.sample(props, rng.randint(0, k))  # noqa: E731
    actions = [
        make_action(f"a{i}", pre=pick(2), neg_pre=pick(1), add=pick(2), delete=pick(2))
        for i in range(rng.randint(4, 12))
    ]
    initial = frozenset(rng.sample(props, rng.randint(0, 2)))
    candidates = [p for p in props if p not in initial]
    goal = frozenset(rng.sample(candidates, rng.randint(1, min(3, len(candidates)))))
    return initial, goal, actions


def shortest_plan_length(initial: State, goal: frozenset[str], actions: list[Action]) -> int | None:
    """Shortest plan length by expanding whole layers of reachable states (no parent links)."""
    layer, seen = {initial}, {initial}
    for depth in range(2 ** 6 + 1):
        if any(goal <= s for s in layer):
            return depth
        layer = {a.apply(s) for s in layer for a in actions if a.is_applicable(s)} - seen
        if not layer:
            return None
        seen |= layer
    return None


@pytest.mark.parametrize("seed", range(80))
def test_random_problems_match_exhaustive_search(seed: int) -> None:
    initial, goal, actions = random_problem(random.Random(seed))
    plan = find_plan(initial, goal, actions)
    expected = shortest_plan_length(initial, goal, actions)
    if expected is None:
        assert plan is None
    else:
        assert plan is not None
        assert len(plan.actions) == expected
        assert verify_plan(initial, goal, plan.actions) == []


# ---------------------------------------------------------------------------
# Prolog as an independent verifier (skipped if SWI-Prolog is not installed)
# ---------------------------------------------------------------------------
@NEEDS_PROLOG
@pytest.mark.parametrize(
    ("goal", "expected"),
    [
        ("can_move(a,b)", True),
        ("can_move(a,c)", False),
        ("valid_move(a,b)", True),
        ("valid_move(b,c)", True),
        ("valid_move(a,c)", False),
        ("valid_route([a,b,c])", True),
        ("valid_route([a,c])", False),
    ],
)
def test_prolog_queries(goal: str, expected: bool) -> None:
    assert query(goal) is expected


@NEEDS_PROLOG
def test_prolog_supports_every_move_in_the_python_plan() -> None:
    plan = find_plan(INITIAL_STATE, GOAL, warehouse_actions())
    assert plan is not None
    moves = moves_in(plan.actions)
    assert moves == [("a", "b"), ("b", "c")]
    assert all(verify_moves(moves))


@NEEDS_PROLOG
def test_prolog_rejects_a_move_the_knowledge_does_not_support() -> None:
    assert verify_moves([("a", "c")]) == [False]


@NEEDS_PROLOG
def test_prolog_chained_inference() -> None:
    road = Path(__file__).with_name("road.pl")
    assert query("reduce_speed", road)
    assert query("slippery", road)
