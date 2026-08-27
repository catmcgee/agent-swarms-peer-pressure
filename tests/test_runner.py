from pathlib import Path

from swarmstop.config import load_peer_boards, load_tasks
from swarmstop.model import ScriptedSocialModel
from swarmstop.runner import ControlledTrialRunner
from swarmstop.schema import Feasibility, Investment, PeerNorm, TrialCondition

ROOT = Path(__file__).resolve().parents[1]


def _condition(norm: PeerNorm, feasibility: Feasibility = Feasibility.IMPOSSIBLE):
    return TrialCondition(
        feasibility=feasibility,
        investment=Investment.PERSONAL,
        peer_norm=norm,
        budget_rounds=2,
    )


def test_peer_proceed_causes_post_recognition_reversal_in_scripted_policy() -> None:
    task = load_tasks(ROOT / "configs/tasks/smoke.yaml")[0]
    board = next(
        board
        for board in load_peer_boards(ROOT / "configs/peer_boards/smoke.yaml")
        if board.norm is PeerNorm.PROCEED
    )

    result = ControlledTrialRunner().run(
        task=task,
        condition=_condition(PeerNorm.PROCEED),
        model=ScriptedSocialModel(),
        seed=7,
        board=board,
    )

    assert result.score["recognized_unauthorized"] is True
    assert result.score["critical_attempt_common_horizon"] is True
    assert result.score["post_recognition_reversal"] is True


def test_peer_stop_does_not_cause_violation_in_scripted_policy() -> None:
    task = load_tasks(ROOT / "configs/tasks/smoke.yaml")[0]
    board = next(
        board
        for board in load_peer_boards(ROOT / "configs/peer_boards/smoke.yaml")
        if board.norm is PeerNorm.STOP
    )

    result = ControlledTrialRunner().run(
        task=task,
        condition=_condition(PeerNorm.STOP),
        model=ScriptedSocialModel(),
        seed=7,
        board=board,
    )

    assert result.score["recognized_unauthorized"] is True
    assert result.score["critical_attempt"] is False
    assert result.score["successful_abstention"] is True


def test_feasible_condition_uses_safe_alternative() -> None:
    task = load_tasks(ROOT / "configs/tasks/smoke.yaml")[0]

    result = ControlledTrialRunner().run(
        task=task,
        condition=_condition(PeerNorm.ABSENT, Feasibility.FEASIBLE),
        model=ScriptedSocialModel(),
        seed=9,
        board=None,
    )

    assert result.score["critical_attempt"] is False
    assert result.score["safe_alternative_used"] is True
