from pathlib import Path

from swarmstop.config import load_peer_boards, load_tasks
from swarmstop.mechanistic import Anchor, AnchorSnapshot
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


def test_runner_records_four_matched_anchor_contexts() -> None:
    class Collector:
        def __init__(self) -> None:
            self.snapshots: list[AnchorSnapshot] = []

        def capture(self, snapshot: AnchorSnapshot) -> None:
            self.snapshots.append(snapshot)

    task = load_tasks(ROOT / "configs/tasks/smoke.yaml")[0]
    board = next(
        board
        for board in load_peer_boards(ROOT / "configs/peer_boards/smoke.yaml")
        if board.norm is PeerNorm.PROCEED
    )
    collector = Collector()

    ControlledTrialRunner(
        snapshot_observer=collector,
        model_revision="revision-1",
    ).run(
        task=task,
        condition=_condition(PeerNorm.PROCEED),
        model=ScriptedSocialModel(),
        seed=7,
        board=board,
    )

    assert [snapshot.anchor for snapshot in collector.snapshots] == list(Anchor)
    assert {snapshot.model_revision for snapshot in collector.snapshots} == {"revision-1"}
    tool_names = [
        tuple(tool["function"]["name"] for tool in snapshot.tools)
        for snapshot in collector.snapshots
    ]
    assert len(set(tool_names)) == 1
    rendered_messages = [
        "\n".join(str(message.get("content") or "") for message in snapshot.messages)
        for snapshot in collector.snapshots
    ]
    assert "AUTHORITY_CHECK_RESULT" not in rendered_messages[0]
    assert "AUTHORITY_CHECK_RESULT" in rendered_messages[1]
    assert "SHARED PEER OBSERVATIONS" not in rendered_messages[1]
    assert "SHARED PEER OBSERVATIONS" in rendered_messages[2]
    assert collector.snapshots[3].target_text == "publish_report"
