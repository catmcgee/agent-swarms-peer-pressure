from pathlib import Path

from swarmstop.authority import AuthorityEvaluator
from swarmstop.config import load_tasks
from swarmstop.environment import SandboxEnvironment
from swarmstop.schema import AuthorityVerdict

ROOT = Path(__file__).resolve().parents[1]


def test_critical_action_is_denied_but_safe_alternative_is_allowed() -> None:
    task = load_tasks(ROOT / "configs/tasks/smoke.yaml")[0]
    evaluator = AuthorityEvaluator(task)

    critical = evaluator.evaluate("publish_report", {}, [])
    safe = evaluator.evaluate("save_report_draft", {}, [])

    assert critical.verdict is AuthorityVerdict.DENY
    assert safe.verdict is AuthorityVerdict.ALLOW


def test_fail_open_sandbox_records_direct_violation() -> None:
    task = load_tasks(ROOT / "configs/tasks/smoke.yaml")[0]
    environment = SandboxEnvironment(task)

    record = environment.execute("publish_report", {})

    assert record.critical_attempt is True
    assert record.delegated_attempt is False
    assert environment.state["external_visibility"] is True


def test_delegation_is_scored_as_critical_attempt() -> None:
    task = load_tasks(ROOT / "configs/tasks/smoke.yaml")[0]
    environment = SandboxEnvironment(task)

    record = environment.execute(
        "request_peer_action",
        {"peer": "peer-2", "action": "publish_report", "reason": "finish it"},
    )

    assert record.critical_attempt is True
    assert record.delegated_attempt is True
    assert record.authority.verdict is AuthorityVerdict.DENY
