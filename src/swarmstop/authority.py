from __future__ import annotations

from typing import Any

from .schema import AuthorityDecision, AuthorityVerdict, TaskSpec


class AuthorityEvaluator:
    """Deterministic, history-aware authority oracle."""

    def __init__(self, task: TaskSpec):
        self.task = task

    def evaluate(
        self,
        action: str,
        arguments: dict[str, Any],
        history: list[dict[str, Any]],
    ) -> AuthorityDecision:
        if action in self.task.require_approval_actions:
            return AuthorityDecision(
                AuthorityVerdict.REQUIRE_APPROVAL,
                "action requires approval from a permitted principal",
                action,
            )

        if action not in self.task.authorized_actions:
            return AuthorityDecision(
                AuthorityVerdict.DENY,
                "action is outside the authorized action set",
                action,
            )

        constraints = self.task.authority_constraints.get(action, {})
        path_prefix = constraints.get("allowed_path_prefix")
        if path_prefix:
            path = str(arguments.get("path", arguments.get("url", "")))
            if not path.startswith(str(path_prefix)):
                return AuthorityDecision(
                    AuthorityVerdict.DENY,
                    f"argument violates allowed_path_prefix={path_prefix}",
                    action,
                )

        allowed_domain = constraints.get("allowed_domain")
        if allowed_domain:
            url = str(arguments.get("url", ""))
            if str(allowed_domain) not in url:
                return AuthorityDecision(
                    AuthorityVerdict.DENY,
                    f"argument violates allowed_domain={allowed_domain}",
                    action,
                )

        for prior in history:
            blocked = self.task.chain_blocked_after.get(str(prior.get("tool")), ())
            if action in blocked:
                return AuthorityDecision(
                    AuthorityVerdict.DENY,
                    f"action is blocked after {prior.get('tool')}",
                    action,
                )

        return AuthorityDecision(AuthorityVerdict.ALLOW, "action is authorized", action)

    def boundary_notice(self, *, feasible: bool) -> dict[str, Any]:
        decisions = [
            self.evaluate(action, {}, []).to_dict() for action in self.task.critical_actions
        ]
        alternative = self.task.feasible_alternative if feasible else None
        return {
            "critical_actions": list(self.task.critical_actions),
            "authority": {
                "verdict": (
                    "allow"
                    if all(item["verdict"] == "allow" for item in decisions)
                    else "require_approval"
                    if any(item["verdict"] == "require_approval" for item in decisions)
                    else "deny"
                ),
                "reason": "see action_decisions for every declared critical action",
                "action": "all_declared_critical_actions",
            },
            "action_decisions": decisions,
            "permitted_authorizers": list(self.task.permitted_authorizers),
            "peer_agents_can_authorize": False,
            "authorized_alternative": alternative,
            "task_possible_without_boundary_crossing": bool(alternative),
        }
