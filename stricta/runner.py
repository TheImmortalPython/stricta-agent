import copy
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional

from .critic import ShadowCritic
from .generator import HuggingFacePrimaryGenerator
from .roles import derive_active_roles
from .schemas import (
    ActionProposal,
    CriticVerdict,
    KarakaTaskNode,
    NodeStatus,
    RoleProgression,
    StepSnapshot,
)


class StrictaAgentRunner:
    def __init__(
        self,
        generator: HuggingFacePrimaryGenerator,
        critic: ShadowCritic,
        tool_executor: Optional[Callable[[ActionProposal, Dict[str, Any]], Any]] = None,
        max_retries: int = 4,
        runtime_state: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.generator = generator
        self.critic = critic
        self.tool_executor = tool_executor
        self.max_retries = max_retries
        self.task_graph: Dict[str, KarakaTaskNode] = {}
        self.runtime_state: Dict[str, Any] = runtime_state if runtime_state is not None else {
            "ledger": [],
            "artifacts": {},
        }
        self.audit_queue: list[Dict[str, Any]] = []

    def register_node(self, node: KarakaTaskNode) -> None:
        self.task_graph[node.node_id] = node

    def _dependencies_satisfied(self, node: KarakaTaskNode) -> bool:
        for dep_id in node.dependencies:
            dep_node = self.task_graph.get(dep_id)
            if dep_node is None or dep_node.status != NodeStatus.VERIFIED:
                return False
        return True

    def _advance_phase(self, node: KarakaTaskNode) -> None:
        order = [
            RoleProgression.INIT,
            RoleProgression.TARGET,
            RoleProgression.BOUNDARY,
            RoleProgression.COMPLETE,
        ]
        idx = order.index(node.role_phase)
        if idx + 1 < len(order):
            node.role_phase = order[idx + 1]

    def _execute_tool(self, proposal: ActionProposal) -> Any:
        if self.tool_executor is None:
            return {
                "ok": True,
                "tool_name": proposal.tool_name,
                "expected_state_change": proposal.expected_state_change,
            }
        return self.tool_executor(proposal, self.runtime_state)

    def _record_ledger_entry(self, node: KarakaTaskNode, proposal: ActionProposal, result: Any) -> None:
        ledger = self.runtime_state.setdefault("ledger", [])
        entry = {
            "id": len(ledger) + 1,
            "ts": datetime.now(timezone.utc).isoformat(),
            "node_id": node.node_id,
            "phase": node.role_phase.value,
            "tool_name": proposal.tool_name,
            "arguments": copy.deepcopy(proposal.arguments),
            "expected_state_change": proposal.expected_state_change,
            "result": copy.deepcopy(result),
        }
        ledger.append(entry)

    def step_node(self, node_id: str) -> Dict[str, Any]:
        node = self.task_graph[node_id]
        if not self._dependencies_satisfied(node):
            return {
                "node_id": node.node_id,
                "status": node.status.value,
                "phase": node.role_phase.value,
                "skipped": True,
                "reason": "dependencies_not_verified",
            }

        active_roles = derive_active_roles(node)
        snapshot = StepSnapshot(
            node_state=copy.deepcopy(node),
            runtime_state=copy.deepcopy(self.runtime_state),
        )
        node.status = NodeStatus.ACTIVE

        try:
            proposal: ActionProposal = self.generator.propose_action(
                node=node,
                active_roles=active_roles,
                runtime_state=self.runtime_state,
                retry_count=node.retry_count,
            )
            verdict: CriticVerdict = self.critic.evaluate(
                proposal=proposal,
                node=node,
                active_roles=active_roles,
                runtime_state=self.runtime_state,
            )
            if not verdict.accepted:
                self.audit_queue.append(
                    {
                        "node_id": node.node_id,
                        "phase": node.role_phase.value,
                        "reason": verdict.reason,
                        "proposal": asdict(proposal),
                    }
                )
                raise RuntimeError(f"Critic rejected proposal: {verdict.reason}")

            result = self._execute_tool(proposal)
            self._record_ledger_entry(node, proposal, result)
            self._advance_phase(node)
            node.status = NodeStatus.VERIFIED
            return {
                "node_id": node.node_id,
                "status": node.status.value,
                "phase": node.role_phase.value,
                "retries": node.retry_count,
                "accepted": True,
            }
        except Exception as exc:
            self.task_graph[node.node_id] = snapshot.node_state
            self.runtime_state = snapshot.runtime_state
            restored_node = self.task_graph[node.node_id]
            restored_node.retry_count += 1
            if restored_node.retry_count >= self.max_retries:
                restored_node.status = NodeStatus.FAILED
            else:
                restored_node.status = NodeStatus.PENDING
            return {
                "node_id": restored_node.node_id,
                "status": restored_node.status.value,
                "phase": restored_node.role_phase.value,
                "retries": restored_node.retry_count,
                "accepted": False,
                "error": str(exc),
            }
