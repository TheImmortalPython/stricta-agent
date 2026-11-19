from typing import Any, Dict, List, Optional, Set

from .schemas import ActionProposal, CriticVerdict, KarakaRole, KarakaTaskNode

_EXPLICIT_DOMAIN_TOOLS: Set[str] = {
    "verify_invariants",
    "record_ledger_entry",
    "emit_audit_manifest",
    "commit_state_ledger",
}


class ShadowCritic:
    def __init__(
        self,
        min_confidence: float = 0.0,
        min_state_change_len: int = 8,
        allowed_tools: Optional[Set[str]] = None,
    ) -> None:
        self.min_confidence = min_confidence
        self.min_state_change_len = min_state_change_len
        self.allowed_tools: Set[str] = (
            set(allowed_tools) if allowed_tools is not None else set(_EXPLICIT_DOMAIN_TOOLS)
        )

    def evaluate(
        self,
        proposal: ActionProposal,
        node: KarakaTaskNode,
        active_roles: List[KarakaRole],
        runtime_state: Optional[Dict[str, Any]] = None,
    ) -> CriticVerdict:
        if proposal.target_node_id != node.node_id:
            return CriticVerdict(
                accepted=False,
                reason=f"Target node mismatch: expected {node.node_id}, got {proposal.target_node_id}",
            )

        if not proposal.tool_name or proposal.tool_name not in self.allowed_tools:
            return CriticVerdict(
                accepted=False,
                reason=f"Disallowed or missing tool_name: {proposal.tool_name!r}",
            )

        if proposal.confidence_score < self.min_confidence:
            return CriticVerdict(
                accepted=False,
                reason=(
                    f"Confidence {proposal.confidence_score} below threshold {self.min_confidence}"
                ),
            )

        state_change = (proposal.expected_state_change or "").strip()
        if len(state_change) < self.min_state_change_len:
            return CriticVerdict(
                accepted=False,
                reason=(
                    f"expected_state_change length {len(state_change)} is below "
                    f"min_state_change_len={self.min_state_change_len}"
                ),
            )

        if not active_roles:
            return CriticVerdict(
                accepted=False,
                reason="No active Pāṇinian Kāraka roles derived for node",
            )

        return CriticVerdict(
            accepted=True,
            reason="approved_by_stricta_shadow_critic",
            metadata={
                "node_id": node.node_id,
                "active_roles": [r.value for r in active_roles],
                "tool_name": proposal.tool_name,
            },
        )
