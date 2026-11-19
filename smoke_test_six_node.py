import json

from stricta import (
    ActionProposal,
    CriticVerdict,
    HuggingFacePrimaryGenerator,
    KarakaRole,
    KarakaTaskNode,
    NodeStatus,
    PaninianKarakaLogitsProcessorFactory,
    RoleProgression,
    ShadowCritic,
    StepSnapshot,
    StrictaAgentRunner,
    derive_active_roles,
    make_jsonl_ledger_executor,
)
from stricta.drivers import run_six_node_branch_driver
from stricta.tools import assert_ledgers_match

EXPECTED_SIX_NODES = {
    "NODE-A",
    "NODE-B",
    "NODE-C",
    "NODE-D",
    "NODE-E",
    "NODE-F",
}


def build_six_node_smoke_graph() -> list[KarakaTaskNode]:
    node_a = KarakaTaskNode(
        node_id="NODE-A",
        objective_description="Initialize local runtime and verify invariants",
        karta="StrictaAgent",
        karma="StateLedger",
        karana="verify_invariants",
        sampradana="AuditManifest",
        apadana="ObservationPayload",
        adhikarana="LocalRuntime",
        role_phase=RoleProgression.INIT,
    )
    node_b = KarakaTaskNode(
        node_id="NODE-B",
        objective_description="Prepare ledger state arm from initialized runtime",
        karta="StrictaAgent",
        karma="StateLedger",
        karana="verify_invariants",
        sampradana="AuditManifest",
        apadana="ObservationPayload",
        adhikarana="LocalRuntime",
        dependencies=["NODE-A"],
        role_phase=RoleProgression.TARGET,
    )
    node_c = KarakaTaskNode(
        node_id="NODE-C",
        objective_description="Prepare audit observation arm from initialized runtime",
        karta="StrictaAgent",
        karma="AuditManifest",
        karana="verify_invariants",
        sampradana="AuditManifest",
        apadana="ObservationPayload",
        adhikarana="LocalRuntime",
        dependencies=["NODE-A"],
        role_phase=RoleProgression.TARGET,
    )
    node_d = KarakaTaskNode(
        node_id="NODE-D",
        objective_description="Finalize ledger state branch ahead of convergence",
        karta="StrictaAgent",
        karma="StateLedger",
        karana="verify_invariants",
        sampradana="AuditManifest",
        apadana="ObservationPayload",
        adhikarana="LocalRuntime",
        dependencies=["NODE-B"],
        role_phase=RoleProgression.BOUNDARY,
    )
    node_e = KarakaTaskNode(
        node_id="NODE-E",
        objective_description="Advance audit observation arm through secondary stage",
        karta="StrictaAgent",
        karma="AuditManifest",
        karana="verify_invariants",
        sampradana="AuditManifest",
        apadana="ObservationPayload",
        adhikarana="LocalRuntime",
        dependencies=["NODE-C"],
        role_phase=RoleProgression.BOUNDARY,
    )
    node_f = KarakaTaskNode(
        node_id="NODE-F",
        objective_description="Commit state ledger and emit audit manifest from NODE-D and NODE-E",
        karta="StrictaAgent",
        karma="StateLedger",
        karana="verify_invariants",
        sampradana="AuditManifest",
        apadana="ObservationPayload",
        adhikarana="LocalRuntime",
        dependencies=["NODE-D", "NODE-E"],
        role_phase=RoleProgression.BOUNDARY,
    )
    return [node_a, node_b, node_c, node_d, node_e, node_f]


def _assert_six_node_summary(
    summary: dict,
    expected_backend: str,
    rollback_target_node: str | None = None,
) -> None:
    assert summary["backend"] == expected_backend, (
        f"Expected backend={expected_backend!r}, got {summary.get('backend')!r}"
    )
    assert summary["ledger_entries"] > 0, (
        f"[{expected_backend}] Expected non-empty ledger for six-node run"
    )
    assert summary["persisted_count"] == summary["ledger_entries"], (
        f"[{expected_backend}] Count mismatch: persisted_count={summary['persisted_count']} "
        f"!= ledger_entries={summary['ledger_entries']}"
    )

    nodes = summary.get("nodes", {})
    assert set(nodes.keys()) == EXPECTED_SIX_NODES, (
        f"[{expected_backend}] Unexpected node set: {sorted(nodes.keys())}"
    )

    if rollback_target_node is not None:
        assert rollback_target_node in EXPECTED_SIX_NODES, (
            f"Unknown rollback_target_node: {rollback_target_node!r}"
        )
        target_info = nodes[rollback_target_node]
        assert target_info.get("retry_count", 0) >= 1, (
            f"[{expected_backend}] Expected {rollback_target_node} retry_count >= 1, "
            f"got {target_info}"
        )

    for nid in sorted(EXPECTED_SIX_NODES):
        info = nodes[nid]
        assert info["status"] == NodeStatus.VERIFIED.value, (
            f"[{expected_backend}] Node {nid} status={info['status']} (expected VERIFIED)"
        )
        assert info["role_phase"] == RoleProgression.COMPLETE.value, (
            f"[{expected_backend}] Node {nid} phase={info['role_phase']} (expected COMPLETE)"
        )


def smoke_test_six_node_stricta(
    model_id: str = "Qwen/Qwen2.5-1.5B-Instruct",
    max_rounds: int = 36,
    jsonl_path: str = "./stricta_six_node_ledger.jsonl",
    db_path: str = "./stricta_six_node_ledger.sqlite",
    rollback_jsonl_path: str = "./stricta_six_node_rollback.jsonl",
    rollback_db_path: str = "./stricta_six_node_rollback.sqlite",
    rollback_target_node: str = "NODE-D",
    verify_parity: bool = True,
) -> dict:
    nodes = build_six_node_smoke_graph()
    assert len(nodes) == 6
    assert KarakaRole.KARTA.value == "Kartā"
    assert len(derive_active_roles(nodes[0])) >= 2

    critic = ShadowCritic(min_confidence=0.0, min_state_change_len=8)
    executor = make_jsonl_ledger_executor(path=jsonl_path, inject_rollback_on=None)
    factory = PaninianKarakaLogitsProcessorFactory(
        tokenizer=None,
        min_valid_tokens=8,
        identifier_boost=2.5,
    )
    assert factory.identifier_boost == 2.5

    class _StubGenerator(HuggingFacePrimaryGenerator):
        def __init__(self):
            pass

    wired_runner = StrictaAgentRunner(
        generator=_StubGenerator(),
        critic=critic,
        tool_executor=executor,
    )
    for n in nodes:
        wired_runner.register_node(n)
    assert set(wired_runner.task_graph.keys()) == EXPECTED_SIX_NODES
    assert isinstance(CriticVerdict(accepted=True, reason="ok"), CriticVerdict)
    assert isinstance(
        StepSnapshot(node_state=nodes[0], runtime_state=wired_runner.runtime_state),
        StepSnapshot,
    )
    assert isinstance(
        ActionProposal(
            action_id="STRICTA-SMOKE-1",
            target_node_id="NODE-A",
            tool_name="verify_invariants",
            arguments={"target": "StateLedger"},
            expected_state_change="Initialize local runtime state",
            confidence_score=0.9,
            tightness_factor=1.0,
        ),
        ActionProposal,
    )

    clean_jsonl = run_six_node_branch_driver(
        model_id=model_id,
        max_rounds=max_rounds,
        backend="jsonl",
        ledger_path=jsonl_path,
        inject_rollback_on=None,
        reset_ledger_file=True,
    )
    _assert_six_node_summary(clean_jsonl, "jsonl", rollback_target_node=None)

    clean_sqlite = run_six_node_branch_driver(
        model_id=model_id,
        max_rounds=max_rounds,
        backend="sqlite",
        ledger_path=db_path,
        inject_rollback_on=None,
        reset_ledger_file=True,
    )
    _assert_six_node_summary(clean_sqlite, "sqlite", rollback_target_node=None)

    clean_parity = (
        assert_ledgers_match(
            jsonl_path=jsonl_path,
            db_path=db_path,
            run_if_missing=False,
            compare_arguments=True,
        )
        if verify_parity
        else None
    )

    rollback_jsonl = run_six_node_branch_driver(
        model_id=model_id,
        max_rounds=max_rounds,
        backend="jsonl",
        ledger_path=rollback_jsonl_path,
        inject_rollback_on=rollback_target_node,
        reset_ledger_file=True,
    )
    _assert_six_node_summary(
        rollback_jsonl,
        "jsonl",
        rollback_target_node=rollback_target_node,
    )

    rollback_sqlite = run_six_node_branch_driver(
        model_id=model_id,
        max_rounds=max_rounds,
        backend="sqlite",
        ledger_path=rollback_db_path,
        inject_rollback_on=rollback_target_node,
        reset_ledger_file=True,
    )
    _assert_six_node_summary(
        rollback_sqlite,
        "sqlite",
        rollback_target_node=rollback_target_node,
    )

    rollback_parity = (
        assert_ledgers_match(
            jsonl_path=rollback_jsonl_path,
            db_path=rollback_db_path,
            run_if_missing=False,
            compare_arguments=True,
        )
        if verify_parity
        else None
    )

    return {
        "clean": {
            "jsonl": clean_jsonl,
            "sqlite": clean_sqlite,
            "parity": clean_parity,
        },
        "rollback_arm": {
            "target_node": rollback_target_node,
            "jsonl": rollback_jsonl,
            "sqlite": rollback_sqlite,
            "parity": rollback_parity,
        },
    }


if __name__ == "__main__":
    result = smoke_test_six_node_stricta(rollback_target_node="NODE-D")
    print("STRICTA SIX-NODE SMOKE TEST PASSED:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
