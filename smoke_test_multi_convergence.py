import json
from pathlib import Path
from typing import Any, Callable, Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from stricta import (
    HuggingFacePrimaryGenerator,
    KarakaTaskNode,
    NodeStatus,
    PaninianKarakaLogitsProcessorFactory,
    RoleProgression,
    ShadowCritic,
    StrictaAgentRunner,
    make_jsonl_ledger_executor,
)
from stricta.tools import (
    assert_ledgers_match,
    make_sqlite_ledger_executor,
    query_jsonl_ledger,
    query_sqlite_ledger,
)

EXPECTED_NODES = {
    "NODE-A", "NODE-B", "NODE-C", "NODE-D",
    "NODE-E", "NODE-F", "NODE-G", "NODE-H",
}

STEP_ORDER = [
    "NODE-A",
    "NODE-B", "NODE-C",
    "NODE-D", "NODE-E",
    "NODE-F",
    "NODE-G",
    "NODE-H",
]


def build_multi_convergence_graph() -> list[KarakaTaskNode]:
    return [
        KarakaTaskNode(
            node_id="NODE-A",
            objective_description="Initialize root",
            karta="StrictaAgent", karma="StateLedger", karana="verify_invariants",
            sampradana="AuditManifest", apadana="ObservationPayload", adhikarana="LocalRuntime",
            role_phase=RoleProgression.INIT,
        ),
        KarakaTaskNode(
            node_id="NODE-B",
            objective_description="Left early branch",
            karta="StrictaAgent", karma="StateLedger", karana="verify_invariants",
            sampradana="AuditManifest", apadana="ObservationPayload", adhikarana="LocalRuntime",
            dependencies=["NODE-A"],
            role_phase=RoleProgression.TARGET,
        ),
        KarakaTaskNode(
            node_id="NODE-C",
            objective_description="Right early branch",
            karta="StrictaAgent", karma="AuditManifest", karana="verify_invariants",
            sampradana="AuditManifest", apadana="ObservationPayload", adhikarana="LocalRuntime",
            dependencies=["NODE-A"],
            role_phase=RoleProgression.TARGET,
        ),
        KarakaTaskNode(
            node_id="NODE-D",
            objective_description="Left continuation",
            karta="StrictaAgent", karma="StateLedger", karana="verify_invariants",
            sampradana="AuditManifest", apadana="ObservationPayload", adhikarana="LocalRuntime",
            dependencies=["NODE-B"],
            role_phase=RoleProgression.BOUNDARY,
        ),
        KarakaTaskNode(
            node_id="NODE-E",
            objective_description="Right continuation",
            karta="StrictaAgent", karma="AuditManifest", karana="verify_invariants",
            sampradana="AuditManifest", apadana="ObservationPayload", adhikarana="LocalRuntime",
            dependencies=["NODE-C"],
            role_phase=RoleProgression.BOUNDARY,
        ),
        KarakaTaskNode(
            node_id="NODE-F",
            objective_description="Right deep step",
            karta="StrictaAgent", karma="AuditManifest", karana="verify_invariants",
            sampradana="AuditManifest", apadana="ObservationPayload", adhikarana="LocalRuntime",
            dependencies=["NODE-E"],
            role_phase=RoleProgression.BOUNDARY,
        ),
        KarakaTaskNode(
            node_id="NODE-G",
            objective_description="First convergence (D + C)",
            karta="StrictaAgent", karma="StateLedger", karana="verify_invariants",
            sampradana="AuditManifest", apadana="ObservationPayload", adhikarana="LocalRuntime",
            dependencies=["NODE-D", "NODE-C"],
            role_phase=RoleProgression.BOUNDARY,
        ),
        KarakaTaskNode(
            node_id="NODE-H",
            objective_description="Second convergence (F + G)",
            karta="StrictaAgent", karma="StateLedger", karana="verify_invariants",
            sampradana="AuditManifest", apadana="ObservationPayload", adhikarana="LocalRuntime",
            dependencies=["NODE-F", "NODE-G"],
            role_phase=RoleProgression.BOUNDARY,
        ),
    ]


def _wrap_single_attempt_rollback(
    base_executor: Callable[[Any, Dict[str, Any]], Any],
    rollback_target_node: str | None,
    backend_label: str,
) -> Callable[[Any, Dict[str, Any]], Any]:
    triggered = False

    def _wrapped(proposal: Any, runtime_state: Dict[str, Any]) -> Any:
        nonlocal triggered
        if (
            rollback_target_node is not None
            and not triggered
            and proposal.target_node_id == rollback_target_node
        ):
            triggered = True
            raise RuntimeError(
                f"Injected transient Stricta {backend_label} ledger rollback on node "
                f"{proposal.target_node_id}"
            )
        return base_executor(proposal, runtime_state)

    return _wrapped


def run_multi_convergence_backend(
    model_id: str,
    max_rounds: int,
    backend: str,
    ledger_path: str,
    rollback_target_node: str | None = None,
) -> dict:
    normalized_backend = backend.strip().lower()
    path_obj = Path(ledger_path)
    if path_obj.exists():
        path_obj.unlink()

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    model.eval()

    factory = PaninianKarakaLogitsProcessorFactory(
        tokenizer=tokenizer,
        min_valid_tokens=8,
        identifier_boost=2.5,
    )
    generator = HuggingFacePrimaryGenerator(
        model=model,
        tokenizer=tokenizer,
        logits_processor_factory=factory,
        default_gen_kwargs={
            "max_new_tokens": 128,
            "do_sample": False,
        },
        retry_temperature=0.45,
    )

    if normalized_backend == "sqlite":
        base_executor = make_sqlite_ledger_executor(db_path=str(path_obj))
    else:
        base_executor = make_jsonl_ledger_executor(path=str(path_obj))

    executor = _wrap_single_attempt_rollback(
        base_executor=base_executor,
        rollback_target_node=rollback_target_node,
        backend_label=normalized_backend.upper(),
    )

    runner = StrictaAgentRunner(
        generator=generator,
        critic=ShadowCritic(min_confidence=0.0, min_state_change_len=8),
        tool_executor=executor,
    )

    for node in build_multi_convergence_graph():
        runner.register_node(node)

    terminal_statuses = {NodeStatus.VERIFIED, NodeStatus.FAILED}
    for _ in range(max_rounds):
        if all(n.status in terminal_statuses for n in runner.task_graph.values()):
            break
        for nid in STEP_ORDER:
            node = runner.task_graph[nid]
            if node.status in terminal_statuses:
                continue
            event = runner.step_node(nid)
            print(json.dumps(event, ensure_ascii=False))

    if normalized_backend == "sqlite":
        persisted_count = len(query_sqlite_ledger(db_path=str(path_obj)))
    else:
        persisted_count = len(query_jsonl_ledger(path=str(path_obj)))

    return {
        "backend": normalized_backend,
        "nodes": {
            nid: {
                "role_phase": n.role_phase.value,
                "status": n.status.value,
                "retry_count": n.retry_count,
                "tightness": n.containment_tightness,
            }
            for nid, n in runner.task_graph.items()
        },
        "audit_count": len(runner.audit_queue),
        "ledger_entries": len(runner.runtime_state.get("ledger", [])),
        "ledger_path": str(path_obj),
        "persisted_count": persisted_count,
    }


def smoke_test_multi_convergence(
    model_id: str = "Qwen/Qwen2.5-0.5B-Instruct",
    max_rounds: int = 48,
    jsonl_path: str = "./stricta_multi_convergence_ledger.jsonl",
    db_path: str = "./stricta_multi_convergence_ledger.sqlite",
    rollback_target_node: str | None = None,
) -> dict:
    jsonl_summary = run_multi_convergence_backend(
        model_id=model_id,
        max_rounds=max_rounds,
        backend="jsonl",
        ledger_path=jsonl_path,
        rollback_target_node=rollback_target_node,
    )
    sqlite_summary = run_multi_convergence_backend(
        model_id=model_id,
        max_rounds=max_rounds,
        backend="sqlite",
        ledger_path=db_path,
        rollback_target_node=rollback_target_node,
    )

    assert set(jsonl_summary["nodes"].keys()) == EXPECTED_NODES
    assert set(sqlite_summary["nodes"].keys()) == EXPECTED_NODES
    for nid in sorted(EXPECTED_NODES):
        for summary in (jsonl_summary, sqlite_summary):
            info = summary["nodes"][nid]
            assert info["status"] == NodeStatus.VERIFIED.value
            assert info["role_phase"] == RoleProgression.COMPLETE.value

    parity = assert_ledgers_match(
        jsonl_path=jsonl_path,
        db_path=db_path,
        run_if_missing=False,
        compare_arguments=True,
    )

    return {
        "jsonl": jsonl_summary,
        "sqlite": sqlite_summary,
        "parity": parity,
    }


if __name__ == "__main__":
    result = smoke_test_multi_convergence()
    print("STRICTA MULTI-CONVERGENCE SMOKE TEST PASSED:")
    print(json.dumps(result, indent=2, ensure_ascii=False))