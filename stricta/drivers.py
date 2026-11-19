import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .critic import ShadowCritic
from .generator import HuggingFacePrimaryGenerator
from .logits import PaninianKarakaLogitsProcessorFactory
from .runner import StrictaAgentRunner
from .schemas import KarakaTaskNode, NodeStatus, RoleProgression
from .tools import (
    make_jsonl_ledger_executor,
    make_sqlite_ledger_executor,
    query_jsonl_ledger,
    query_sqlite_ledger,
)


def build_diamond_graph() -> List[KarakaTaskNode]:
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
        objective_description="Prepare ledger state branch from initialized runtime",
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
        objective_description="Prepare audit observation branch from initialized runtime",
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
        objective_description="Commit state ledger and emit audit manifest from branches",
        karta="StrictaAgent",
        karma="StateLedger",
        karana="verify_invariants",
        sampradana="AuditManifest",
        apadana="ObservationPayload",
        adhikarana="LocalRuntime",
        dependencies=["NODE-B", "NODE-C"],
        role_phase=RoleProgression.BOUNDARY,
    )
    return [node_a, node_b, node_c, node_d]


def build_six_node_branch_graph() -> List[KarakaTaskNode]:
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


def _run_graph_driver(
    nodes: List[KarakaTaskNode],
    step_order: List[str],
    model_id: str,
    device_map: str,
    max_rounds: int,
    backend: str,
    ledger_path: Optional[str],
    default_jsonl_path: str,
    default_sqlite_path: str,
    inject_rollback_on: Optional[str],
    reset_ledger_file: bool,
) -> Dict[str, Any]:
    normalized_backend = backend.strip().lower()
    if normalized_backend not in {"jsonl", "sqlite"}:
        raise ValueError(
            f"Unsupported ledger backend: {backend!r} (expected 'jsonl' or 'sqlite')"
        )

    default_path = (
        default_jsonl_path if normalized_backend == "jsonl" else default_sqlite_path
    )
    path_obj = Path(ledger_path or default_path)
    if reset_ledger_file and path_obj.exists():
        path_obj.unlink()

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map=device_map,
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
        executor = make_sqlite_ledger_executor(
            db_path=str(path_obj),
            inject_rollback_on=inject_rollback_on,
        )
    else:
        executor = make_jsonl_ledger_executor(
            path=str(path_obj),
            inject_rollback_on=inject_rollback_on,
        )

    runner = StrictaAgentRunner(
        generator=generator,
        critic=ShadowCritic(min_confidence=0.0, min_state_change_len=8),
        tool_executor=executor,
    )

    for n in nodes:
        runner.register_node(n)

    terminal_statuses = {NodeStatus.VERIFIED, NodeStatus.FAILED}
    for _ in range(max_rounds):
        if all(n.status in terminal_statuses for n in runner.task_graph.values()):
            break
        for nid in step_order:
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


def run_multi_branch_driver(
    model_id: str = "Qwen/Qwen2.5-1.5B-Instruct",
    device_map: str = "auto",
    max_rounds: int = 24,
    backend: str = "jsonl",
    ledger_path: Optional[str] = None,
    inject_rollback_on: Optional[str] = None,
    reset_ledger_file: bool = True,
) -> Dict[str, Any]:
    return _run_graph_driver(
        nodes=build_diamond_graph(),
        step_order=["NODE-A", "NODE-B", "NODE-C", "NODE-D"],
        model_id=model_id,
        device_map=device_map,
        max_rounds=max_rounds,
        backend=backend,
        ledger_path=ledger_path,
        default_jsonl_path="./stricta_ledger.jsonl",
        default_sqlite_path="./stricta_ledger.sqlite",
        inject_rollback_on=inject_rollback_on,
        reset_ledger_file=reset_ledger_file,
    )


def run_six_node_branch_driver(
    model_id: str = "Qwen/Qwen2.5-1.5B-Instruct",
    device_map: str = "auto",
    max_rounds: int = 36,
    backend: str = "jsonl",
    ledger_path: Optional[str] = None,
    inject_rollback_on: Optional[str] = None,
    reset_ledger_file: bool = True,
) -> Dict[str, Any]:
    return _run_graph_driver(
        nodes=build_six_node_branch_graph(),
        step_order=["NODE-A", "NODE-B", "NODE-C", "NODE-D", "NODE-E", "NODE-F"],
        model_id=model_id,
        device_map=device_map,
        max_rounds=max_rounds,
        backend=backend,
        ledger_path=ledger_path,
        default_jsonl_path="./stricta_six_node_ledger.jsonl",
        default_sqlite_path="./stricta_six_node_ledger.sqlite",
        inject_rollback_on=inject_rollback_on,
        reset_ledger_file=reset_ledger_file,
    )