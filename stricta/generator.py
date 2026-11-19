import json
import re
from typing import Any, Dict, List, Optional

from .logits import PaninianKarakaLogitsProcessorFactory
from .schemas import ActionProposal, KarakaRole, KarakaTaskNode


class HuggingFacePrimaryGenerator:
    def __init__(
        self,
        model: Any = None,
        tokenizer: Any = None,
        logits_processor_factory: Optional[PaninianKarakaLogitsProcessorFactory] = None,
        default_gen_kwargs: Optional[Dict[str, Any]] = None,
        retry_temperature: float = 0.45,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.logits_processor_factory = logits_processor_factory
        self.default_gen_kwargs: Dict[str, Any] = dict(default_gen_kwargs or {})
        self.retry_temperature = retry_temperature

    def _build_prompt(
        self,
        node: KarakaTaskNode,
        active_roles: List[KarakaRole],
        runtime_state: Dict[str, Any],
        retry_count: int,
    ) -> str:
        role_labels = [r.value for r in active_roles]
        ledger_len = len(runtime_state.get("ledger", []))
        return (
            f"Stricta Deterministic Action Proposal\n"
            f"Node: {node.node_id} | Phase: {node.role_phase.value} | Retry: {retry_count}\n"
            f"Objective: {node.objective_description}\n"
            f"Karaka: karta={node.karta}, karma={node.karma}, karana={node.karana}, "
            f"sampradana={node.sampradana}, apadana={node.apadana}, adhikarana={node.adhikarana}\n"
            f"Active Roles: {role_labels}\n"
            f"Ledger Entries: {ledger_len}\n"
            f"Return JSON with keys: tool_name, arguments, expected_state_change, confidence_score.\n"
        )

    def _extract_json_dict(self, raw_text: str) -> Dict[str, Any]:
        if not raw_text:
            return {}
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, flags=re.DOTALL)
        candidate = fenced.group(1) if fenced else None
        if candidate is None:
            start = raw_text.find("{")
            end = raw_text.rfind("}")
            if start != -1 and end != -1 and end > start:
                candidate = raw_text[start : end + 1]
        if candidate is None:
            return {}
        try:
            parsed = json.loads(candidate)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            repaired = re.sub(r",\s*([}\]])", r"\1", candidate)
            repaired = re.sub(r"'([A-Za-z0-9_]+)'\s*:", r'"\1":', repaired)
            try:
                parsed = json.loads(repaired)
                return parsed if isinstance(parsed, dict) else {}
            except Exception:
                return {}

    def _generate_raw_completion(
        self,
        prompt: str,
        node: KarakaTaskNode,
        active_roles: List[KarakaRole],
        retry_count: int,
    ) -> str:
        if self.model is None or self.tokenizer is None:
            return ""

        gen_kwargs = dict(self.default_gen_kwargs)
        if retry_count > 0:
            gen_kwargs["temperature"] = min(
                float(gen_kwargs.get("temperature", self.retry_temperature)),
                self.retry_temperature,
            )

        context = {
            "node_id": node.node_id,
            "karta": node.karta,
            "karma": node.karma,
            "karana": node.karana,
            "sampradana": node.sampradana,
            "apadana": node.apadana,
            "adhikarana": node.adhikarana,
            "active_roles": [r.value for r in active_roles],
        }
        logits_processors = None
        if self.logits_processor_factory is not None:
            logits_processors = self.logits_processor_factory.build_processors(context=context)

        inputs = self.tokenizer(prompt, return_tensors="pt")
        device = getattr(self.model, "device", None)
        if device is not None and hasattr(inputs, "to"):
            inputs = inputs.to(device)
        elif device is not None and isinstance(inputs, dict):
            inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

        call_kwargs: Dict[str, Any] = {
            "max_new_tokens": gen_kwargs.get("max_new_tokens", 128),
            "do_sample": gen_kwargs.get("do_sample", False),
        }
        if call_kwargs["do_sample"]:
            call_kwargs["temperature"] = gen_kwargs.get("temperature", 0.7)
            call_kwargs["top_p"] = gen_kwargs.get("top_p", 0.9)
        if logits_processors:
            call_kwargs["logits_processor"] = logits_processors

        output_ids = self.model.generate(**inputs, **call_kwargs)
        prompt_len = inputs["input_ids"].shape[-1] if "input_ids" in inputs else 0
        gen_ids = output_ids[0][prompt_len:] if prompt_len else output_ids[0]
        return self.tokenizer.decode(gen_ids, skip_special_tokens=True)

    def propose_action(
        self,
        node: KarakaTaskNode,
        active_roles: List[KarakaRole],
        runtime_state: Dict[str, Any],
        retry_count: int = 0,
    ) -> ActionProposal:
        prompt = self._build_prompt(
            node=node,
            active_roles=active_roles,
            runtime_state=runtime_state,
            retry_count=retry_count,
        )
        raw_text = self._generate_raw_completion(
            prompt=prompt,
            node=node,
            active_roles=active_roles,
            retry_count=retry_count,
        )
        payload = self._extract_json_dict(raw_text)

        tool_name = str(payload.get("tool_name") or node.karana or "verify_invariants")
        raw_args = payload.get("arguments")
        arguments = dict(raw_args) if isinstance(raw_args, dict) else {"target": node.karma}

        expected_change = str(payload.get("expected_state_change") or "").strip()
        if len(expected_change) < 8:
            expected_change = (
                f"Advance {node.node_id} ({node.role_phase.value}) on {node.karma} via {tool_name}"
            )

        raw_conf = payload.get("confidence_score", 0.9)
        try:
            confidence = float(raw_conf)
        except Exception:
            confidence = 0.9

        return ActionProposal(
            action_id=f"STRICTA-{node.node_id}-R{retry_count}",
            target_node_id=node.node_id,
            tool_name=tool_name,
            arguments=arguments,
            expected_state_change=expected_change,
            confidence_score=confidence,
            tightness_factor=node.containment_tightness,
        )
