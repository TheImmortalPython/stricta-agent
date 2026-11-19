from typing import Any, Dict, Iterable, List, Optional


class _KarakaTokenLogitsProcessor:
    def __init__(
        self,
        tokenizer: Any,
        allowed_identifiers: Optional[Iterable[str]] = None,
        min_valid_tokens: int = 8,
        identifier_boost: float = 2.5,
    ) -> None:
        self.tokenizer = tokenizer
        self.allowed_identifiers = list(allowed_identifiers or [])
        self.min_valid_tokens = min_valid_tokens
        self.identifier_boost = identifier_boost
        self._boosted_token_ids: List[int] = []
        self._resolve_boost_tokens()

    def _resolve_boost_tokens(self) -> None:
        if self.tokenizer is None or not self.allowed_identifiers:
            return
        seen: set[int] = set()
        for ident in self.allowed_identifiers:
            if not ident:
                continue
            try:
                token_ids = self.tokenizer.encode(ident, add_special_tokens=False)
            except Exception:
                continue
            for tid in token_ids:
                if isinstance(tid, int) and tid >= 0 and tid not in seen:
                    seen.add(tid)
                    self._boosted_token_ids.append(tid)

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        if not self._boosted_token_ids or scores is None:
            return scores
        try:
            for tid in self._boosted_token_ids:
                if tid < scores.shape[-1]:
                    scores[..., tid] = scores[..., tid] + self.identifier_boost
        except Exception:
            pass
        return scores


class PaninianKarakaLogitsProcessorFactory:
    def __init__(
        self,
        tokenizer: Any = None,
        min_valid_tokens: int = 8,
        identifier_boost: float = 2.5,
    ) -> None:
        self.tokenizer = tokenizer
        self.min_valid_tokens = min_valid_tokens
        self.identifier_boost = identifier_boost

    def build_processors(
        self,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[_KarakaTokenLogitsProcessor]:
        identifiers: List[str] = []
        if context:
            for key in ("node_id", "karta", "karma", "karana", "sampradana", "apadana", "adhikarana"):
                val = context.get(key)
                if isinstance(val, str) and val:
                    identifiers.append(val)
            extra = context.get("allowed_identifiers")
            if isinstance(extra, (list, tuple, set)):
                identifiers.extend(str(x) for x in extra if x)

        return [
            _KarakaTokenLogitsProcessor(
                tokenizer=self.tokenizer,
                allowed_identifiers=identifiers,
                min_valid_tokens=self.min_valid_tokens,
                identifier_boost=self.identifier_boost,
            )
        ]
