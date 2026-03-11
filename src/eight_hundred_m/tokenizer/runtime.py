from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class TokenizerRuntime:
    tokenizer_path: Path
    backend: object

    def encode(self, text: str) -> list[int]:
        encoding = self.backend.encode(text)
        token_ids = getattr(encoding, "ids", None)
        if not isinstance(token_ids, list):
            raise ValueError("tokenizer backend returned an invalid encoding payload")
        return [int(token_id) for token_id in token_ids]

    def count_tokens(self, text: str) -> int:
        return len(self.encode(text))


def load_tokenizer_runtime(tokenizer_path: str | Path) -> TokenizerRuntime:
    tokenizers = importlib.import_module("tokenizers")
    backend = tokenizers.Tokenizer.from_file(str(tokenizer_path))
    return TokenizerRuntime(tokenizer_path=Path(tokenizer_path), backend=backend)