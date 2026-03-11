from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .spec import TokenizerCorpusManifest


@dataclass(slots=True)
class TokenizerCorpusFile:
    entry_name: str
    path: Path
    content_type: str
    weight: float


def _expand_brace_glob(pattern: str) -> tuple[str, ...]:
    start = pattern.find("{")
    end = pattern.find("}", start + 1)
    if start == -1 or end == -1 or end < start:
        return (pattern,)

    prefix = pattern[:start]
    suffix = pattern[end + 1 :]
    variants = pattern[start + 1 : end].split(",")
    expanded: list[str] = []
    for variant in variants:
        expanded.extend(_expand_brace_glob(prefix + variant + suffix))
    return tuple(expanded)


def iter_manifest_glob_matches(root: str | Path, pattern: str) -> tuple[Path, ...]:
    root_path = Path(root)
    matches: set[Path] = set()
    for expanded_pattern in _expand_brace_glob(pattern):
        for path in root_path.glob(expanded_pattern):
            matches.add(path)
    return tuple(sorted(matches))


def discover_tokenizer_corpus_files(
    manifest: TokenizerCorpusManifest,
    root: str | Path,
) -> tuple[TokenizerCorpusFile, ...]:
    discovered: list[TokenizerCorpusFile] = []
    for entry in manifest.entries:
        for path in iter_manifest_glob_matches(root, entry.path_glob):
            if path.is_file():
                discovered.append(
                    TokenizerCorpusFile(
                        entry_name=entry.name,
                        path=path,
                        content_type=entry.content_type,
                        weight=entry.weight,
                    )
                )
    return tuple(sorted(discovered, key=lambda item: (item.entry_name, str(item.path))))


def summarize_tokenizer_corpus(
    manifest: TokenizerCorpusManifest,
    root: str | Path,
) -> dict[str, int]:
    summary = {entry.name: 0 for entry in manifest.entries}
    for discovered in discover_tokenizer_corpus_files(manifest, root):
        summary[discovered.entry_name] += 1
    return summary


def load_normalized_corpus_texts(
    files: Sequence[TokenizerCorpusFile],
    normalization: TokenizerCorpusManifest | None = None,
) -> tuple[str, ...]:
    texts: list[str] = []
    normalizer = normalization.normalization if normalization is not None else None
    for corpus_file in files:
        content = corpus_file.path.read_text(encoding="utf-8")
        texts.append(normalizer.normalize_text(content) if normalizer else content)
    return tuple(texts)
