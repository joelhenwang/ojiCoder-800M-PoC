from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Sequence

from .spec import TokenizerNormalizationConfig


PATH_LINE_PATTERN = re.compile(r"[/\\]|\.[A-Za-z0-9]+")
DIFF_LINE_PREFIXES = ("diff --git", "@@", "+++", "---", "+", "-")
TOOL_MARKERS = ("<|tool_call|>", "<|tool_result|>")


def _canonical_lines(text: str) -> list[str]:
    canonical = text.replace("\r\n", "\n").replace("\r", "\n")
    return canonical.rstrip("\n").split("\n") if canonical else []


def _leading_whitespace_profile(text: str) -> tuple[str, ...]:
    profile: list[str] = []
    for line in _canonical_lines(text):
        prefix_length = len(line) - len(line.lstrip(" \t"))
        profile.append(line[:prefix_length])
    return tuple(profile)


def _extract_path_lines(text: str) -> tuple[str, ...]:
    return tuple(line for line in _canonical_lines(text) if PATH_LINE_PATTERN.search(line))


def _extract_diff_lines(text: str) -> tuple[str, ...]:
    lines: list[str] = []
    for line in _canonical_lines(text):
        if any(line.startswith(prefix) for prefix in DIFF_LINE_PREFIXES):
            lines.append(line)
    return tuple(lines)


def _extract_tool_lines(text: str) -> tuple[str, ...]:
    return tuple(line for line in _canonical_lines(text) if any(marker in line for marker in TOOL_MARKERS))


@dataclass(slots=True)
class TokenizerValidationResult:
    whitespace_fidelity: bool
    path_preservation: bool
    diff_markup: bool
    tool_transcript_markup: bool

    def passed_targets(self) -> dict[str, bool]:
        return {
            "whitespace_fidelity": self.whitespace_fidelity,
            "path_preservation": self.path_preservation,
            "diff_markup": self.diff_markup,
            "tool_transcript_markup": self.tool_transcript_markup,
        }


@dataclass(slots=True)
class TokenizerValidationSummary:
    sample_count: int
    passed_counts: dict[str, int]


def validate_normalized_text(
    text: str,
    normalization: TokenizerNormalizationConfig,
) -> TokenizerValidationResult:
    normalized = normalization.normalize_text(text)
    return TokenizerValidationResult(
        whitespace_fidelity=_leading_whitespace_profile(text) == _leading_whitespace_profile(normalized),
        path_preservation=_extract_path_lines(text) == _extract_path_lines(normalized),
        diff_markup=_extract_diff_lines(text) == _extract_diff_lines(normalized),
        tool_transcript_markup=_extract_tool_lines(text) == _extract_tool_lines(normalized),
    )


def summarize_validation_results(
    texts: Sequence[str],
    normalization: TokenizerNormalizationConfig,
) -> TokenizerValidationSummary:
    passed_counts = {
        "whitespace_fidelity": 0,
        "path_preservation": 0,
        "diff_markup": 0,
        "tool_transcript_markup": 0,
    }
    for text in texts:
        result = validate_normalized_text(text, normalization)
        for name, passed in result.passed_targets().items():
            passed_counts[name] += int(passed)
    return TokenizerValidationSummary(sample_count=len(texts), passed_counts=passed_counts)