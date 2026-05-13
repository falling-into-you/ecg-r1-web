from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Protocol


@dataclass
class InferenceRequest:
    messages: list[dict[str, str]]
    images: list[str] = field(default_factory=list)
    objects: dict[str, list[str]] = field(default_factory=dict)
    request_config: dict[str, Any] = field(default_factory=dict)


@dataclass
class InferenceResult:
    content: str
    reasoning: str = ""
    raw: Any = None


@dataclass
class InferenceChunk:
    event: str
    text: str


class InferenceProvider(Protocol):
    name: str

    def status(self) -> Mapping[str, Any]:
        ...

    def infer(self, request: InferenceRequest) -> InferenceResult:
        ...

    def stream(self, request: InferenceRequest) -> Iterable[InferenceChunk]:
        ...


def chunk_text(text: str, size: int = 64) -> Iterable[str]:
    for idx in range(0, len(text), size):
        yield text[idx:idx + size]
