from __future__ import annotations

from typing import Any, Iterable, Mapping

from .base import InferenceChunk, InferenceRequest, InferenceResult, chunk_text


class MockProvider:
    name = "mock"

    def status(self) -> Mapping[str, Any]:
        return {
            "status": "online",
            "detail": "Mock inference provider is active",
            "provider": self.name,
        }

    def infer(self, request: InferenceRequest) -> InferenceResult:
        modalities = []
        if request.images:
            modalities.append("image")
        if request.objects.get("ecg"):
            modalities.append("ecg")
        label = " + ".join(modalities) if modalities else "text"
        content = (
            f"<think>Mock provider received {label} input.</think>\n"
            "<answer>This is a mock ECG-R1 response. Start a real rollout service "
            "and set INFERENCE_BACKEND=swift_rollout for model inference.</answer>"
        )
        return InferenceResult(content=content, raw={"provider": self.name})

    def stream(self, request: InferenceRequest) -> Iterable[InferenceChunk]:
        result = self.infer(request)
        for text in chunk_text(result.content):
            yield InferenceChunk(event="content", text=text)
