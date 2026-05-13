from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Iterable, Mapping

from .base import InferenceChunk, InferenceRequest, InferenceResult, chunk_text


class SwiftRolloutProvider:
    name = "swift_rollout"

    def __init__(self, infer_url: str, health_url: str | None = None, timeout_s: float = 300.0):
        self.infer_url = infer_url
        self.health_url = health_url or infer_url.rstrip("/").rsplit("/", 1)[0] + "/health/"
        self.timeout_s = timeout_s

    def status(self) -> Mapping[str, Any]:
        try:
            req = urllib.request.Request(self.health_url, method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                body = resp.read().decode("utf-8", errors="ignore")
            return {
                "status": "online",
                "detail": f"Swift rollout service returned health status {resp.status}",
                "provider": self.name,
                "infer_url": self.infer_url,
                "health_url": self.health_url,
                "health_body": body[:300],
            }
        except Exception as exc:
            return {
                "status": "offline",
                "detail": str(exc),
                "provider": self.name,
                "infer_url": self.infer_url,
                "health_url": self.health_url,
            }

    def infer(self, request: InferenceRequest) -> InferenceResult:
        payload = {
            "infer_requests": [{
                "messages": request.messages,
                "images": request.images,
                "objects": request.objects,
            }],
            "request_config": request.request_config,
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            self.infer_url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                raw_text = resp.read().decode("utf-8", errors="ignore")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"swift rollout error {exc.code}: {detail[:1000]}") from exc

        try:
            raw = json.loads(raw_text)
        except json.JSONDecodeError:
            raw = raw_text
        return InferenceResult(content=self._extract_text(raw), raw=raw)

    def stream(self, request: InferenceRequest) -> Iterable[InferenceChunk]:
        result = self.infer(request)
        if result.reasoning:
            for text in chunk_text(result.reasoning):
                yield InferenceChunk(event="reasoning", text=text)
        for text in chunk_text(result.content):
            yield InferenceChunk(event="content", text=text)

    def _extract_text(self, raw: Any) -> str:
        if isinstance(raw, list) and raw:
            return self._extract_text(raw[0])
        if isinstance(raw, dict):
            for key in ("response", "result", "content", "text", "output"):
                value = raw.get(key)
                if isinstance(value, str):
                    return value
            choices = raw.get("choices")
            if isinstance(choices, list) and choices:
                return self._extract_text(choices[0])
            message = raw.get("message")
            if isinstance(message, dict):
                return self._extract_text(message)
        if isinstance(raw, str):
            return raw
        return json.dumps(raw, ensure_ascii=False)
