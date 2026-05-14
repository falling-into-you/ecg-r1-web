from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any, Iterable, Mapping

from .base import InferenceChunk, InferenceRequest, InferenceResult


class VLLMDirectProvider:
    name = "vllm_direct"

    def __init__(self, infer_url: str, health_url: str, timeout_s: float = 300.0):
        self.infer_url = infer_url
        self.health_url = health_url
        self.timeout_s = timeout_s
        self._last_online_at: float | None = None
        self._last_health_body = ""

    def status(self) -> Mapping[str, Any]:
        try:
            req = urllib.request.Request(self.health_url, method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                body = resp.read().decode("utf-8", errors="ignore")
            health_status = self._health_status(body)
            if health_status not in ("ok", "online", "ready"):
                return {
                    "status": "loading" if health_status in ("loading", "pending") else "offline",
                    "detail": f"Direct vLLM health status is {health_status or 'unknown'}",
                    "provider": self.name,
                    "infer_url": self.infer_url,
                    "health_url": self.health_url,
                    "health_body": body[:300],
                }
            self._last_online_at = time.time()
            self._last_health_body = body[:300]
            return {
                "status": "online",
                "detail": f"Direct vLLM service returned health status {resp.status}",
                "provider": self.name,
                "infer_url": self.infer_url,
                "health_url": self.health_url,
                "health_body": self._last_health_body,
            }
        except Exception as exc:
            if self._last_online_at is not None:
                age_s = time.time() - self._last_online_at
                if age_s < 120:
                    return {
                        "status": "loading",
                        "detail": f"Direct vLLM health check failed after a recent success ({age_s:.1f}s ago): {exc}",
                        "provider": self.name,
                        "infer_url": self.infer_url,
                        "health_url": self.health_url,
                        "health_body": self._last_health_body,
                    }
            return {
                "status": "offline",
                "detail": str(exc),
                "provider": self.name,
                "infer_url": self.infer_url,
                "health_url": self.health_url,
            }

    def _health_status(self, body: str) -> str:
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            return "ok" if body.strip() else "unknown"
        if isinstance(payload, dict):
            return str(payload.get("status") or "").strip().lower()
        return "unknown"

    def infer(self, request: InferenceRequest) -> InferenceResult:
        payload = self._payload(request, stream=False)
        raw = self._post_json(payload)
        return InferenceResult(content=self._extract_text(raw), raw=raw)

    def stream(self, request: InferenceRequest) -> Iterable[InferenceChunk]:
        payload = self._payload(request, stream=True)
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            self.infer_url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                yield from self._iter_sse(resp)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"direct vLLM error {exc.code}: {detail[:1000]}") from exc

    def _payload(self, request: InferenceRequest, stream: bool) -> dict[str, Any]:
        request_config = dict(request.request_config)
        request_config["stream"] = stream
        return {
            "infer_requests": [{
                "messages": request.messages,
                "images": request.images,
                "objects": request.objects,
            }],
            "request_config": request_config,
        }

    def _post_json(self, payload: dict[str, Any]) -> Any:
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
            raise RuntimeError(f"direct vLLM error {exc.code}: {detail[:1000]}") from exc

        try:
            return json.loads(raw_text)
        except json.JSONDecodeError:
            return raw_text

    def _iter_sse(self, resp) -> Iterable[InferenceChunk]:
        event = "message"
        data_lines: list[str] = []

        for raw_line in resp:
            line = raw_line.decode("utf-8", errors="ignore").rstrip("\r\n")
            if not line:
                if data_lines:
                    payload = "\n".join(data_lines)
                    try:
                        value = json.loads(payload)
                    except json.JSONDecodeError:
                        value = payload
                    if event == "content":
                        yield InferenceChunk(event="content", text=str(value))
                    elif event == "reasoning":
                        yield InferenceChunk(event="reasoning", text=str(value))
                    elif event == "error":
                        if isinstance(value, dict):
                            raise RuntimeError(str(value.get("detail") or value))
                        raise RuntimeError(str(value))
                event = "message"
                data_lines = []
                continue
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                data_lines.append(line[5:].strip())

    def _extract_text(self, raw: Any) -> str:
        if isinstance(raw, list) and raw:
            return self._extract_text(raw[0])
        if isinstance(raw, dict):
            for key in ("response", "result", "content", "text", "output"):
                value = raw.get(key)
                if isinstance(value, str):
                    return value
                if isinstance(value, (dict, list)):
                    return self._extract_text(value)
            choices = raw.get("choices")
            if isinstance(choices, list) and choices:
                return self._extract_text(choices[0])
            message = raw.get("message")
            if isinstance(message, dict):
                return self._extract_text(message)
        if isinstance(raw, str):
            return raw
        return json.dumps(raw, ensure_ascii=False)
