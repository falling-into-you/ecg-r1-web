"""Direct vLLM inference server for ECG-R1.

This server is for Web inference only. It avoids Swift GRPO rollout and calls
vLLM's AsyncLLMEngine directly, so streaming can return generation deltas.
"""

from __future__ import annotations

import json
import os
import uuid
from typing import Any, AsyncGenerator

import config

for _key, _value in config.RUNTIME_ENV_VARS.items():
    os.environ[_key] = str(_value)
os.environ["VLLM_LOAD_FORMAT"] = str(config.VLLM_LOAD_FORMAT)

from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import StreamingResponse
from PIL import Image
from transformers import AutoTokenizer
from vllm import AsyncLLMEngine, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs

from ecg_r1_runtime.ecg_io import load_ecg
from ecg_r1_runtime.vllm_plugin import register as register_vllm_plugin


SYSTEM_PROMPT = "You are a helpful, harmless clinical ECG assistant. Provide concise, evidence-based interpretations."
IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"
ECG_PLACEHOLDER = "<|ecg_start|><|ecg_pad|><|ecg_end|>"
QWEN3VL_FACTOR = 32

app = FastAPI()
engine: AsyncLLMEngine | None = None
tokenizer = None


def _mm_processor_kwargs() -> dict[str, int]:
    min_tokens = int(config.IMAGE_MIN_TOKEN_NUM)
    max_tokens = int(config.IMAGE_MAX_TOKEN_NUM)
    return {
        "min_pixels": min_tokens * (QWEN3VL_FACTOR ** 2),
        "max_pixels": max_tokens * (QWEN3VL_FACTOR ** 2),
    }


@app.on_event("startup")
async def startup() -> None:
    global engine, tokenizer

    register_vllm_plugin()
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_PATH, trust_remote_code=True)

    engine_args = AsyncEngineArgs(
        model=config.MODEL_PATH,
        tokenizer=config.MODEL_PATH,
        served_model_name=config.MODEL_DISPLAY_NAME,
        trust_remote_code=True,
        dtype="bfloat16",
        load_format=config.VLLM_LOAD_FORMAT,
        enforce_eager=config.VLLM_ENFORCE_EAGER,
        max_model_len=config.VLLM_MAX_MODEL_LEN,
        max_num_seqs=config.VLLM_MAX_NUM_SEQS,
        disable_custom_all_reduce=True,
        limit_mm_per_prompt={"image": 4, "ecg": 1},
        mm_processor_kwargs=_mm_processor_kwargs(),
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)


@app.get("/health/")
async def health() -> dict[str, str]:
    if engine is None:
        return {"status": "loading"}
    return {"status": "ok"}


def _replace_modality_tags(text: str) -> str:
    return text.replace("<image>", IMAGE_PLACEHOLDER).replace("<ecg>", ECG_PLACEHOLDER)


def _build_messages(messages: list[dict[str, Any]], images: list[str], ecgs: list[str]) -> list[dict[str, str]]:
    rendered: list[dict[str, str]] = []
    has_system = any(message.get("role") == "system" for message in messages)
    if not has_system:
        rendered.append({"role": "system", "content": SYSTEM_PROMPT})

    image_count = 0
    ecg_count = 0
    for message in messages:
        role = str(message.get("role") or "user")
        content = message.get("content") or ""
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and (item.get("type") == "image" or "image" in item or "image_url" in item):
                    parts.append("<image>")
                elif isinstance(item, dict) and item.get("type") == "ecg":
                    parts.append("<ecg>")
                elif isinstance(item, dict) and "text" in item:
                    parts.append(str(item.get("text") or ""))
                else:
                    parts.append(str(item))
            content = "".join(parts)
        content = _replace_modality_tags(str(content))
        image_count += content.count(IMAGE_PLACEHOLDER)
        ecg_count += content.count(ECG_PLACEHOLDER)
        rendered.append({"role": role, "content": content})

    missing_images = max(0, len(images) - image_count)
    missing_ecgs = max(0, len(ecgs) - ecg_count)
    if rendered and (missing_images or missing_ecgs):
        prefix = IMAGE_PLACEHOLDER * missing_images + ECG_PLACEHOLDER * missing_ecgs
        for message in rendered:
            if message["role"] == "user":
                message["content"] = prefix + message["content"]
                break
    return rendered


def _build_prompt(infer_request: dict[str, Any]) -> dict[str, Any]:
    images = [str(path) for path in infer_request.get("images") or [] if path]
    objects = infer_request.get("objects") if isinstance(infer_request.get("objects"), dict) else {}
    ecgs = [str(path) for path in objects.get("ecg") or [] if path]
    messages = _build_messages(infer_request.get("messages") or [], images, ecgs)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    mm_data: dict[str, Any] = {}
    if images:
        mm_data["image"] = [Image.open(path).convert("RGB") for path in images]
    if ecgs:
        seq_length = int(config.ECG_SEQ_LENGTH)
        root_ecg_dir = config.ROOT_ECG_DIR
        mm_data["ecg"] = [load_ecg(path, seq_length, root_ecg_dir) for path in ecgs]

    llm_input: dict[str, Any] = {
        "prompt": prompt,
        "mm_processor_kwargs": _mm_processor_kwargs(),
    }
    if mm_data:
        llm_input["multi_modal_data"] = mm_data
    return llm_input


def _sampling_params(request_config: dict[str, Any]) -> SamplingParams:
    return SamplingParams(
        temperature=float(request_config.get("temperature", 0.0)),
        top_p=float(request_config.get("top_p", 1.0)),
        top_k=int(request_config.get("top_k", 0)),
        repetition_penalty=float(request_config.get("repetition_penalty", 1.0)),
        max_tokens=int(request_config.get("max_tokens", config.DEFAULT_MAX_TOKENS)),
    )


async def _generate(prompt_input: dict[str, Any], sampling_params: SamplingParams) -> AsyncGenerator[str, None]:
    if engine is None:
        raise RuntimeError("vLLM engine is not ready")

    request_id = f"ecg-r1-{uuid.uuid4()}"
    previous_text = ""
    async for output in engine.generate(prompt_input, sampling_params, request_id=request_id):
        if not output.outputs:
            continue
        text = output.outputs[0].text or ""
        if len(text) > len(previous_text):
            delta = text[len(previous_text):]
            previous_text = text
            if delta:
                yield delta


async def _generate_text(prompt_input: dict[str, Any], sampling_params: SamplingParams) -> str:
    chunks = []
    async for chunk in _generate(prompt_input, sampling_params):
        chunks.append(chunk)
    return "".join(chunks)


def _response_payload(text: str) -> list[dict[str, Any]]:
    return [{
        "response": {
            "model": config.MODEL_DISPLAY_NAME,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": text,
                },
                "finish_reason": "stop",
            }],
        }
    }]


@app.post("/infer/")
async def infer(payload: dict[str, Any] = Body(...)):
    infer_requests = payload.get("infer_requests") or []
    if not infer_requests:
        raise HTTPException(status_code=400, detail="Missing infer_requests.")
    if len(infer_requests) != 1:
        raise HTTPException(status_code=400, detail="Direct vLLM server currently expects one request at a time.")

    request_config = payload.get("request_config") or {}
    prompt_input = _build_prompt(infer_requests[0])
    sampling_params = _sampling_params(request_config)

    if request_config.get("stream"):
        async def event_gen():
            try:
                async for chunk in _generate(prompt_input, sampling_params):
                    yield f"event: content\ndata: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                yield f"event: done\ndata: {json.dumps({'status': 'ok'}, ensure_ascii=False)}\n\n"
            except Exception as exc:
                yield f"event: error\ndata: {json.dumps({'detail': str(exc)}, ensure_ascii=False)}\n\n"

        return StreamingResponse(
            event_gen(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    text = await _generate_text(prompt_input, sampling_params)
    return _response_payload(text)


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=config.VLLM_PORT, log_level=config.WEB_LOG_LEVEL)


if __name__ == "__main__":
    main()
