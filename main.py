import sys
import os
import config
from config import DATA_COLLECTION_DIR

import shutil
import asyncio
import queue
import threading
import datetime
import uuid
import time
import ipaddress
import urllib.request
import urllib.parse

# Force flush stdout
sys.stdout.reconfigure(line_buffering=True)
from typing import Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from fastapi.middleware.cors import CORSMiddleware
import json
from collections import Counter, defaultdict
from providers import get_provider
from providers.base import InferenceRequest

inference_provider = get_provider()
stream_states = {}

def _peer_ip(request: Request) -> str:
    if request.client and request.client.host:
        return request.client.host
    return "unknown"

def _safe_filename(name: str) -> str:
    base = os.path.basename(str(name or "")).strip()
    base = base.replace("/", "_").replace("\\", "_").replace("\x00", "")
    return base or "file"

def _client_ip(request: Request) -> str:
    candidates = []
    for key in ("cf-connecting-ip", "true-client-ip", "x-real-ip"):
        raw = request.headers.get(key)
        if raw:
            candidates.append(raw.strip())
    xff = request.headers.get("x-forwarded-for")
    if xff:
        candidates.extend([p.strip() for p in xff.split(",") if p.strip()])

    public_ip = None
    fallback_ip = None
    for ip in candidates:
        try:
            ipa = ipaddress.ip_address(ip)
        except Exception:
            continue
        if fallback_ip is None:
            fallback_ip = ip
        if not (ipa.is_private or ipa.is_loopback or ipa.is_link_local or ipa.is_reserved or ipa.is_multicast):
            public_ip = ip
            break
    if public_ip:
        return public_ip
    if fallback_ip:
        return fallback_ip
    return _peer_ip(request)

def _client_geo(request: Request) -> dict:
    country = request.headers.get("cf-ipcountry") or request.headers.get("x-geo-country") or request.headers.get("x-country")
    region = request.headers.get("x-geo-region") or request.headers.get("x-region")
    city = request.headers.get("x-geo-city") or request.headers.get("x-city")
    geo = {
        "country": country,
        "country_code": country,
        "region": region,
        "city": city,
        "source": "headers" if any([country, region, city]) else None,
    }
    return _normalize_geo_policy(geo)

_geoip_cache = {}

def _is_public_ip(ip: str) -> bool:
    try:
        ipa = ipaddress.ip_address(ip)
    except Exception:
        return False
    if ipa.is_private or ipa.is_loopback or ipa.is_link_local or ipa.is_reserved or ipa.is_multicast:
        return False
    return True

def _normalize_geo_policy(geo: dict) -> dict:
    if not isinstance(geo, dict):
        return geo
    tw_as_cn = os.environ.get("GEO_OVERRIDE_TW_AS_CN", "1").strip().lower() in ("1", "true", "yes", "on")
    cc = (geo.get("country_code") or geo.get("country") or "").strip().upper()
    region = (geo.get("region") or "").strip()
    city = (geo.get("city") or "").strip()
    country_name = (geo.get("country_name") or "").strip()
    if tw_as_cn and cc == "TW":
        geo = dict(geo)
        geo["country_code"] = "CN"
        geo["country_name"] = "China"
        geo["region"] = region or "Taiwan"
        geo["city"] = city or geo.get("city")
        geo["source"] = geo.get("source") or "ip-api"
    return geo

def _lookup_geo_ipapi(ip: str) -> Optional[dict]:
    enabled = os.environ.get("ENABLE_GEOIP_LOOKUP", "1").strip().lower() not in ("0", "false", "no", "off")
    if not enabled:
        return None
    if not _is_public_ip(ip):
        return None
    now = time.time()
    cached = _geoip_cache.get(ip)
    if cached:
        ts, val = cached
        if now - float(ts or 0.0) < 86400:
            return val
    url = f"http://ip-api.com/json/{urllib.parse.quote(ip)}?fields=status,message,country,countryCode,regionName,city,lat,lon,query"
    try:
        with urllib.request.urlopen(url, timeout=1.5) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="ignore") or "{}")
    except Exception:
        _geoip_cache[ip] = (now, None)
        return None
    if not isinstance(data, dict) or data.get("status") != "success":
        _geoip_cache[ip] = (now, None)
        return None
    geo = {
        "country_code": data.get("countryCode"),
        "country_name": data.get("country"),
        "region": data.get("regionName"),
        "city": data.get("city"),
        "lat": data.get("lat"),
        "lon": data.get("lon"),
        "source": "ip-api",
    }
    geo = _normalize_geo_policy(geo)
    _geoip_cache[ip] = (now, geo)
    return geo

def _require_internal(request: Request) -> None:
    ip = _peer_ip(request)
    try:
        ipa = ipaddress.ip_address(ip)
    except Exception:
        raise HTTPException(status_code=403, detail="Forbidden")
    if not (ipa.is_loopback or ipa.is_private or ipa.is_link_local):
        raise HTTPException(status_code=403, detail="Forbidden")

def _date_str() -> str:
    return datetime.datetime.now().date().isoformat()

def _make_request_id(date_str: str) -> str:
    compact = date_str.replace("-", "")
    return f"{compact}-{uuid.uuid4()}"

def _date_from_request_id(request_id: str) -> str:
    if not request_id or len(request_id) < 9 or request_id[8] != "-":
        raise ValueError("invalid request_id format")
    compact = request_id[:8]
    if not compact.isdigit():
        raise ValueError("invalid request_id format")
    return f"{compact[:4]}-{compact[4:6]}-{compact[6:8]}"

def _make_request_dir(request_id: str, date_str: str) -> str:
    request_dir = os.path.join(DATA_COLLECTION_DIR, date_str, request_id)
    os.makedirs(request_dir, exist_ok=True)
    return request_dir

app = FastAPI()

# Add CORS middleware to allow requests from other servers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=False,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Setup directories
os.makedirs(DATA_COLLECTION_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "model_display_name": config.MODEL_DISPLAY_NAME})

@app.get("/startup-logs")
async def get_startup_logs():
    return {
        "status": "ready",
        "logs": [
            f"Web server ready. INFERENCE_BACKEND={config.INFERENCE_BACKEND}",
            "Model loading is handled by the selected inference provider.",
        ],
    }

@app.get("/status")
async def get_status():
    return JSONResponse(content=dict(inference_provider.status()))

def _iter_request_json_paths() -> list[str]:
    paths = []
    for root, dirs, files in os.walk(DATA_COLLECTION_DIR):
        if "data.json" in files:
            paths.append(os.path.join(root, "data.json"))
    paths.sort(reverse=True)
    return paths

def _load_request_record(path: str) -> Optional[dict]:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None

def _compute_analytics() -> dict:
    total = 0
    unique_ips = set()
    by_region = Counter()
    by_ip = Counter()
    by_day = Counter()
    feedback = Counter()
    recent = []
    by_point = defaultdict(int)

    for path in _iter_request_json_paths():
        rec = _load_request_record(path)
        if not isinstance(rec, dict):
            continue
        total += 1
        request_id = rec.get("request_id")
        ts = rec.get("timestamp")
        date = rec.get("date") or (ts[:10] if isinstance(ts, str) and len(ts) >= 10 else None)
        if date:
            by_day[date] += 1

        client = rec.get("client") if isinstance(rec.get("client"), dict) else {}
        ip = client.get("ip") or "unknown"
        unique_ips.add(ip)
        by_ip[ip] += 1

        geo = client.get("geo") if isinstance(client.get("geo"), dict) else {}
        country_name = geo.get("country_name") or geo.get("country")
        country_code = geo.get("country_code")
        region = geo.get("region")
        city = geo.get("city")
        lat = geo.get("lat")
        lon = geo.get("lon")
        if not (country_name and region and lat is not None and lon is not None):
            resolved = _lookup_geo_ipapi(ip)
            if resolved:
                country_name = resolved.get("country_name") or country_name
                country_code = resolved.get("country_code") or country_code
                region = resolved.get("region") or region
                city = resolved.get("city") or city
                lat = resolved.get("lat") if resolved.get("lat") is not None else lat
                lon = resolved.get("lon") if resolved.get("lon") is not None else lon

        country_name = (country_name or "Unknown").strip()
        region = (region or "Unknown").strip()
        city = (city or "").strip()
        location_label = f"{country_name} / {region}" if region else country_name
        by_region[location_label] += 1
        if lat is not None and lon is not None and country_name != "Unknown" and region != "Unknown":
            key = (float(lat), float(lon), location_label)
            by_point[key] += 1

        fb = rec.get("feedback")
        if isinstance(fb, str) and fb:
            feedback[fb] += 1

        if len(recent) < 50:
            recent.append({
                "request_id": request_id,
                "timestamp": ts,
                "date": date,
                "ip": ip,
                "country": country_name,
                "region": region,
                "city": city,
                "feedback": fb,
            })

    return {
        "total_requests": total,
        "unique_ips": len(unique_ips),
        "by_region": by_region.most_common(),
        "by_ip": by_ip.most_common(200),
        "by_day": sorted(by_day.items()),
        "feedback": feedback,
        "recent": recent,
        "markers": [
            {"lat": lat, "lon": lon, "label": label, "count": count}
            for (lat, lon, label), count in sorted(by_point.items(), key=lambda x: x[1], reverse=True)[:500]
        ],
    }

_analytics_cache = {"ts": 0.0, "data": None}

@app.get("/admin/analytics")
async def admin_analytics(request: Request):
    _require_internal(request)
    return templates.TemplateResponse("analytics.html", {"request": request})

@app.get("/admin/analytics_data")
async def admin_analytics_data(request: Request):
    _require_internal(request)
    now = time.time()
    if _analytics_cache["data"] is None or now - float(_analytics_cache["ts"] or 0.0) > 10:
        _analytics_cache["data"] = _compute_analytics()
        _analytics_cache["ts"] = now
    return JSONResponse(content=_analytics_cache["data"])

@app.get("/admin/whoami")
async def admin_whoami(request: Request):
    _require_internal(request)
    return JSONResponse(content={
        "peer_ip": _peer_ip(request),
        "client_ip": _client_ip(request),
        "geo": _client_geo(request),
        "x_forwarded_for": request.headers.get("x-forwarded-for"),
        "x_real_ip": request.headers.get("x-real-ip"),
        "cf_connecting_ip": request.headers.get("cf-connecting-ip"),
    })

def _default_request_config(stream: bool = False) -> dict:
    return {
        "temperature": 0.0,
        "max_tokens": 2048,
        "top_p": 1.0,
        "top_k": 0,
        "repetition_penalty": 1.0,
        "stream": stream,
    }

def _build_prompt(prompt_tags: str) -> str:
    return (
        f"{prompt_tags}Interpret the provided ECG image, identify key features "
        "and abnormalities in each lead, and generate a clinical diagnosis that "
        "is supported by the observed evidence."
    )

def _prepare_provider_request(
    image: Optional[UploadFile],
    ecg: Optional[list[UploadFile]],
    request_dir: str,
    stream: bool,
) -> tuple[InferenceRequest, dict]:
    images_list = []
    objects_dict = {}
    prompt_tags = ""
    inputs: dict = {}

    if ecg:
        dat_file = None
        hea_file = None
        for uploaded in ecg:
            fname = _safe_filename(uploaded.filename)
            ext = os.path.splitext(fname)[1].lower()
            path = os.path.join(request_dir, fname)
            with open(path, "wb") as fo:
                shutil.copyfileobj(uploaded.file, fo)

            if ext == ".dat":
                dat_file = path
            elif ext == ".hea":
                hea_file = path
            inputs.setdefault("ecg_files", []).append(fname)

        if not dat_file or not hea_file:
            raise HTTPException(status_code=400, detail="ECG signal requires both .dat and .hea files.")

        record_path = os.path.splitext(hea_file)[0]
        objects_dict["ecg"] = [record_path]
        prompt_tags += "<ecg>"
        inputs["ecg_record"] = os.path.basename(record_path)

    if image and image.filename:
        image_name = _safe_filename(image.filename)
        image_path = os.path.join(request_dir, image_name)
        with open(image_path, "wb") as f:
            shutil.copyfileobj(image.file, f)
        images_list.append(image_path)
        prompt_tags += "<image>"
        inputs["image"] = image_name

    prompt = _build_prompt(prompt_tags)
    provider_request = InferenceRequest(
        messages=[{"role": "user", "content": prompt}],
        images=images_list,
        objects=objects_dict,
        request_config=_default_request_config(stream=stream),
    )
    return provider_request, inputs

def _record_meta(request_config: dict, stream: bool) -> dict:
    meta = {
        "provider": inference_provider.name,
        "model_path": config.MODEL_PATH,
        "model_display_name": config.MODEL_DISPLAY_NAME,
        "ecg_tower_path": config.ECG_TOWER_PATH,
        "request_config": dict(request_config),
    }
    meta["request_config"]["stream"] = stream
    return meta

def _write_request_record(
    request_dir: str,
    request_id: str,
    date_str: str,
    inputs: dict,
    client_ip: str,
    client_geo: dict,
    user_agent: Optional[str],
    content: str,
    reasoning: str,
    request_config: dict,
    stream: bool,
):
    collected_info = {
        "request_id": request_id,
        "timestamp": datetime.datetime.now().isoformat(),
        "date": date_str,
        "inputs": inputs,
        "client": {
            "ip": client_ip,
            "geo": client_geo,
            "user_agent": user_agent,
        },
        "model_output": content,
        "reasoning_output": reasoning,
        "meta_info": _record_meta(request_config, stream=stream),
        "feedback": None,
    }
    with open(os.path.join(request_dir, "data.json"), "w") as f:
        json.dump(collected_info, f, indent=4, ensure_ascii=False)

async def _predict_once_provider(
    request: Request,
    image: Optional[UploadFile],
    ecg: Optional[list[UploadFile]],
):
    if not image and not ecg:
        raise HTTPException(status_code=400, detail="Please provide at least one input (Image or ECG signal).")

    date_str = _date_str()
    request_id = _make_request_id(date_str)
    request_dir = _make_request_dir(request_id, date_str)
    provider_request, inputs = _prepare_provider_request(image, ecg, request_dir, stream=False)
    client_ip = _client_ip(request)
    client_geo = _client_geo(request)

    result = inference_provider.infer(provider_request)
    _write_request_record(
        request_dir=request_dir,
        request_id=request_id,
        date_str=date_str,
        inputs=inputs,
        client_ip=client_ip,
        client_geo=client_geo,
        user_agent=request.headers.get("user-agent"),
        content=result.content,
        reasoning=result.reasoning,
        request_config=provider_request.request_config,
        stream=False,
    )
    return JSONResponse(content={"result": result.content, "request_id": request_id})

def _start_provider_background(
    request_id: str,
    request_dir: str,
    date_str: str,
    client_ip: str,
    client_geo: dict,
    user_agent: Optional[str],
    provider_request: InferenceRequest,
    inputs: dict,
):
    def _run():
        content_buf = ""
        reasoning_buf = ""
        try:
            for chunk in inference_provider.stream(provider_request):
                if chunk.event == "reasoning":
                    stream_states[request_id]["reasoning"] += chunk.text
                    reasoning_buf += chunk.text
                else:
                    stream_states[request_id]["content"] += chunk.text
                    content_buf += chunk.text

            _write_request_record(
                request_dir=request_dir,
                request_id=request_id,
                date_str=date_str,
                inputs=inputs,
                client_ip=client_ip,
                client_geo=client_geo,
                user_agent=user_agent,
                content=content_buf,
                reasoning=reasoning_buf,
                request_config=provider_request.request_config,
                stream=True,
            )
            stream_states[request_id]["done"] = True
        except Exception as exc:
            stream_states[request_id]["error"] = str(exc)
            stream_states[request_id]["done"] = True

    threading.Thread(target=_run, daemon=True).start()

async def _predict_start_provider(
    request: Request,
    image: Optional[UploadFile],
    ecg: Optional[list[UploadFile]],
):
    if not image and not ecg:
        raise HTTPException(status_code=400, detail="Please provide at least one input (Image or ECG signal).")

    date_str = _date_str()
    request_id = _make_request_id(date_str)
    request_dir = _make_request_dir(request_id, date_str)
    client_ip = _client_ip(request)
    client_geo = _client_geo(request)
    provider_request, inputs = _prepare_provider_request(image, ecg, request_dir, stream=True)

    stream_states[request_id] = {
        "started_at": time.time(),
        "request_dir": request_dir,
        "date": date_str,
        "client_ip": client_ip,
        "content": "",
        "reasoning": "",
        "done": False,
        "error": None,
    }
    _start_provider_background(
        request_id=request_id,
        request_dir=request_dir,
        date_str=date_str,
        client_ip=client_ip,
        client_geo=client_geo,
        user_agent=request.headers.get("user-agent"),
        provider_request=provider_request,
        inputs=inputs,
    )
    return {"request_id": request_id}

async def _predict_stream_provider(
    request: Request,
    image: Optional[UploadFile],
    ecg: Optional[list[UploadFile]],
):
    if not image and not ecg:
        raise HTTPException(status_code=400, detail="Please provide at least one input (Image or ECG signal).")

    date_str = _date_str()
    request_id = _make_request_id(date_str)
    request_dir = _make_request_dir(request_id, date_str)
    client_ip = _client_ip(request)
    client_geo = _client_geo(request)
    provider_request, inputs = _prepare_provider_request(image, ecg, request_dir, stream=True)

    stream_states[request_id] = {
        "started_at": time.time(),
        "request_dir": request_dir,
        "date": date_str,
        "client_ip": client_ip,
        "content": "",
        "reasoning": "",
        "done": False,
        "error": None,
    }

    async def event_gen():
        content_buf = ""
        reasoning_buf = ""
        q: "queue.Queue[tuple[str, str]]" = queue.Queue()
        started_at = time.time()
        max_wait_s = config.INFERENCE_TIMEOUT_S + 30

        def _run_infer():
            try:
                for chunk in inference_provider.stream(provider_request):
                    if chunk.event == "reasoning":
                        stream_states[request_id]["reasoning"] += chunk.text
                        q.put(("reasoning", chunk.text))
                    else:
                        stream_states[request_id]["content"] += chunk.text
                        q.put(("content", chunk.text))
                stream_states[request_id]["done"] = True
                q.put(("done", request_id))
            except Exception as exc:
                stream_states[request_id]["error"] = str(exc)
                stream_states[request_id]["done"] = True
                q.put(("error", str(exc)))

        threading.Thread(target=_run_infer, daemon=True).start()

        try:
            yield f"event: ready\ndata: {json.dumps({'request_id': request_id}, ensure_ascii=False)}\n\n"
            while True:
                if time.time() - started_at > max_wait_s:
                    yield f"event: error\ndata: {json.dumps({'detail': f'timeout after {max_wait_s}s'}, ensure_ascii=False)}\n\n"
                    return

                try:
                    event_type, payload = await asyncio.to_thread(q.get, True, 1.0)
                except Exception:
                    yield f"event: ping\ndata: {json.dumps({'t': time.time()}, ensure_ascii=False)}\n\n"
                    continue

                if event_type == "reasoning":
                    reasoning_buf += payload
                    yield f"event: reasoning\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
                elif event_type == "content":
                    content_buf += payload
                    yield f"event: content\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
                elif event_type == "done":
                    break
                elif event_type == "error":
                    yield f"event: error\ndata: {json.dumps({'detail': payload}, ensure_ascii=False)}\n\n"
                    return

            _write_request_record(
                request_dir=request_dir,
                request_id=request_id,
                date_str=date_str,
                inputs=inputs,
                client_ip=client_ip,
                client_geo=client_geo,
                user_agent=request.headers.get("user-agent"),
                content=content_buf,
                reasoning=reasoning_buf,
                request_config=provider_request.request_config,
                stream=True,
            )
            yield f"event: done\ndata: {json.dumps({'request_id': request_id}, ensure_ascii=False)}\n\n"
        except Exception as exc:
            stream_states[request_id]["error"] = str(exc)
            stream_states[request_id]["done"] = True
            yield f"event: error\ndata: {json.dumps({'detail': str(exc)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "X-Request-ID": request_id,
        },
    )

@app.post("/predict")
async def predict(request: Request, image: Optional[UploadFile] = File(None), ecg: list[UploadFile] = File(None)):
    return await _predict_once_provider(request, image, ecg)


@app.post("/predict_start")
async def predict_start(
    request: Request,
    image: Optional[UploadFile] = File(None),
    ecg: list[UploadFile] = File(None),
):
    return await _predict_start_provider(request, image, ecg)


@app.post("/predict_stream")
async def predict_stream(request: Request, image: Optional[UploadFile] = File(None), ecg: list[UploadFile] = File(None)):
    return await _predict_stream_provider(request, image, ecg)


@app.get("/predict_progress/{request_id}")
async def predict_progress(request_id: str):
    state = stream_states.get(request_id)
    if not state:
        raise HTTPException(status_code=404, detail="Request not found")
    return state


@app.post("/feedback")
async def submit_feedback(request: Request, data: dict = Body(...)):
    request_id = data.get("request_id")
    feedback_type = data.get("feedback")  # "like" or "dislike"
    feedback_comment = data.get("comment")
    
    if not request_id or not feedback_type:
        raise HTTPException(status_code=400, detail="Missing request_id or feedback type")
        
    try:
        date_str = _date_from_request_id(request_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid request_id format")

    request_dir = os.path.join(DATA_COLLECTION_DIR, date_str, request_id)

    json_path = os.path.join(request_dir, "data.json")
    
    if not os.path.exists(json_path):
        raise HTTPException(status_code=404, detail="Request data not found")
        
    try:
        with open(json_path, "r") as f:
            record = json.load(f)
            
        record["feedback"] = feedback_type
        record["feedback_at"] = datetime.datetime.now().isoformat()
        record["feedback_client"] = {
            "ip": _client_ip(request),
            "geo": _client_geo(request),
            "user_agent": request.headers.get("user-agent"),
        }
        if isinstance(feedback_comment, str):
            trimmed = feedback_comment.strip()
            record["feedback_comment"] = trimmed if trimmed else None
        
        with open(json_path, "w") as f:
            json.dump(record, f, indent=4, ensure_ascii=False)
            
        return JSONResponse(content={"status": "success", "message": "Feedback recorded"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
