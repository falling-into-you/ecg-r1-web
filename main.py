import sys
import os
import config
from config import DATA_COLLECTION_DIR
sys.path.insert(0, config.ECG_R1_ROOT)
for k, v in config.ENV_VARS.items():
    os.environ[k] = v

import shutil
import importlib.util
import asyncio
import queue
import torch
import threading
import datetime
import uuid
import time
import ipaddress
import urllib.request
import urllib.parse
from contextlib import asynccontextmanager

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

# Global engine variable
engine = None
processor = None
template = None
_cuda_probe_tensor = None

# Loading state
loading_logs = []
model_loading_status = "pending" # pending, loading, success, failed
stream_states = {}

def _safe_filename(name: str) -> str:
    base = os.path.basename(str(name or "")).strip()
    base = base.replace("/", "_").replace("\\", "_").replace("\x00", "")
    return base or "file"

def _client_ip(request: Request) -> str:
    for key in ("cf-connecting-ip", "true-client-ip", "x-real-ip"):
        raw = request.headers.get(key)
        if raw:
            ip = raw.strip()
            try:
                ipaddress.ip_address(ip)
                return ip
            except Exception:
                pass
    xff = request.headers.get("x-forwarded-for")
    if xff:
        for part in xff.split(","):
            ip = part.strip()
            if not ip:
                continue
            try:
                ipa = ipaddress.ip_address(ip)
            except Exception:
                continue
            if ipa.is_private or ipa.is_loopback or ipa.is_link_local or ipa.is_reserved:
                continue
            return ip
        first = xff.split(",")[0].strip()
        if first:
            return first
    if request.client and request.client.host:
        return request.client.host
    return "unknown"

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
    ip = _client_ip(request)
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

def add_log(msg):
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    log_entry = f"[{timestamp}] {msg}"
    loading_logs.append(log_entry)
    print(log_entry)

def load_custom_register():
    # Load LOCAL my_register_v3.py
    register_path = os.path.join(os.getcwd(), "my_register_v3.py")
    if not os.path.exists(register_path):
        add_log(f"Warning: Register file not found at {register_path}")
        return
        
    spec = importlib.util.spec_from_file_location("my_register", register_path)
    my_register = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(my_register)
    add_log("Custom register loaded (local version).")

def load_model_background():
    global engine, processor, template, model_loading_status
    model_loading_status = "loading"
    add_log("Initializing system...")
    
    try:
        # Setup env vars
        for k, v in config.ENV_VARS.items():
            os.environ[k] = v
        
        add_log(f"DEBUG: ECG_TOWER_PATH={os.environ.get('ECG_TOWER_PATH')}")
        add_log(f"DEBUG: ROOT_ECG_DIR={os.environ.get('ROOT_ECG_DIR')}")
        add_log(f"Environment variables set. CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
        add_log(f"PID={os.getpid()}")
        add_log(f"torch.version.cuda={getattr(torch.version, 'cuda', None)}")
        add_log(f"torch.cuda.is_available={torch.cuda.is_available()}")
        add_log(f"torch.cuda.device_count={torch.cuda.device_count()}")
        if torch.cuda.is_available():
            try:
                current_idx = torch.cuda.current_device()
                add_log(f"torch.cuda.current_device={current_idx}")
                add_log(f"torch.cuda.get_device_name={torch.cuda.get_device_name(current_idx)}")
            except Exception as e:
                add_log(f"CUDA device query failed: {e}")
            global _cuda_probe_tensor
            try:
                _cuda_probe_tensor = torch.empty(1, device="cuda")
                add_log("CUDA probe allocation ok")
            except Exception as e:
                add_log(f"CUDA probe allocation failed: {e}")

        add_log("Importing swift modules...")
        # Import swift modules here to ensure env vars are set
        from swift.llm import PtEngine, get_model_tokenizer, get_template
        add_log("Swift modules imported.")
        
        load_custom_register()
        
        # Check if model path exists
        if not os.path.exists(config.MODEL_PATH):
            add_log(f"Error: Model path {config.MODEL_PATH} does not exist.")
            model_loading_status = "failed"
            return
        
        add_log(f"Loading model from {config.MODEL_PATH} with bfloat16...")
        model, processor = get_model_tokenizer(
            config.MODEL_PATH, 
            model_type='ecg_r1', 
            torch_dtype=torch.bfloat16
        )
        add_log("Model and tokenizer loaded into memory.")
        try:
            hf_device_map = getattr(model, "hf_device_map", None)
            if hf_device_map is not None:
                add_log(f"hf_device_map={hf_device_map}")
            first_param = next(model.parameters(), None)
            if first_param is not None:
                add_log(f"first_param_device={first_param.device}")
        except Exception as e:
            add_log(f"Model device inspection failed: {e}")
        
        template = get_template('ecg_r1', processor)
        engine = PtEngine.from_model_template(model, template, max_batch_size=1)
        add_log("Inference engine initialized. System Ready.")
        if torch.cuda.is_available():
            try:
                allocated = torch.cuda.memory_allocated() / (1024 ** 2)
                reserved = torch.cuda.memory_reserved() / (1024 ** 2)
                add_log(f"cuda.memory_allocated_mib={allocated:.1f}")
                add_log(f"cuda.memory_reserved_mib={reserved:.1f}")
            except Exception as e:
                add_log(f"CUDA memory stats failed: {e}")
        
        model_loading_status = "success"
        
    except Exception as e:
        import traceback
        err = traceback.format_exc()
        add_log(f"Critical Error: {str(e)}")
        add_log(err)
        model_loading_status = "failed"

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start loading in background thread
    thread = threading.Thread(target=load_model_background)
    thread.start()
    yield
    print("Shutting down...")

app = FastAPI(lifespan=lifespan)

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
    if model_loading_status == "success":
        return templates.TemplateResponse("index.html", {"request": request, "model_display_name": config.MODEL_DISPLAY_NAME})
    else:
        return templates.TemplateResponse("loading.html", {"request": request})

@app.get("/startup-logs")
async def get_startup_logs():
    return {"status": model_loading_status, "logs": loading_logs}

@app.get("/status")
async def get_status():
    if model_loading_status in ("pending", "loading"):
        return JSONResponse(content={"status": "loading", "detail": f"Model loading ({model_loading_status})", "model_loading_status": model_loading_status})
    if model_loading_status == "failed":
        return JSONResponse(content={"status": "offline", "detail": "Model failed to load", "model_loading_status": model_loading_status})
    if engine is None:
        return JSONResponse(content={"status": "offline", "detail": "Model not loaded", "model_loading_status": model_loading_status})
    return JSONResponse(content={"status": "online", "detail": "System ready", "model_loading_status": model_loading_status})

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

@app.post("/predict")
async def predict(request: Request, image: Optional[UploadFile] = File(None), ecg: list[UploadFile] = File(None)):
    if engine is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    if not image and not ecg:
        raise HTTPException(status_code=400, detail="Please provide at least one input (Image or ECG signal).")

    from swift.llm import InferRequest, RequestConfig

    date_str = _date_str()
    request_id = _make_request_id(date_str)
    request_dir = _make_request_dir(request_id, date_str)
    
    try:
        images_list = []
        objects_dict = {}
        prompt_tags = ""
        inputs = {}
        
        # Handle ECG files
        if ecg:
            # Expect .dat and .hea
            dat_file = None
            hea_file = None
            for f in ecg:
                fname = _safe_filename(f.filename)
                ext = os.path.splitext(fname)[1].lower()
                path = os.path.join(request_dir, fname)
                with open(path, "wb") as fo:
                    shutil.copyfileobj(f.file, fo)
                
                if ext == '.dat':
                    dat_file = path
                elif ext == '.hea':
                    hea_file = path
                
                # Also record in inputs for logging
                if "ecg_files" not in inputs:
                    inputs["ecg_files"] = []
                inputs["ecg_files"].append(fname)

            if dat_file and hea_file:
                # Use .hea path for objects_dict as wfdb.rdsamp expects header path (without extension usually works, but here we pass .hea and let loader handle or pass base)
                # Actually wfdb rdsamp expects the record name (without extension).
                # But our custom loader might expect the full path to .hea or .dat?
                # Let's check my_register_v3.py load_ecg. It uses wfdb.rdsamp(path).
                # wfdb.rdsamp(path) where path is /path/to/record (no extension) usually works if both .dat and .hea exist.
                
                # We will pass the common prefix (record name) to the engine
                # Assuming they share the same basename.
                base_dat = os.path.splitext(dat_file)[0]
                base_hea = os.path.splitext(hea_file)[0]
                
                if base_dat != base_hea:
                    # If basenames differ, we might have issues if wfdb expects them to match.
                    # For now, we assume user uploaded matching pair.
                    pass

                # Pass the record path (without extension) to the engine
                # But wait, my_register_v3.py: load_ecg calls wfdb.rdsamp(path).
                # If path has extension, wfdb might handle it or fail.
                # Standard wfdb.rdsamp arg is 'record_name'.
                
                # Let's pass the record name (without extension).
                # Ensure we strip extension.
                record_path = base_hea 
                
                objects_dict['ecg'] = [record_path]
                prompt_tags += "<ecg>"
                inputs["ecg_record"] = os.path.basename(record_path)
            else:
                 # If only one provided or mismatch, we might skip or fail? 
                 # For now if we don't have a pair, we just don't add to objects_dict?
                 # Or we try our best.
                 pass
            
        # Handle Image file
        if image and image.filename:
            image_name = _safe_filename(image.filename)
            image_path = os.path.join(request_dir, image_name)
            with open(image_path, "wb") as f:
                shutil.copyfileobj(image.file, f)
            images_list.append(image_path)
            prompt_tags += "<image>"
            inputs["image"] = image_name
            
        # Construct prompt
        prompt = f"{prompt_tags}nterpret the provided ECG image, identify key features and abnormalities in each lead, and generate a clinical diagnosis that is supported by the observed evidence."
        
        infer_request = InferRequest(
            messages=[{'role': 'user', 'content': prompt}],
            images=images_list,
            objects=objects_dict
        )
        
        request_config = RequestConfig(temperature=0.0, max_tokens=2048, top_p=0, top_k=0, repetition_penalty=1.0)
        
        resp_list = engine.infer([infer_request], request_config)
        result_text = resp_list[0].choices[0].message.content
        
        # --- Data Collection ---
        client_ip = _client_ip(request)
        client_geo = _client_geo(request)

        collected_info = {
            "request_id": request_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "date": date_str,
            "inputs": inputs,
            "client": {
                "ip": client_ip,
                "geo": client_geo,
                "user_agent": request.headers.get("user-agent"),
            },
            "model_output": result_text,
            "meta_info": {
                "model_path": config.MODEL_PATH,
                "model_display_name": config.MODEL_DISPLAY_NAME,
                "ecg_tower_path": config.ECG_TOWER_PATH,
                "request_config": {
                    "temperature": 0.0,
                    "max_tokens": 2048,
                    "top_p": 0,
                    "top_k": 0,
                    "repetition_penalty": 1.0,
                    "stream": False,
                },
            },
            "feedback": None
        }
            
        # Save JSON data
        with open(os.path.join(request_dir, "data.json"), "w") as f:
            json.dump(collected_info, f, indent=4, ensure_ascii=False)
            
        return JSONResponse(content={"result": result_text, "request_id": request_id})
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_stream")
async def predict_stream(request: Request, image: Optional[UploadFile] = File(None), ecg: list[UploadFile] = File(None)):
    if engine is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    if not image and not ecg:
        raise HTTPException(status_code=400, detail="Please provide at least one input (Image or ECG signal).")

    from swift.llm import InferRequest, RequestConfig

    date_str = _date_str()
    request_id = _make_request_id(date_str)
    request_dir = _make_request_dir(request_id, date_str)
    client_ip = _client_ip(request)
    client_geo = _client_geo(request)

    images_list = []
    objects_dict = {}
    prompt_tags = ""
    inputs = {}

    if ecg:
        dat_file = None
        hea_file = None
        for f in ecg:
            fname = _safe_filename(f.filename)
            ext = os.path.splitext(fname)[1].lower()
            path = os.path.join(request_dir, fname)
            with open(path, "wb") as fo:
                shutil.copyfileobj(f.file, fo)
            
            if ext == '.dat':
                dat_file = path
            elif ext == '.hea':
                hea_file = path
            
            if "ecg_files" not in inputs:
                inputs["ecg_files"] = []
            inputs["ecg_files"].append(fname)
        
        if dat_file and hea_file:
            base_hea = os.path.splitext(hea_file)[0]
            record_path = base_hea
            objects_dict['ecg'] = [record_path]
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

    prompt = f"{prompt_tags}Interpret the provided ECG image, identify key features and abnormalities in each lead, and generate a clinical diagnosis that is supported by the observed evidence."
    infer_request = InferRequest(
        messages=[{"role": "user", "content": prompt}],
        images=images_list,
        objects=objects_dict,
    )
    request_config = RequestConfig(temperature=0.0, max_tokens=2048, top_p=0, top_k=0, repetition_penalty=1.0, stream=True)

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
        max_wait_s = 600

        def _run_infer():
            try:
                resp_list = engine.infer([infer_request], request_config)
                item = resp_list[0] if resp_list else None

                if item is not None and hasattr(item, "__iter__") and not hasattr(item, "choices"):
                    for chunk in item:
                        for choice in getattr(chunk, "choices", []) or []:
                            delta = getattr(choice, "delta", None)
                            if delta is None:
                                continue
                            rc = getattr(delta, "reasoning_content", None)
                            if rc:
                                stream_states[request_id]["reasoning"] += rc
                                q.put(("reasoning", rc))
                            c = getattr(delta, "content", None)
                            if c:
                                stream_states[request_id]["content"] += c
                                q.put(("content", c))
                else:
                    result_text = getattr(getattr(getattr(item, "choices", [None])[0], "message", None), "content", "") if item is not None else ""
                    if result_text:
                        chunk_size = 64
                        for i in range(0, len(result_text), chunk_size):
                            chunk = result_text[i:i + chunk_size]
                            stream_states[request_id]["content"] += chunk
                            q.put(("content", chunk))

                stream_states[request_id]["done"] = True
                q.put(("done", request_id))
            except Exception as e:
                stream_states[request_id]["error"] = str(e)
                q.put(("error", str(e)))

        thread = threading.Thread(target=_run_infer, daemon=True)
        thread.start()

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

            collected_info = {
                "request_id": request_id,
                "timestamp": datetime.datetime.now().isoformat(),
                "date": date_str,
                "inputs": inputs,
                "model_output": content_buf,
                "reasoning_output": reasoning_buf,
                "client": {
                    "ip": client_ip,
                    "geo": client_geo,
                    "user_agent": request.headers.get("user-agent"),
                },
                "meta_info": {
                    "model_path": config.MODEL_PATH,
                    "model_display_name": config.MODEL_DISPLAY_NAME,
                    "ecg_tower_path": config.ECG_TOWER_PATH,
                    "request_config": {
                        "temperature": 0.0,
                        "max_tokens": 2048,
                        "top_p": 0,
                        "top_k": 0,
                        "repetition_penalty": 1.0,
                        "stream": True,
                    },
                },
                "feedback": None,
            }

            with open(os.path.join(request_dir, "data.json"), "w") as f:
                json.dump(collected_info, f, indent=4, ensure_ascii=False)

            yield f"event: done\ndata: {json.dumps({'request_id': request_id}, ensure_ascii=False)}\n\n"
        except Exception as e:
            stream_states[request_id]["error"] = str(e)
            yield f"event: error\ndata: {json.dumps({'detail': str(e)}, ensure_ascii=False)}\n\n"

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
