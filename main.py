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
import glob
import ipaddress
from contextlib import asynccontextmanager

# Force flush stdout
sys.stdout.reconfigure(line_buffering=True)
from typing import Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
import json

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
    return {
        "country": country,
        "region": region,
        "city": city,
        "source": "headers" if any([country, region, city]) else None,
    }

def _date_str() -> str:
    return datetime.datetime.now().date().isoformat()

def _make_request_dir(request_id: str, date_str: str) -> str:
    request_dir = os.path.join(DATA_COLLECTION_DIR, date_str, request_id)
    os.makedirs(request_dir, exist_ok=True)
    return request_dir

def _find_request_dir(request_id: str) -> str | None:
    legacy = os.path.join(DATA_COLLECTION_DIR, request_id)
    if os.path.isdir(legacy):
        return legacy
    matches = glob.glob(os.path.join(DATA_COLLECTION_DIR, "*", request_id))
    for m in matches:
        if os.path.isdir(m):
            return m
    return None

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
    if engine is None:
        return JSONResponse(content={"status": "offline", "detail": "Model not loaded"})
    return JSONResponse(content={"status": "online", "detail": "System ready"})

@app.post("/predict")
async def predict(request: Request, image: Optional[UploadFile] = File(None), ecg: Optional[UploadFile] = File(None)):
    if engine is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    if not image and not ecg:
        raise HTTPException(status_code=400, detail="Please provide at least one input (Image or ECG signal).")
    
    try:
        from swift.llm import InferRequest, RequestConfig
        
        request_id = str(uuid.uuid4())
        date_str = _date_str()
        request_dir = _make_request_dir(request_id, date_str)

        inputs = {}
        images_list = []
        objects_dict = {}
        prompt_tags = ""
        
        # Handle ECG file
        if ecg and ecg.filename:
            ecg_name = _safe_filename(ecg.filename)
            ecg_path = os.path.join(request_dir, ecg_name)
            with open(ecg_path, "wb") as f:
                shutil.copyfileobj(ecg.file, f)
            objects_dict['ecg'] = [ecg_path]
            prompt_tags += "<ecg>"
            inputs["ecg"] = ecg_name
            
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
async def predict_stream(request: Request, image: Optional[UploadFile] = File(None), ecg: Optional[UploadFile] = File(None)):
    if engine is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    if not image and not ecg:
        raise HTTPException(status_code=400, detail="Please provide at least one input (Image or ECG signal).")

    from swift.llm import InferRequest, RequestConfig

    request_id = str(uuid.uuid4())
    date_str = _date_str()
    request_dir = _make_request_dir(request_id, date_str)
    client_ip = _client_ip(request)
    client_geo = _client_geo(request)

    images_list = []
    objects_dict = {}
    prompt_tags = ""
    inputs = {}

    if ecg and ecg.filename:
        ecg_name = _safe_filename(ecg.filename)
        ecg_path = os.path.join(request_dir, ecg_name)
        with open(ecg_path, "wb") as f:
            shutil.copyfileobj(ecg.file, f)
        objects_dict["ecg"] = [ecg_path]
        prompt_tags += "<ecg>"
        inputs["ecg"] = ecg_name

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
    
    if not request_id or not feedback_type:
        raise HTTPException(status_code=400, detail="Missing request_id or feedback type")
        
    request_dir = _find_request_dir(request_id)
    if not request_dir:
        raise HTTPException(status_code=404, detail="Request data not found")

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
        
        with open(json_path, "w") as f:
            json.dump(record, f, indent=4, ensure_ascii=False)
            
        return JSONResponse(content={"status": "success", "message": "Feedback recorded"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
