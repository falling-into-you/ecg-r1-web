import sys
import os
import shutil
import importlib.util
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
import config

# Setup paths and env vars
sys.path.insert(0, config.ECG_R1_ROOT)
for k, v in config.ENV_VARS.items():
    os.environ[k] = v

# Global engine variable
engine = None
processor = None
template = None

def load_custom_register():
    # Load my_register_v3.py as done in test_inference.py
    register_path = os.path.join(config.ECG_R1_ROOT, "ecg_r1/my_register_v3.py")
    if not os.path.exists(register_path):
        print(f"Warning: Register file not found at {register_path}")
        return
        
    spec = importlib.util.spec_from_file_location("my_register", register_path)
    my_register = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(my_register)
    print("Custom register loaded.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine, processor, template
    print("Loading model...")
    try:
        # Import swift modules here to ensure env vars are set
        from swift.llm import PtEngine, get_model_tokenizer, get_template
        
        load_custom_register()
        
        # Check if model path exists
        if not os.path.exists(config.MODEL_PATH):
            print(f"Error: Model path {config.MODEL_PATH} does not exist.")
            # We don't raise error here to allow server to start, but inference will fail
        else:
            model, processor = get_model_tokenizer(config.MODEL_PATH, model_type='ecg_r1', attn_impl='flash_attention_2')
            template = get_template('ecg_r1', processor)
            engine = PtEngine.from_model_template(model, template, max_batch_size=1)
            print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")
    
    yield
    print("Shutting down...")

app = FastAPI(lifespan=lifespan)

# Setup directories
os.makedirs(config.UPLOAD_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/status")
async def get_status():
    if engine is None:
        return JSONResponse(content={"status": "offline", "detail": "Model not loaded"})
    return JSONResponse(content={"status": "online", "detail": "System ready"})

@app.post("/predict")
async def predict(image: Optional[UploadFile] = File(None), ecg: Optional[UploadFile] = File(None)):
    if engine is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    if not image and not ecg:
        raise HTTPException(status_code=400, detail="Please provide at least one input (Image or ECG signal).")
    
    try:
        from swift.llm import InferRequest, RequestConfig
        
        images_list = []
        objects_dict = {}
        prompt_tags = ""
        
        # Handle ECG file
        if ecg:
            ecg_path = os.path.join(config.UPLOAD_DIR, ecg.filename)
            with open(ecg_path, "wb") as f:
                shutil.copyfileobj(ecg.file, f)
            objects_dict['ecg'] = [ecg_path]
            prompt_tags += "<ecg>"
            
        # Handle Image file
        if image:
            image_path = os.path.join(config.UPLOAD_DIR, image.filename)
            with open(image_path, "wb") as f:
                shutil.copyfileobj(image.file, f)
            images_list.append(image_path)
            prompt_tags += "<image>"
            
        # Construct prompt
        prompt = f"{prompt_tags}\nInterpret the provided data, identify key features and abnormalities, and generate a clinical diagnosis that is supported by the observed evidence."
        
        infer_request = InferRequest(
            messages=[{'role': 'user', 'content': prompt}],
            images=images_list,
            objects=objects_dict
        )
        
        request_config = RequestConfig(temperature=0.0, max_tokens=2048, top_p=0, top_k=0, repetition_penalty=1.0)
        
        resp_list = engine.infer([infer_request], request_config)
        result_text = resp_list[0].choices[0].message.content
        
        # --- Data Collection ---
        request_id = str(uuid.uuid4())
        request_dir = os.path.join(DATA_COLLECTION_DIR, request_id)
        os.makedirs(request_dir, exist_ok=True)
        
        collected_info = {
            "request_id": request_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "inputs": {},
            "model_output": result_text,
            "meta_info": {
                "model_path": config.MODEL_PATH,
                "ecg_tower_path": config.ECG_TOWER_PATH
            },
            "feedback": None
        }

        # Save input files to collection dir
        if image:
            saved_image_path = os.path.join(request_dir, image.filename)
            shutil.copy2(image_path, saved_image_path)
            collected_info["inputs"]["image"] = image.filename
            
        if ecg:
            saved_ecg_path = os.path.join(request_dir, ecg.filename)
            shutil.copy2(ecg_path, saved_ecg_path)
            collected_info["inputs"]["ecg"] = ecg.filename
            
        # Save JSON data
        with open(os.path.join(request_dir, "data.json"), "w") as f:
            json.dump(collected_info, f, indent=4, ensure_ascii=False)
            
        return JSONResponse(content={"result": result_text, "request_id": request_id})
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback")
async def submit_feedback(data: dict = Body(...)):
    request_id = data.get("request_id")
    feedback_type = data.get("feedback")  # "like" or "dislike"
    
    if not request_id or not feedback_type:
        raise HTTPException(status_code=400, detail="Missing request_id or feedback type")
        
    request_dir = os.path.join(DATA_COLLECTION_DIR, request_id)
    json_path = os.path.join(request_dir, "data.json")
    
    if not os.path.exists(json_path):
        raise HTTPException(status_code=404, detail="Request data not found")
        
    try:
        with open(json_path, "r") as f:
            record = json.load(f)
            
        record["feedback"] = feedback_type
        
        with open(json_path, "w") as f:
            json.dump(record, f, indent=4, ensure_ascii=False)
            
        return JSONResponse(content={"status": "success", "message": "Feedback recorded"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
