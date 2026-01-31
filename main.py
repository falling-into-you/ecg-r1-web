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

@app.post("/predict")
async def predict(image: UploadFile = File(...), ecg: UploadFile = File(...)):
    if engine is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        from swift.llm import InferRequest, RequestConfig
        
        # Save files
        image_path = os.path.join(config.UPLOAD_DIR, image.filename)
        ecg_path = os.path.join(config.UPLOAD_DIR, ecg.filename)
        
        with open(image_path, "wb") as f:
            shutil.copyfileobj(image.file, f)
        with open(ecg_path, "wb") as f:
            shutil.copyfileobj(ecg.file, f)
            
        # Construct input
        # Note: The model expects specific object format
        # objects: {'ecg': [path]}
        # messages content: <ecg><image>\nPrompt...
        
        prompt = "<ecg><image>\nInterpret the provided ECG image, identify key features and abnormalities in each lead, and generate a clinical diagnosis that is supported by the observed evidence."
        
        # The engine expects absolute paths or paths relative to ROOT_IMAGE_DIR/ROOT_ECG_DIR
        # We set ROOT dirs to "/" in config, so we can use absolute paths.
        
        infer_request = InferRequest(
            messages=[{'role': 'user', 'content': prompt}],
            images=[image_path],
            objects={'ecg': [ecg_path]}
        )
        
        request_config = RequestConfig(temperature=0.0, max_tokens=2048, top_p=0, top_k=0, repetition_penalty=1.0)
        
        resp_list = engine.infer([infer_request], request_config)
        result_text = resp_list[0].choices[0].message.content
        
        return JSONResponse(content={"result": result_text})
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
