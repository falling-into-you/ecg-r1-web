import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Web/runtime selection. Edit this file for project-local configuration.
INFERENCE_BACKEND = "vllm_direct"  # "mock", "vllm_direct", or "swift_rollout"
INFERENCE_TIMEOUT_S = 300.0

# Direct vLLM inference service used by the vllm_direct provider.
VLLM_HOST = "127.0.0.1"
VLLM_PORT = 8023
VLLM_URL = f"http://{VLLM_HOST}:{VLLM_PORT}/infer/"
VLLM_HEALTH_URL = f"http://{VLLM_HOST}:{VLLM_PORT}/health/"
VLLM_ENFORCE_EAGER = True
VLLM_GPU_MEMORY_UTILIZATION = 0.9
VLLM_MAX_MODEL_LEN = 8192
VLLM_MAX_NUM_SEQS = 8
VLLM_LOAD_FORMAT = "auto"

# Legacy aliases used by older scripts or docs.
SWIFT_ROLLOUT_HOST = VLLM_HOST
SWIFT_ROLLOUT_PORT = VLLM_PORT
SWIFT_ROLLOUT_URL = VLLM_URL
SWIFT_ROLLOUT_HEALTH_URL = VLLM_HEALTH_URL

# Display and persistence.
MODEL_DISPLAY_NAME = "ECG-R1-8B-0129"
MODEL_PATH = "/data/jinjiarui/run/ecg-r1/training/ecg-r1-8b-dapo/v16-20251227-163009/checkpoint-493"
DATA_COLLECTION_DIR = os.path.join(BASE_DIR, "data_collection")

# Python environment used by startup scripts. Leave *_CONDA_ENV empty to use the
# current shell environment.
CONDA_ACTIVATE_SCRIPT = "/home/jinjiarui/miniconda3/bin/activate"
WEB_CONDA_ENV = "swift2"
VLLM_CONDA_ENV = "swift2"
ROLLOUT_CONDA_ENV = VLLM_CONDA_ENV

# Web service.
WEB_HOST = "0.0.0.0"
WEB_PORT = 8000
WEB_LOG_LEVEL = "info"
DEFAULT_MAX_TOKENS = 1024

# Runtime values consumed by ecg_r1_runtime when starting direct vLLM.
# Physical GPU index for vLLM. With this set, the process sees GPU 3 as cuda:0.
CUDA_VISIBLE_DEVICES = "3"
VLLM_DATA_PARALLEL_SIZE = 1
PYTORCH_ALLOC_CONF = "expandable_segments:True"
ECG_TOWER_PATH = "/data/jinjiarui/run/ecg-r1/ecg_coca/checkpoint/cpt_wfep_epoch_20.pt"
ECG_SEQ_LENGTH = "5000"
ECG_PATCH_SIZE = "50"
IMAGE_MIN_TOKEN_NUM = "4"
IMAGE_MAX_TOKEN_NUM = "768"
ECG_PROJECTOR_TYPE = "mlp2x_gelu"
ECG_MODEL_CONFIG = "coca_ViT-B-32"
ROOT_ECG_DIR = "/"
ROOT_IMAGE_DIR = "/"
FREEZE_ECG_TOWER = "True"
FREEZE_ECG_PROJECTOR = "True"

RUNTIME_ENV_VARS = {
    "CUDA_VISIBLE_DEVICES": CUDA_VISIBLE_DEVICES,
    "PYTORCH_ALLOC_CONF": PYTORCH_ALLOC_CONF,
    "ECG_SEQ_LENGTH": ECG_SEQ_LENGTH,
    "ECG_PATCH_SIZE": ECG_PATCH_SIZE,
    "IMAGE_MIN_TOKEN_NUM": IMAGE_MIN_TOKEN_NUM,
    "ROOT_ECG_DIR": ROOT_ECG_DIR,
    "ROOT_IMAGE_DIR": ROOT_IMAGE_DIR,
    "IMAGE_MAX_TOKEN_NUM": IMAGE_MAX_TOKEN_NUM,
    "ECG_TOWER_PATH": ECG_TOWER_PATH,
    "ECG_PROJECTOR_TYPE": ECG_PROJECTOR_TYPE,
    "ECG_MODEL_CONFIG": ECG_MODEL_CONFIG,
    "FREEZE_ECG_TOWER": FREEZE_ECG_TOWER,
    "FREEZE_ECG_PROJECTOR": FREEZE_ECG_PROJECTOR,
}
