import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Web/runtime selection. The web process never imports ECG-R1 source code.
INFERENCE_BACKEND = os.environ.get("INFERENCE_BACKEND", "mock").strip()
INFERENCE_TIMEOUT_S = float(os.environ.get("INFERENCE_TIMEOUT_S", "300"))

# Swift rollout service used by the swift_rollout provider.
SWIFT_ROLLOUT_URL = os.environ.get("SWIFT_ROLLOUT_URL", "http://127.0.0.1:8023/infer/")
SWIFT_ROLLOUT_HEALTH_URL = os.environ.get("SWIFT_ROLLOUT_HEALTH_URL", "http://127.0.0.1:8023/health/")

# Display and persistence.
MODEL_DISPLAY_NAME = os.environ.get("MODEL_DISPLAY_NAME", "ECG-R1")
MODEL_PATH = os.environ.get("MODEL_PATH", "")
DATA_COLLECTION_DIR = os.environ.get("DATA_COLLECTION_DIR", os.path.join(BASE_DIR, "data_collection"))

# Runtime values consumed by ecg_r1_runtime when starting a rollout process.
CUDA_DEVICE = os.environ.get("CUDA_DEVICE", "0")
ECG_TOWER_PATH = os.environ.get(
    "ECG_TOWER_PATH",
    os.path.join(BASE_DIR, "checkpoints", "cpt_wfep_epoch_20.pt"),
)
ECG_SEQ_LENGTH = os.environ.get("ECG_SEQ_LENGTH", "5000")
ECG_PATCH_SIZE = os.environ.get("ECG_PATCH_SIZE", "50")
IMAGE_MAX_TOKEN_NUM = os.environ.get("IMAGE_MAX_TOKEN_NUM", "768")
ECG_PROJECTOR_TYPE = os.environ.get("ECG_PROJECTOR_TYPE", "mlp2x_gelu")
ECG_MODEL_CONFIG = os.environ.get("ECG_MODEL_CONFIG", "coca_ViT-B-32")

RUNTIME_ENV_VARS = {
    "CUDA_VISIBLE_DEVICES": CUDA_DEVICE,
    "ECG_SEQ_LENGTH": ECG_SEQ_LENGTH,
    "ECG_PATCH_SIZE": ECG_PATCH_SIZE,
    "ROOT_ECG_DIR": os.environ.get("ROOT_ECG_DIR", "/"),
    "ROOT_IMAGE_DIR": os.environ.get("ROOT_IMAGE_DIR", "/"),
    "IMAGE_MAX_TOKEN_NUM": IMAGE_MAX_TOKEN_NUM,
    "ECG_TOWER_PATH": ECG_TOWER_PATH,
    "ECG_PROJECTOR_TYPE": ECG_PROJECTOR_TYPE,
    "ECG_MODEL_CONFIG": ECG_MODEL_CONFIG,
    "FREEZE_ECG_TOWER": os.environ.get("FREEZE_ECG_TOWER", "True"),
    "FREEZE_ECG_PROJECTOR": os.environ.get("FREEZE_ECG_PROJECTOR", "True"),
}
