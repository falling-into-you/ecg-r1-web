import os

# Base paths
ECG_R1_ROOT = "/data/jinjiarui/run/ecg-r1"
# Hardcoded absolute path as requested to avoid any relative path issues
ECG_TOWER_PATH = "/data/jinjiarui/run/ecg-r1/ecg_coca/checkpoint/cpt_wfep_epoch_20.pt"
# NOTE: Verify this path exists. If not, update it to the correct checkpoint path.
MODEL_PATH = "/data/jinjiarui/run/ecg-r1/training/ecg-r1-8b-dapo/v16-20251227-163009/checkpoint-493"
MODEL_DISPLAY_NAME = "ECG-R1-8B-0131"
# CUDA Configuration
CUDA_DEVICE = "0"

# Environment variables required by the model
ENV_VARS = {
    "CUDA_VISIBLE_DEVICES": CUDA_DEVICE,
    "ECG_SEQ_LENGTH": "5000",
    "ECG_PATCH_SIZE": "50",
    "ROOT_ECG_DIR": "/",  # We will provide absolute paths, so root can be /
    "ROOT_IMAGE_DIR": "/", # We will provide absolute paths
    "IMAGE_MAX_TOKEN_NUM": "768",
    "ECG_TOWER_PATH": ECG_TOWER_PATH,
    "ECG_PROJECTOR_TYPE": "mlp2x_gelu",
    "ECG_MODEL_CONFIG": "coca_ViT-B-32",
    "FREEZE_ECG_TOWER": "True",
    "FREEZE_ECG_PROJECTOR": "True",
}

UPLOAD_DIR = "/data/jinjiarui/run/ecg-r1-web/uploads"
DATA_COLLECTION_DIR = "/data/jinjiarui/run/ecg-r1-web/data_collection"
