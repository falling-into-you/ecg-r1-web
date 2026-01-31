import os

# Base paths
ECG_R1_ROOT = "/data/jinjiarui/run/ecg-r1"
ECG_TOWER_PATH = os.path.join(ECG_R1_ROOT, "ecg_coca/checkpoint/cpt_wfep_epoch_20.pt")
# NOTE: Verify this path exists. If not, update it to the correct checkpoint path.
MODEL_PATH = os.path.join(ECG_R1_ROOT, "training/ecg-r1-8b/v2-20251129-121933/checkpoint-15000") 

# Environment variables required by the model
ENV_VARS = {
    "CUDA_VISIBLE_DEVICES": "0",
    "ECG_SEQ_LENGTH": "5000",
    "ECG_PATCH_SIZE": "50",
    "ROOT_ECG_DIR": "/",  # We will provide absolute paths, so root can be /
    "ROOT_IMAGE_DIR": "/", # We will provide absolute paths
    "IMAGE_MAX_TOKEN_NUM": "768",
    "ECG_TOWER_PATH": ECG_TOWER_PATH,
    "ECG_PROJECTOR_TYPE": "mlp2x_gelu",
    "ECG_MODEL_CONFIG": "coca_ViT-B-32",
    "FREEZE_ECG_TOWER": "True",
    "FREEZE_ECG_PROJECTOR": "False",
}

UPLOAD_DIR = "/data/jinjiarui/run/ecg-r1-web/uploads"
