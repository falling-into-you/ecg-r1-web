# ECG-R1 Web Frontend

This is a web frontend for the ECG-R1 multimodal model, providing an interface for diagnosing ECG images and signals.

## 必读文档（开发流程要求）
- 开始任何开发/修复前：先阅读 [PROJECT.md](file:///data/jinjiarui/run/ecg-r1-web/PROJECT.md)
- 每次功能更新后必须：本地验证 → git commit → git push → 更新 PROJECT.md（已实现/待实现/版本记录）

## Setup

1.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    pip install -e .
    ```
2.  For vLLM rollout, install runtime extras in the serving environment if needed:
    ```bash
    pip install -e ".[runtime]"
    ```

## Running the Application

Start the FastAPI server with the mock provider:

```bash
INFERENCE_BACKEND=mock uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Access the web interface at `http://localhost:8000`.

Run with a separate Swift rollout service:

```bash
MODEL_PATH=/path/to/ecg-r1-checkpoint \
ECG_TOWER_PATH=/path/to/cpt_wfep_epoch_20.pt \
bash scripts/serve_rollout.sh

INFERENCE_BACKEND=swift_rollout bash scripts/serve_web.sh
```

## tmux (recommended)
Run in a persistent tmux session:

```bash
tmux new -s ecg_r1_web
cd /data/jinjiarui/run/ecg-r1-web
source /home/jinjiarui/miniconda3/bin/activate swift2
bash scripts/serve_web.sh
```

## Configuration

Use environment variables for deployment-specific values: `INFERENCE_BACKEND`, `SWIFT_ROLLOUT_URL`, `MODEL_PATH`, `ECG_TOWER_PATH`, and `DATA_COLLECTION_DIR`.
