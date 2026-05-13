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
2.  For direct vLLM inference, install runtime extras in the serving environment if needed:
    ```bash
    pip install -e ".[runtime]"
    ```

## Running the Application

Edit `config.py` for project-local settings such as `INFERENCE_BACKEND`,
`MODEL_PATH`, `ECG_TOWER_PATH`, `WEB_PORT`, `VLLM_PORT`, and the
startup conda environment.

Start the FastAPI server with the mock provider by setting
`INFERENCE_BACKEND = "mock"` in `config.py`:

```bash
bash scripts/serve_web.sh
```

Access the web interface at `http://127.0.0.1:8000` locally or
`http://<server-ip>:8000` from another machine.

Run with a separate direct vLLM service by setting
`INFERENCE_BACKEND = "vllm_direct"` in `config.py`:

```bash
bash scripts/serve_vllm.sh

bash scripts/serve_web.sh
```

## tmux (recommended)
Run the vLLM and web services in persistent tmux sessions:

```bash
tmux new -d -s ecg-r1-rollout 'cd /data/jinjiarui/run/ecg-r1-web && bash scripts/serve_vllm.sh'
tmux new -d -s ecg-r1-web 'cd /data/jinjiarui/run/ecg-r1-web && bash scripts/serve_web.sh'
```

## Configuration

Use `config.py` for deployment-specific values. Startup scripts export a small
process-local environment only because vLLM runtime code reads those keys;
the values still come from this repository's `config.py`.
