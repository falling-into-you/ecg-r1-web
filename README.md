# ECG-R1 Web Frontend

This is a web frontend for the ECG-R1 multimodal model, providing an interface for diagnosing ECG images and signals.

## 必读文档（开发流程要求）
- 开始任何开发/修复前：先阅读 [PROJECT.md](file:///data/jinjiarui/run/ecg-r1-web/PROJECT.md)
- 每次功能更新后必须：本地验证 → git commit → git push → 更新 PROJECT.md（已实现/待实现/版本记录）

## Setup

1.  Ensure you have access to the `ecg-r1` model and code in `/data/jinjiarui/run/ecg-r1`.
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Running the Application

Start the FastAPI server:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Access the web interface at `http://localhost:8000`.

## tmux (recommended)
Run in a persistent tmux session:

```bash
tmux new -s ecg_r1_web
cd /data/jinjiarui/run/ecg-r1-web
source /home/jinjiarui/miniconda3/bin/activate swift2
uvicorn main:app --host 0.0.0.0 --port 8000 --log-level info
```

## Configuration

Modify `config.py` to adjust model paths and environment variables.
