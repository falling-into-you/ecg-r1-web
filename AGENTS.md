# Repository Guidelines

## Project Structure & Module Organization

- `main.py`: FastAPI app, request handling, provider calls, analytics, feedback, and data persistence.
- `providers/`: inference backends such as `mock`, `vllm_direct`, and legacy `swift_rollout`.
- `ecg_r1_runtime/`: project-local ECG-R1 vLLM runtime, plugin registration, and ECG loading code.
- `ecg_coca/`: minimal ECG-COCA runtime code; checkpoints are not stored in Git.
- `config.py`: provider selection, service URLs, conda envs, runtime paths, and `DATA_COLLECTION_DIR`.
- `templates/`: Jinja2 pages, including `index.html`, `loading.html`, and `analytics.html`.
- `static/`: browser JS, CSS, responsive styles, analytics assets, and icon fonts.
- `scripts/`: startup helpers and shell integration checks.
- `deploy/nginx/`: example Nginx reverse-proxy configs.
- `data_collection/`: runtime output, not source code.

Read `PROJECT.md` before feature work; it records behavior, deployment notes, backlog, and update flow.

Runtime startup must use conda environment `swift2`. Do not start Web or vLLM from base Python or a conda environment named `swift`.

## Build, Test, and Development Commands

- `pip install -r requirements.txt`: install dependencies.
- `bash scripts/serve_web.sh`: run the web app using `WEB_HOST`, `WEB_PORT`, and `INFERENCE_BACKEND` from `config.py`.
- `bash scripts/serve_vllm.sh`: start direct vLLM inference using `MODEL_PATH`, `ECG_TOWER_PATH`, and runtime settings from `config.py`.
- `bash archive/scripts/serve_rollout.sh`: compatibility wrapper that delegates to `scripts/serve_vllm.sh`.
- `bash archive/scripts/test_remote_api_44000.sh http://127.0.0.1:8000`: check `/status`, `/predict`, and `/predict_stream`.
- `bash archive/scripts/test_predict_stream_with_image.sh http://127.0.0.1:8000 /path/to/image.png`: test streaming image inference.
- `bash archive/scripts/test_forwarded_ip_capture.sh http://127.0.0.1:8000`: verify forwarded client IP capture.

Do not reintroduce `ECG_R1_ROOT` or imports from an external ECG-R1 checkout. Use `ecg_r1_runtime/` and project-local `config.py` instead. Runtime environment variables may be exported only inside startup scripts for vLLM compatibility.

## Coding Style & Naming Conventions

Use 4-space indentation for Python. Prefer `snake_case` for functions and variables, `UPPER_CASE` for constants, and descriptive endpoint names.

For frontend files, keep the existing plain JavaScript and CSS style. Use semantic class names such as `analytics-*`, `result-*`, or `feedback-*`.

No formatter or linter config is currently committed. Preserve nearby style when editing.

## Testing Guidelines

There is no formal unit-test framework. Current validation uses `scripts/test_*.sh` against a running server. Run the script that matches the changed behavior. For inference changes, include `/predict` or `/predict_stream` validation. For proxy/client metadata changes, run the forwarded-IP script.

If adding tests, place them under `tests/` or add `scripts/test_<feature>.sh` with clear defaults.

## Commit & Pull Request Guidelines

Recent history uses prefixes such as `feat:`, `fix:`, `style:`, `test:`, `docs:`, and `chore:`. Keep subjects short and behavior-focused.

Pull requests should include a concise description, affected routes or UI areas, validation commands, and screenshots for UI changes. Feature updates must also update `PROJECT.md`.

## Security & Configuration Tips

Do not commit private model checkpoints, generated patient/request data, or environment-specific secrets. Treat `config.py` absolute paths as deployment configuration.
