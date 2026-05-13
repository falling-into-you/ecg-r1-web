# Repository Guidelines

## Project Structure & Module Organization

- `main.py`: FastAPI app, request handling, provider calls, analytics, feedback, and data persistence.
- `providers/`: inference backends such as `mock` and `swift_rollout`.
- `ecg_r1_runtime/`: local Swift/vLLM runtime registration code.
- `ecg_coca/`: minimal ECG-COCA runtime code; checkpoints are not stored in Git.
- `config.py`: provider selection, service URLs, runtime paths, and `DATA_COLLECTION_DIR`.
- `templates/`: Jinja2 pages, including `index.html`, `loading.html`, and `analytics.html`.
- `static/`: browser JS, CSS, responsive styles, analytics assets, and icon fonts.
- `scripts/`: shell integration checks for local or remote HTTP endpoints.
- `deploy/nginx/`: example Nginx reverse-proxy configs.
- `data_collection/`: runtime output, not source code.

Read `PROJECT.md` before feature work; it records behavior, deployment notes, backlog, and update flow.

## Build, Test, and Development Commands

- `pip install -r requirements.txt`: install dependencies.
- `INFERENCE_BACKEND=mock uvicorn main:app --host 0.0.0.0 --port 8000 --reload`: run the web app without a model service.
- `MODEL_PATH=/path/to/model ECG_TOWER_PATH=/path/to/cpt_wfep_epoch_20.pt bash scripts/serve_rollout.sh`: start Swift rollout.
- `INFERENCE_BACKEND=swift_rollout bash scripts/serve_web.sh`: run the web app against rollout.
- `bash scripts/test_remote_api_44000.sh http://127.0.0.1:8000`: check `/status`, `/predict`, and `/predict_stream`.
- `bash scripts/test_predict_stream_with_image.sh http://127.0.0.1:8000 /path/to/image.png`: test streaming image inference.
- `bash scripts/test_forwarded_ip_capture.sh http://127.0.0.1:8000`: verify forwarded client IP capture.

Do not reintroduce `ECG_R1_ROOT` or imports from an external ECG-R1 checkout. Use `ecg_r1_runtime/` and environment variables instead.

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
