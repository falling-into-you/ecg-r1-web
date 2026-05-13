# Repository Guidelines

## Project Structure & Module Organization

This is a FastAPI web frontend for ECG-R1.

- `main.py`: FastAPI app, request handling, streaming inference, analytics, feedback, and data persistence.
- `config.py`: local model paths, CUDA settings, environment variables, and `DATA_COLLECTION_DIR`.
- `templates/`: Jinja2 pages, including `index.html`, `loading.html`, and `analytics.html`.
- `static/`: browser assets, including JS, CSS, responsive styles, analytics assets, and bundled icon fonts.
- `scripts/`: shell integration checks for local or remote HTTP endpoints.
- `deploy/nginx/`: example Nginx reverse-proxy configs.
- `data_collection/`: runtime output. Do not treat generated request data as source code.

Read `PROJECT.md` before feature work; it records current behavior, deployment notes, backlog, and update flow.

## Build, Test, and Development Commands

- `pip install -r requirements.txt`: install dependencies.
- `uvicorn main:app --host 0.0.0.0 --port 8000 --reload`: run locally with reload.
- `uvicorn main:app --host 0.0.0.0 --port 8000 --log-level info`: run a production-like local server.
- `bash scripts/test_remote_api_44000.sh http://127.0.0.1:8000`: check `/status`, `/predict`, and `/predict_stream`.
- `bash scripts/test_predict_stream_with_image.sh http://127.0.0.1:8000 /path/to/image.png`: test streaming image inference.
- `bash scripts/test_forwarded_ip_capture.sh http://127.0.0.1:8000`: verify forwarded client IP capture.

The model code is expected at `/data/jinjiarui/run/ecg-r1`. Change `config.py` only for local paths or runtime settings.

## Coding Style & Naming Conventions

Use 4-space indentation for Python. Prefer `snake_case` for functions and variables, `UPPER_CASE` for constants, and descriptive endpoint/helper names.

For frontend files, keep the existing plain JavaScript and CSS style. Use semantic class names tied to page sections or interactions, for example `analytics-*`, `result-*`, or `feedback-*`.

No formatter or linter config is currently committed. Preserve nearby style when editing.

## Testing Guidelines

There is no formal unit-test framework. Current validation uses `scripts/test_*.sh` against a running server. Run the script that matches the changed behavior before committing. For inference changes, include `/predict` or `/predict_stream` validation. For proxy/client metadata changes, run the forwarded-IP script.

If adding tests, place them under `tests/` or add `scripts/test_<feature>.sh` with clear defaults.

## Commit & Pull Request Guidelines

Recent history uses Conventional Commit prefixes such as `feat:`, `fix:`, `style:`, `test:`, `docs:`, and `chore:`. Keep subjects short and behavior-focused, for example `fix: preserve streamed answer sections`.

Pull requests should include a concise description, affected routes or UI areas, validation commands, and screenshots for visible UI changes. Link related issues when available. Feature updates must also update `PROJECT.md`.

## Security & Configuration Tips

Do not commit private model checkpoints, generated patient/request data, or environment-specific secrets. Treat `config.py` absolute paths as deployment configuration.
