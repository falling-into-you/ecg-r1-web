# ECG-R1 Web Frontend

This is a web frontend for the ECG-R1 multimodal model, providing an interface for diagnosing ECG images and signals.

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

## Configuration

Modify `config.py` to adjust model paths and environment variables.
