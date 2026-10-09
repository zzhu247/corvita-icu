"""FastAPI app.

Run locally:
    uvicorn api.main:create_app --factory --port 8000

Env vars:
    MODEL_DIR  model version folder (default models/v1)
    LOG_PATH   also append request logs to this JSONL file (default: stdout only)

Every request writes one JSON log line: request_id, path, status, latency_ms,
model_version, and for predictions the quality flag, missing vitals, vital counts
and the score. Never logged: record_id, age, sex, raw vital values.
"""
import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone

from fastapi import FastAPI, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from api.schemas import CFG, PredictRequest, PredictResponse
from api.service import Predictor
from src.config import p


class RequestLogger:
    def __init__(self, path: str | None):
        self.path = path
        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    def write(self, event: dict) -> None:
        line = json.dumps(event, separators=(",", ":"))
        print(line, file=sys.stdout, flush=True)
        if self.path:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")


def create_app(model_dir: str | None = None, log_path: str | None = None) -> FastAPI:
    model_dir = model_dir or os.environ.get("MODEL_DIR") or str(p("models/v1"))
    log_path = log_path or os.environ.get("LOG_PATH")
    predictor = Predictor(model_dir, CFG)          # fails fast if the model is missing or mismatched
    logger = RequestLogger(log_path)

    app = FastAPI(title="Corvita ICU 24h mortality risk",
                  version=predictor.meta["model_version"],
                  description="Research prototype. Not for clinical use.")

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex
        request.state.request_id = rid
        request.state.log = {}
        start = time.perf_counter()
        try:
            response = await call_next(request)
            status = response.status_code
        except Exception as exc:                    # unexpected bug: log type only, return 500
            request.state.log["error"] = type(exc).__name__
            response = JSONResponse({"detail": "internal error", "request_id": rid}, status_code=500)
            status = 500
        response.headers["x-request-id"] = rid
        logger.write({
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "request_id": rid, "method": request.method, "path": request.url.path,
            "status": status, "latency_ms": round((time.perf_counter() - start) * 1000, 2),
            "model_version": predictor.meta["model_version"], **request.state.log,
        })
        return response

    @app.exception_handler(RequestValidationError)
    async def on_validation_error(request: Request, exc: RequestValidationError):
        # log which fields failed, not what was sent
        request.state.log["error"] = "validation"
        request.state.log["fields"] = sorted({".".join(str(x) for x in e["loc"][1:]) for e in exc.errors()})
        return await request_validation_exception_handler(request, exc)

    @app.get("/health")
    def health():
        return {"status": "ok", "model_version": predictor.meta["model_version"],
                "data_version": predictor.meta["data_version"]}

    @app.get("/model-info")
    def model_info():
        keys = ["model_version", "model_family", "hyperparameters", "threshold", "threshold_rule",
                "data_version", "cutoff_minutes", "created_at", "git_commit", "feature_columns"]
        return {k: predictor.meta.get(k) for k in keys}

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest, request: Request):
        out = predictor.predict(req)
        log = out.pop("_log")
        request.state.log.update({
            "data_quality_flag": out["data_quality_flag"], "missing_vitals": out["missing_vitals"],
            "probability": out["probability"], "high_risk": out["high_risk"],
            "n_warnings": len(out["warnings"]), **log,
        })
        return {**out, "record_id": req.record_id, "request_id": request.state.request_id}

    return app
