"""Run with: python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000."""
import logging
import time
from contextlib import asynccontextmanager
from typing import Literal

import pandas as pd
from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from starlette.concurrency import run_in_threadpool

from backend.config import Settings
from backend.csv_utils import parse_upload, results_csv
from backend.schemas import PredictionRequest, PredictionResponse
from backend.services.history_service import HistoryService
from backend.services.model_service import ModelService
from backend.services.prediction_service import PredictionService

logger = logging.getLogger("sentinel.web")


class BodyLimitMiddleware:
    def __init__(self, app, limit):
        self.app, self.limit = app, limit

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] not in {"POST", "PUT", "PATCH"}:
            return await self.app(scope, receive, send)
        limit = self.limit if scope["path"].startswith("/predict/batch") else 65536
        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            body.extend(message.get("body", b""))
            if len(body) > limit:
                return await JSONResponse({"detail": f"Request exceeds {limit:,} bytes."}, status_code=413)(scope, receive, send)
            if not message.get("more_body", False):
                break
        delivered = False

        async def replay():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            return await receive()
        await self.app(scope, replay, send)


def create_app(settings=None, model=None):
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app):
        app.state.service = None
        app.state.started = time.monotonic()
        try:
            loaded = model or ModelService(settings.model_dir)
            app.state.service = PredictionService(loaded, settings, HistoryService(settings.history_db))
            logger.info("Fraud model loaded: %s", loaded.metadata["model_version"])
        except Exception as exc:
            # No transaction payloads or model contents in logs.
            logger.error("Model initialization failed (%s). Verify local artifacts and package versions.", type(exc).__name__)
        yield

    app = FastAPI(title="Sentinel · Fraud Intelligence API", version="1.0.0", lifespan=lifespan,
                  description="Credit-card transaction benchmark. Predictions are model flags, not confirmed fraud. Use /model-info for the active dataset and feature schema.")
    app.add_middleware(BodyLimitMiddleware, limit=settings.max_upload_bytes + 65536)
    app.add_middleware(CORSMiddleware, allow_origins=[origin.strip() for origin in settings.cors_origins.split(",")],
                       allow_methods=["GET", "POST"], allow_headers=["Content-Type"], allow_credentials=False,
                       expose_headers=["Content-Disposition"])

    def service():
        if app.state.service is None:
            raise HTTPException(503, "No verified model is ready. Train with python -m src.train and restart the API.")
        return app.state.service

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request, exc):
        return JSONResponse(status_code=422, content={"detail": [{"location": list(error["loc"]), "message": error["msg"]} for error in exc.errors()]})

    @app.exception_handler(ValueError)
    async def invalid_value(request, exc):
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.get("/health")
    def health():
        ready = app.state.service is not None
        if ready:
            try:
                with app.state.service.history.connect() as connection:
                    connection.execute("SELECT 1")
            except Exception:
                ready = False
        return JSONResponse(status_code=200 if ready else 503, content={"status": "healthy" if ready else "not_ready",
                            "model_loaded": app.state.service is not None,
                            "uptime_seconds": round(time.monotonic() - app.state.started, 1)})

    @app.get("/model-info")
    def model_info(svc=Depends(service)):
        return {**svc.model.metadata, "active_threshold": svc.default_threshold,
                "risk_bands": {"medium": settings.risk_medium, "high": settings.risk_high},
                "limits": {"max_upload_bytes": settings.max_upload_bytes, "max_batch_rows": settings.max_batch_rows}}

    @app.get("/statistics")
    def statistics(scope: Literal["evaluation", "live"] = "evaluation", svc=Depends(service)):
        snapshot = svc.model.snapshot if scope == "evaluation" else svc.history.snapshot(svc.model.metadata["model_version"])
        return {**snapshot, "dataset": svc.model.metadata["dataset"], "scope_key": scope,
                "risk_bands": svc.model.metadata.get("risk_bands", {"medium": .3, "high": .7}) if scope == "evaluation" else {"medium": settings.risk_medium, "high": settings.risk_high},
                "classification_threshold": svc.model.metadata["selected_threshold"] if scope == "evaluation" else None}

    @app.get("/examples")
    def examples(svc=Depends(service)):
        return {"source": f"{svc.model.metadata['dataset'].get('name', 'Benchmark')} training-partition examples; intended for a demonstration", "transactions": svc.model.examples}

    @app.get("/sample-csv")
    def sample_csv(svc=Depends(service)):
        return Response((svc.model.directory / "sample_transactions.csv").read_bytes(), media_type="text/csv",
                        headers={"Content-Disposition": 'attachment; filename="sample_transactions.csv"'})

    @app.post("/predict", response_model=PredictionResponse)
    def predict(request: PredictionRequest, svc=Depends(service)):
        return svc.score(pd.DataFrame([request.features]), request.threshold, explain=request.explain)[0]

    async def upload_frame(file, svc):
        content = await file.read(settings.max_upload_bytes + 1)
        await file.close()
        if len(content) > settings.max_upload_bytes:
            raise HTTPException(413, f"CSV exceeds {settings.max_upload_bytes:,} bytes.")
        return await run_in_threadpool(parse_upload, file.filename, content, svc.model.schema, settings.max_batch_rows)

    @app.post("/predict/batch/validate")
    async def validate_batch(file: UploadFile = File(...), svc=Depends(service)):
        frame = await upload_frame(file, svc)
        return {"valid": True, "rows": len(frame), "columns": frame.columns.tolist(),
                "imputed_cells": int(frame.isna().sum().sum()), "preview": frame.head(8).astype(object).where(frame.head(8).notna(), None).to_dict(orient="records")}

    @app.post("/predict/batch")
    async def predict_batch(file: UploadFile = File(...), threshold: float | None = Query(None, ge=0, le=1),
                            format: Literal["json", "csv"] = "json", svc=Depends(service)):
        frame = await upload_frame(file, svc)
        results = await run_in_threadpool(svc.score, frame, threshold, False, True, "batch")
        csv = await run_in_threadpool(results_csv, frame, results)
        if format == "csv":
            return Response(csv, media_type="text/csv", headers={"Content-Disposition": 'attachment; filename="fraud_predictions.csv"'})
        fraud = sum(result["prediction"] == "Fraud" for result in results)
        return {"total": len(results), "fraud_count": fraud, "legitimate_count": len(results) - fraud,
                "fraud_percentage": fraud / len(results) * 100, "threshold": svc.default_threshold if threshold is None else threshold,
                "results": results, "csv": csv}

    return app


app = create_app()
