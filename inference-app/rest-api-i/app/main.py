from fastapi import (APIRouter, Body, Depends, FastAPI, File, HTTPException,
                     Query, UploadFile)
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from .config import settings
from .db import get_session
from .middleware import MetricsMiddleware, metrics_endpoint
from .schemas import (AnalyzeRequest, AnalyzeResponse, OrdersResponse,
                      PredictResponse)
from .workloads import cpu as cpu_work
from .workloads import io as io_work
from .workloads import memory as mem_work

app = FastAPI(title=settings.APP_NAME)
app.add_middleware(MetricsMiddleware)
app.add_route("/metrics", metrics_endpoint, methods=["GET"])

@app.get("/healthz")
async def health():
    return {"status":"ok"}

@app.post("/predict", response_model=PredictResponse)
async def predict(
    file: UploadFile = File(..., description="Upload product image"),
    topk: int = Query(..., description="Number of top predictions to return")
):
    try:
        content = await file.read()
        result = cpu_work.classify_image(content, topk=topk)
        return PredictResponse(
            result=result["predictions"],
            elapsed_ms=result["elapsed_ms"]
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    payload: AnalyzeRequest = Body(..., description="Large JSON transactions"),
):
    res = mem_work.analyze(
        list(payload.transactions),
        expand_factor=payload.expand_factor,
        heavy_agg=payload.heavy_agg,
    )
    return AnalyzeResponse(**res)

@app.get("/orders", response_model=OrdersResponse)
async def orders(
    limit: int = Query(...),
    offset: int = Query(...),
    session: AsyncSession = Depends(get_session),
):
    res = await io_work.get_orders(session, limit, offset)
    return OrdersResponse(**res)
