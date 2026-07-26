"""
FastAPI app exposing IMGNet metrics over HTTP.

Run:
    uvicorn imgnet.api:app --reload

Docs:
    http://127.0.0.1:8000/docs
"""

from __future__ import annotations

from typing import List, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from imgnet.metrics import (
    amp_img_score,
    batch_compare,
    chain_score,
    cosine_similarity,
    img_sign_score,
)

app = FastAPI(
    title="IMGNet API",
    description="Universal similarity metrics API.",
    version="0.1.0",
)


class CompareRequest(BaseModel):
    a: List[float] = Field(..., min_length=2)
    b: List[float] = Field(..., min_length=2)
    threshold: float = Field(default=0.79, ge=0.0, le=1.0)
    window_size: int = Field(default=11, ge=2)
    threshold_window: int = Field(default=8, ge=1)


class MetricRequest(BaseModel):
    a: List[float] = Field(..., min_length=2)
    b: List[float] = Field(..., min_length=2)
    window_size: int = Field(default=11, ge=2)
    threshold: int = Field(default=8, ge=1)


class BatchRequest(BaseModel):
    pairs: List[CompareRequest]


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/v1/metrics/img-sign")
def metric_img_sign(body: MetricRequest) -> dict:
    score = img_sign_score(body.a, body.b, window_size=body.window_size, threshold=body.threshold)
    return {"metric": "img_sign", "score": score}


@app.post("/v1/metrics/amp-img")
def metric_amp_img(body: MetricRequest) -> dict:
    score = amp_img_score(body.a, body.b, window_size=body.window_size, threshold=body.threshold)
    return {"metric": "amp_img", "score": score}


@app.post("/v1/metrics/chain")
def metric_chain(body: MetricRequest) -> dict:
    score, chains, avg_chain = chain_score(body.a, body.b, window_size=body.window_size, threshold=body.threshold)
    return {"metric": "chain_score", "score": score, "chains": chains, "avg_chain": avg_chain}


@app.post("/v1/metrics/cosine")
def metric_cosine(body: MetricRequest) -> dict:
    score = cosine_similarity(body.a, body.b)
    return {"metric": "cosine", "score": score}


@app.post("/v1/compare")
def compare(body: CompareRequest) -> dict:
    result = batch_compare(
        body.a,
        body.b,
        threshold=body.threshold,
        window_size=body.window_size,
        threshold_window=body.threshold_window,
    )
    return result


@app.post("/v1/batch")
def batch(body: BatchRequest) -> List[dict]:
    return [batch_compare(pair.a, pair.b) for pair in body.pairs]
