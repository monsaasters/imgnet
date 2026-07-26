"""
ImgNet Demo App — FastAPI + vanilla JS/HTML
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from imgnet.metrics import batch_compare, cosine_similarity, img_sign_score
from imgnet.visualizer import compute_embedding, model_status

app = FastAPI(title="ImgNet Demo")

static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return (static_dir / "index.html").read_text(encoding="utf-8")


@app.get("/api/health")
async def health() -> dict:
    return {"status": "ok", **model_status()}


@app.post("/api/compare")
async def compare(request: Request) -> dict:
    form = await request.form()
    file_a = form.get("image_a")
    file_b = form.get("image_b")

    if file_a is None or file_b is None:
        return {"error": "Both images are required."}

    try:
        img_a = Image.open(io.BytesIO(await file_a.read())).convert("RGB")
        img_b = Image.open(io.BytesIO(await file_b.read())).convert("RGB")
    except Exception as exc:
        return {"error": f"Invalid image: {exc}"}

    try:
        emb_a = compute_embedding(np.array(img_a))
        emb_b = compute_embedding(np.array(img_b))
    except Exception as exc:
        return {"error": f"Embedding failed: {exc}"}

    result = batch_compare(emb_a, emb_b)

    return {
        "img_sign": result["img_sign"],
        "amp_img": result["amp_img"],
        "chain_score": result["chain_score"],
        "cosine": result["cosine"],
        "vote": result["vote"],
        "chains": result["chains"],
        "avg_chain": result["avg_chain"],
        "model_status": model_status(),
    }
