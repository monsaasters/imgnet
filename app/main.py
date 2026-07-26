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
from imgnet.liveness import check_liveness

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


@app.post("/api/liveness")
async def liveness(request: Request) -> dict:
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
        result = check_liveness(np.array(img_a), np.array(img_b))
    except Exception as exc:
        return {"error": f"Liveness check failed: {exc}"}

    return {
        "live": result.live,
        "confidence": result.confidence,
        "vote": result.vote,
        "img_sign": result.img_sign,
        "cosine": result.cosine,
        "chain_score": result.chain_score,
        "chains": result.chains,
        "img_sign_delta": result.img_sign_delta,
        "reason": result.reason,
    }


@app.post("/api/report/csv")
async def report_csv(request: Request) -> HTMLResponse:
    form = await request.form()
    # Accept either direct fields or a JSON blob named `payload`
    payload = form.get("payload")
    if payload is None:
        data = {
            "img_sign": float(form.get("img_sign", 0)),
            "amp_img": float(form.get("amp_img", 0)),
            "chain_score": float(form.get("chain_score", 0)),
            "cosine": float(form.get("cosine", 0)),
            "vote": str(form.get("vote", "")),
            "chains": int(form.get("chains", 0)),
            "avg_chain": float(form.get("avg_chain", 0)),
        }
    else:
        import json as _json
        data = _json.loads(payload)

    lines = ["metric,value", f"img_sign,{data.get('img_sign', '')}", f"amp_img,{data.get('amp_img', '')}",
             f"chain_score,{data.get('chain_score', '')}", f"cosine,{data.get('cosine', '')}",
             f"vote,{data.get('vote', '')}", f"chains,{data.get('chains', '')}",
             f"avg_chain,{data.get('avg_chain', '')}"]
    csv_content = "\n".join(lines) + "\n"
    return HTMLResponse(content=csv_content, media_type="text/csv", headers={"Content-Disposition": "attachment; filename=imgnet_report.csv"})


@app.post("/api/report/json")
async def report_json(request: Request) -> dict:
    form = await request.form()
    payload = form.get("payload")
    if payload is None:
        return {"error": "payload is required"}
    import json as _json
    try:
        data = _json.loads(payload)
    except Exception as exc:
        return {"error": f"Invalid JSON payload: {exc}"}
    return {"report": data, "generated_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()}
