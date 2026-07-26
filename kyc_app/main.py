"""
Offline KYC Verification App
=============================

Flow:
1. Upload selfie + KTP photo
2. Detect/crop faces using MTCNN
3. Generate embeddings using IMGNet model
4. Compare using IMG metrics + voting
5. Return result + confidence
6. Export CSV/JSON report

Run:
    PYTHONPATH=src uvicorn kyc_app.main:app --host 0.0.0.0 --port 8012
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from imgnet.metrics import batch_compare
from imgnet.visualizer import compute_embedding, model_status

app = FastAPI(title="IMGNet Offline KYC")

kyc_dir = Path(__file__).parent
templates_dir = kyc_dir / "templates"
static_dir = kyc_dir / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="kyc-static")


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return (templates_dir / "index.html").read_text(encoding="utf-8")


@app.get("/api/health")
async def health() -> dict:
    return {"status": "ok", **model_status()}


@app.post("/api/kyc/verify")
async def kyc_verify(request: Request) -> dict:
    form = await request.form()
    selfie = form.get("selfie")
    ktp = form.get("ktp")

    if selfie is None or ktp is None:
        return {"error": "Both selfie and KTP images are required."}

    try:
        selfie_img = Image.open(io.BytesIO(await selfie.read())).convert("RGB")
        ktp_img = Image.open(io.BytesIO(await ktp.read())).convert("RGB")
    except Exception as exc:
        return {"error": f"Invalid image: {exc}"}

    try:
        selfie_emb = compute_embedding(np.array(selfie_img))
        ktp_emb = compute_embedding(np.array(ktp_img))
    except Exception as exc:
        return {"error": f"Embedding failed: {exc}"}

    result = batch_compare(selfie_emb, ktp_emb)
    confidence = float(np.mean([
        result["img_sign"],
        result["amp_img"],
        result["chain_score"],
        max(0.0, (result["cosine"] + 1.0) / 2.0),
    ]))

    return {
        "vote": result["vote"],
        "confidence": round(confidence, 4),
        "img_sign": result["img_sign"],
        "amp_img": result["amp_img"],
        "chain_score": result["chain_score"],
        "cosine": result["cosine"],
        "chains": result["chains"],
        "avg_chain": result["avg_chain"],
        "model_status": model_status(),
    }


@app.post("/api/kyc/report/csv")
async def kyc_report_csv(request: Request) -> HTMLResponse:
    form = await request.form()
    payload = form.get("payload")
    if payload is None:
        data = {k: form.get(k, "") for k in ["vote", "confidence", "img_sign", "amp_img", "chain_score", "cosine", "chains", "avg_chain"]}
    else:
        import json as _json
        data = _json.loads(payload)

    lines = [
        "field,value",
        f"vote,{data.get('vote', '')}",
        f"confidence,{data.get('confidence', '')}",
        f"img_sign,{data.get('img_sign', '')}",
        f"amp_img,{data.get('amp_img', '')}",
        f"chain_score,{data.get('chain_score', '')}",
        f"cosine,{data.get('cosine', '')}",
        f"chains,{data.get('chains', '')}",
        f"avg_chain,{data.get('avg_chain', '')}",
    ]
    csv_content = "\n".join(lines) + "\n"
    return HTMLResponse(content=csv_content, media_type="text/csv", headers={"Content-Disposition": "attachment; filename=kyc_report.csv"})


@app.post("/api/kyc/report/json")
async def kyc_report_json(request: Request) -> dict:
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
