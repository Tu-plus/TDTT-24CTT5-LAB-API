
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
import base64
import requests as req
from io import BytesIO
from PIL import Image

app = FastAPI(title="Image Classification API")

try:
    classifier = pipeline(
        "image-classification",
        model="google/vit-base-patch16-224"
    )
    model_loaded = True
except Exception as e:
    classifier = None
    model_loaded = False

class ImageURLInput(BaseModel):
    url: str
    top_k: int = 5

class ImageBase64Input(BaseModel):
    image_base64: str
    top_k: int = 5

def run_classifier(image, top_k):
    results = classifier(image, top_k=top_k)
    return [{"label": r["label"], "score": round(r["score"], 4)} for r in results]

@app.get("/")
def root():
    return {
        "name": "Image Classification API",
        "model": "google/vit-base-patch16-224",
    }

@app.get("/health")
def health():
    return {"status": "ok" if model_loaded else "error", "model_loaded": model_loaded}

@app.post("/predict/url")
def predict_from_url(body: ImageURLInput):
    if not body.url.strip():
        raise HTTPException(status_code=400, detail="Truong url khong duoc de trong.")
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model chua duoc tai.")
    if body.top_k < 1 or body.top_k > 10:
        raise HTTPException(status_code=400, detail="top_k phai tu 1 den 10.")
    try:
        # ✅ Thêm User-Agent để không bị chặn
        headers = {"User-Agent": "Mozilla/5.0"}
        response = req.get(body.url, timeout=10, headers=headers)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Khong the tai anh tu URL.")
    try:
        results = run_classifier(image, body.top_k)
        return {"input_url": body.url, "top_k": body.top_k, "predictions": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Loi khi phan loai: {str(e)}")

@app.post("/predict/base64")
def predict_from_base64(body: ImageBase64Input):
    if not body.image_base64.strip():
        raise HTTPException(status_code=400, detail="Truong image_base64 khong duoc de trong.")
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model chua duoc tai.")
    if body.top_k < 1 or body.top_k > 10:
        raise HTTPException(status_code=400, detail="top_k phai tu 1 den 10.")
    try:
        image_data = base64.b64decode(body.image_base64)
        image = Image.open(BytesIO(image_data)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Chuoi base64 khong hop le.")
    try:
        results = run_classifier(image, body.top_k)
        return {"top_k": body.top_k, "predictions": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Loi khi phan loai: {str(e)}")
