from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
import base64
import requests as req
from io import BytesIO
from PIL import Image

app = FastAPI(title="Image Classification API")

# Load model lúc khởi động server
try:
    classifier = pipeline(
        "image-classification",
        model="google/vit-base-patch16-224"
    )
    model_loaded = True
except Exception as e:
    classifier = None
    model_loaded = False
    print(f"Lỗi load model: {e}")


# ── Schemas ──────────────────────────────────────────────────────────

class ImageURLInput(BaseModel):
    url: str                  # Link ảnh từ internet
    top_k: int = 5            # Số lượng kết quả trả về (mặc định 5)

class ImageBase64Input(BaseModel):
    image_base64: str         # Ảnh mã hóa base64
    top_k: int = 5


# ── Helper ───────────────────────────────────────────────────────────

def run_classifier(image: Image.Image, top_k: int):
    results = classifier(image, top_k=top_k)
    return [
        {"label": r["label"], "score": round(r["score"], 4)}
        for r in results
    ]


# ── Endpoints ─────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "name": "Image Classification API",
        "description": "Phân loại hình ảnh sử dụng mô hình google/vit-base-patch16-224",
        "model": "google/vit-base-patch16-224",
        "endpoints": {
            "GET  /":             "Thông tin API",
            "GET  /health":       "Kiểm tra trạng thái",
            "POST /predict/url":  "Phân loại ảnh từ URL",
            "POST /predict/base64": "Phân loại ảnh từ base64"
        }
    }


@app.get("/health")
def health():
    return {
        "status": "ok" if model_loaded else "error",
        "model_loaded": model_loaded,
        "model": "google/vit-base-patch16-224"
    }


@app.post("/predict/url")
def predict_from_url(body: ImageURLInput):
    """Phân loại ảnh từ đường dẫn URL"""
    if not body.url.strip():
        raise HTTPException(status_code=400, detail="Trường 'url' không được để trống.")
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model chưa được tải.")
    if body.top_k < 1 or body.top_k > 10:
        raise HTTPException(status_code=400, detail="top_k phải từ 1 đến 10.")

    try:
        response = req.get(body.url, timeout=10)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Không thể tải ảnh từ URL. Kiểm tra lại đường dẫn.")

    try:
        results = run_classifier(image, body.top_k)
        return {
            "input_url": body.url,
            "top_k": body.top_k,
            "predictions": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi phân loại: {str(e)}")


@app.post("/predict/base64")
def predict_from_base64(body: ImageBase64Input):
    """Phân loại ảnh từ chuỗi base64"""
    if not body.image_base64.strip():
        raise HTTPException(status_code=400, detail="Trường 'image_base64' không được để trống.")
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model chưa được tải.")
    if body.top_k < 1 or body.top_k > 10:
        raise HTTPException(status_code=400, detail="top_k phải từ 1 đến 10.")

    try:
        image_data = base64.b64decode(body.image_base64)
        image = Image.open(BytesIO(image_data)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Chuỗi base64 không hợp lệ hoặc không phải ảnh.")

    try:
        results = run_classifier(image, body.top_k)
        return {
            "top_k": body.top_k,
            "predictions": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi phân loại: {str(e)}")
