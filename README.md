# 🖼️ Image Classification API

> Hệ thống Web API phân loại hình ảnh sử dụng mô hình **Vision Transformer** từ Hugging Face, xây dựng bằng **FastAPI**.

---

## 👤 Thông tin sinh viên

| Thông tin | Chi tiết |
|-----------|----------|
| **Họ và tên** | Trần Ngọc Thanh Tú |
| **MSSV** | 24120152 |
| **Lớp** | 24CTT5  |
| **Môn học** | Tư Duy Tính Toán |
| **Giảng viên** | Lê Đức Khoan |

---

## 🤖 Mô hình sử dụng

| Thông tin | Chi tiết |
|-----------|----------|
| **Tên mô hình** | `google/vit-base-patch16-224` |
| **Link Hugging Face** | https://huggingface.co/google/vit-base-patch16-224 |
| **Loại mô hình** | Image Classification (Vision Transformer) |
| **Số lớp phân loại** | 1000 loại vật thể (ImageNet) |

**Mô tả:** ViT (Vision Transformer) là mô hình phân loại hình ảnh của Google, được huấn luyện trên tập dữ liệu ImageNet gồm 1000 loại vật thể như chó, mèo, xe hơi, máy bay, v.v. Mô hình nhận ảnh đầu vào và trả về danh sách các nhãn có xác suất cao nhất.

---

## 📁 Cấu trúc dự án

```
image-classification-api/
├── main.py              # FastAPI application chính
├── test_api.py          # File kiểm thử API bằng requests
├── requirements.txt     # Danh sách thư viện cần thiết
├── README.md            # Tài liệu hướng dẫn (file này)
└── notebook.ipynb       # Notebook Colab 
```

---

## ⚙️ Yêu cầu hệ thống

- Python **3.9** trở lên
- pip
- Kết nối Internet (để tải model lần đầu)

---

## 🚀 Hướng dẫn cài đặt và chạy chương trình

### Bước 1 — Clone repository

```bash
git clone https://github.com/[username]/image-classification-api.git
cd image-classification-api
```

### Bước 2 — Tạo môi trường ảo (Virtual Environment)

Tạo môi trường ảo giúp tách biệt các thư viện của dự án, tránh xung đột với các dự án khác.

**Trên Windows:**
```bash
# Tạo môi trường ảo tên là "venv"
python -m venv venv

# Kích hoạt môi trường ảo
venv\Scripts\activate
```

**Trên macOS / Linux:**
```bash
# Tạo môi trường ảo tên là "venv"
python3 -m venv venv

# Kích hoạt môi trường ảo
source venv/bin/activate
```

> ✅ Khi kích hoạt thành công, terminal sẽ hiển thị `(venv)` ở đầu dòng.

**Tắt môi trường ảo (khi xong):**
```bash
deactivate
```

### Bước 3 — Cài đặt thư viện

```bash
pip install -r requirements.txt
```

> ⏳ Lần đầu chạy sẽ mất vài phút để tải model (~350MB).

### Bước 4 — Chạy server

```bash
uvicorn main:app --reload
```

Nếu thành công, terminal sẽ hiển thị:
```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started reloader process
```

### Bước 5 — Kiểm tra API

Mở trình duyệt và truy cập:

| URL | Mô tả |
|-----|-------|
| http://localhost:8000 | Thông tin API |
| http://localhost:8000/health | Kiểm tra trạng thái |
| http://localhost:8000/docs | **Giao diện Swagger UI** (test trực tiếp) |

---

## 📡 Hướng dẫn gọi API

### `GET /` — Thông tin API

**Request:**
```bash
curl http://localhost:8000/
```

**Response:**
```json
{
  "name": "Image Classification API",
  "description": "Phân loại hình ảnh sử dụng mô hình google/vit-base-patch16-224",
  "model": "google/vit-base-patch16-224",
  "endpoints": {
    "GET  /": "Thông tin API",
    "GET  /health": "Kiểm tra trạng thái",
    "POST /predict/url": "Phân loại ảnh từ URL",
    "POST /predict/base64": "Phân loại ảnh từ base64"
  }
}
```

---

### `GET /health` — Kiểm tra trạng thái

**Request:**
```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "ok",
  "model_loaded": true,
  "model": "google/vit-base-patch16-224"
}
```

---

### `POST /predict/url` — Phân loại ảnh từ URL

**Request body:**

| Trường | Kiểu | Bắt buộc | Mô tả |
|--------|------|----------|-------|
| `url` | string | ✅ | Đường dẫn URL đến ảnh |
| `top_k` | integer | ❌ | Số lượng kết quả (1–10, mặc định: 5) |

**Ví dụ — Ảnh con chó:**
```bash
curl -X POST http://localhost:8000/predict/url \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/2/26/YellowLabradorLooking_new.jpg/1200px-YellowLabradorLooking_new.jpg",
    "top_k": 3
  }'
```

**Response:**
```json
{
  "input_url": "https://upload.wikimedia.org/.../YellowLabradorLooking_new.jpg",
  "top_k": 3,
  "predictions": [
    { "label": "Labrador retriever", "score": 0.9231 },
    { "label": "golden retriever",   "score": 0.0412 },
    { "label": "kuvasz",             "score": 0.0087 }
  ]
}
```

**Ví dụ — Ảnh con mèo:**
```bash
curl -X POST http://localhost:8000/predict/url \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Cat_November_2010-1a.jpg/1200px-Cat_November_2010-1a.jpg",
    "top_k": 3
  }'
```

**Response:**
```json
{
  "input_url": "https://upload.wikimedia.org/.../Cat_November_2010-1a.jpg",
  "top_k": 3,
  "predictions": [
    { "label": "tabby cat",   "score": 0.6821 },
    { "label": "tiger cat",   "score": 0.2134 },
    { "label": "Egyptian cat","score": 0.0512 }
  ]
}
```

---

### `POST /predict/base64` — Phân loại ảnh từ Base64

**Request body:**

| Trường | Kiểu | Bắt buộc | Mô tả |
|--------|------|----------|-------|
| `image_base64` | string | ✅ | Ảnh được mã hóa dạng base64 |
| `top_k` | integer | ❌ | Số lượng kết quả (1–10, mặc định: 5) |

**Ví dụ (Python):**
```python
import base64, requests

with open("my_image.jpg", "rb") as f:
    encoded = base64.b64encode(f.read()).decode("utf-8")

response = requests.post("http://localhost:8000/predict/base64", json={
    "image_base64": encoded,
    "top_k": 3
})
print(response.json())
```

**Response:**
```json
{
  "top_k": 3,
  "predictions": [
    { "label": "tabby cat",  "score": 0.7102 },
    { "label": "tiger cat",  "score": 0.1893 },
    { "label": "Persian cat","score": 0.0421 }
  ]
}
```

---

### ⚠️ Các trường hợp lỗi

| HTTP Status | Nguyên nhân | Ví dụ response |
|-------------|-------------|----------------|
| `400` | URL rỗng hoặc không hợp lệ | `{"detail": "Trường 'url' không được để trống."}` |
| `400` | top_k ngoài khoảng 1–10 | `{"detail": "top_k phải từ 1 đến 10."}` |
| `400` | Base64 không hợp lệ | `{"detail": "Chuỗi base64 không hợp lệ hoặc không phải ảnh."}` |
| `503` | Model chưa được tải | `{"detail": "Model chưa được tải."}` |
| `500` | Lỗi trong quá trình suy luận | `{"detail": "Lỗi khi phân loại: ..."}` |

---

## 🧪 Chạy file kiểm thử

Đảm bảo server đang chạy, sau đó mở terminal mới:

```bash
# Kích hoạt môi trường ảo trước
source venv/bin/activate       # macOS/Linux
# hoặc
venv\Scripts\activate          # Windows

# Chạy file test
python test_api.py
```

---

## 🎬 Video Demo

[![Video Demo](https://img.shields.io/badge/▶️_Xem_Video_Demo-YouTube-red?style=for-the-badge)](https://[link-video-cua-ban])

> 📌 Link video: **[Dán link video vào đây]**

---

## 📚 Thư viện sử dụng

| Thư viện | Mục đích |
|----------|----------|
| `fastapi` | Framework xây dựng Web API |
| `uvicorn` | ASGI server để chạy FastAPI |
| `transformers` | Tải và sử dụng mô hình Hugging Face |
| `torch` | Backend tính toán cho mô hình |
| `Pillow` | Xử lý ảnh (mở, convert) |
| `requests` | Gọi HTTP request (tải ảnh từ URL, kiểm thử) |
| `pydantic` | Validate dữ liệu đầu vào |