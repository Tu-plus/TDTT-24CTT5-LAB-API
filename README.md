# Image Classification API

Hệ thống Web API phân loại hình ảnh sử dụng mô hình **Vision Transformer** từ Hugging Face, xây dựng bằng **FastAPI**.

---

## Thông tin sinh viên

| Thông tin | Chi tiết |
|-----------|----------|
| **Họ và tên** | Trần Ngọc Thanh Tú |
| **MSSV** | 24120152 |
| **Lớp** | 24CTT5 |
| **Môn học** | Tư Duy Tính Toán |
| **Giảng viên** | ThS. Lê Đức Khoan |

---

## Mô hình sử dụng

| Thông tin | Chi tiết |
|-----------|----------|
| **Tên mô hình** | `google/vit-base-patch16-224` |
| **Link Hugging Face** | https://huggingface.co/google/vit-base-patch16-224 |
| **Loại mô hình** | Image Classification (Vision Transformer) |
| **Số lớp phân loại** | 1000 loại vật thể (ImageNet) |

**Mô tả:** ViT (Vision Transformer) là mô hình phân loại hình ảnh của Google, được huấn luyện trên tập dữ liệu ImageNet gồm 1000 loại vật thể như chó, mèo, xe hơi, máy bay, v.v. Mô hình nhận ảnh đầu vào và trả về danh sách các nhãn có xác suất cao nhất.

Điểm đặc biệt là thay vì dùng CNN như truyền thống, mô hình này sẽ chia ảnh thành các phần nhỏ để xử lý.

Cụ thể, mỗi ảnh đầu vào sẽ được resize về kích thước 224x224. Sau đó, ảnh được chia thành các patch kích thước 16x16. Như vậy, một ảnh sẽ được tách thành 196 patch, và mỗi patch được xem như một “token” tương tự như một từ trong câu.

Tiếp theo, các patch này sẽ được biến đổi thành vector số, gọi là embedding. Mô hình cũng thêm một token đặc biệt gọi là [CLS] ở đầu để đại diện cho toàn bộ bức ảnh, cùng với thông tin vị trí của từng patch.

Toàn bộ chuỗi này sẽ được đưa vào Transformer Encoder. Tại đây, cơ chế self-attention sẽ giúp mô hình học được mối quan hệ giữa các vùng khác nhau trong ảnh, từ đó hiểu được nội dung tổng thể.

Sau khi đi qua các lớp Transformer, vector của token [CLS] sẽ được đưa qua một lớp fully connected để dự đoán nhãn của ảnh, ví dụ như chó, mèo hoặc các đối tượng khác.

Về quá trình huấn luyện, mô hình này đã được pretrain trên tập dữ liệu rất lớn là ImageNet-21k với hơn 14 triệu ảnh, sau đó được fine-tune lại trên ImageNet với 1000 lớp. Nhờ vậy, mô hình có khả năng nhận diện ảnh khá tốt ngay cả khi áp dụng vào các bài toán thực tế.

Tóm lại, Vision Transformer hoạt động bằng cách biến ảnh thành chuỗi patch và sử dụng self-attention để học mối quan hệ giữa chúng, thay vì dùng các phép tích chập như CNN truyền thống.


---

## Cấu trúc dự án

```
image-classification-api/
├── main.py              # FastAPI application chính
├── test_api.py          # File kiểm thử API bằng requests
├── requirements.txt     # Danh sách thư viện cần thiết
├── README.md            # Tài liệu hướng dẫn (file này)
└── notebook.ipynb       # Notebook Colab
```

---

## Yêu cầu hệ thống

- Python **3.9** trở lên
- pip
- Kết nối Internet (để tải model lần đầu)

---

## Hướng dẫn cài đặt và chạy chương trình

### Bước 1 — Clone repository

```bash
git clone https://github.com/Tu-plus/TDTT-24CTT5-LAB-API.git
cd TDTT-24CTT5-LAB-API
```

### Bước 2 — Tạo môi trường ảo (Virtual Environment)

Tạo môi trường ảo giúp tách biệt các thư viện của dự án, tránh xung đột với các dự án khác.

**Trên Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Trên macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

Khi kích hoạt thành công, terminal sẽ hiển thị `(venv)` ở đầu dòng.

**Tắt môi trường ảo khi xong:**
```bash
deactivate
```

### Bước 3 — Cài đặt thư viện

```bash
pip install -r requirements.txt
```

Lần đầu chạy sẽ mất vài phút để tải model (khoảng 350MB).

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
| http://localhost:8000/docs | Giao diện Swagger UI (test trực tiếp) |

---

## Hướng dẫn gọi API

### GET / — Thông tin API

**Request:**
```bash
curl http://localhost:8000/
```

**Response:**
```json
{"name":"Image Classification API","model":"google/vit-base-patch16-224"}
```

---

### GET /health — Kiểm tra trạng thái

**Request:**
```bash
curl http://localhost:8000/health
```

**Response:**
```json
{"status":"ok","model_loaded":true}
```

---

### POST /predict/url — Phân loại ảnh từ URL

**Request body:**

| Trường | Kiểu | Bắt buộc | Mô tả |
|--------|------|----------|-------|
| `url` | string | Có | Đường dẫn URL đến ảnh |
| `top_k` | integer | Không | Số lượng kết quả (1–>10, mặc định: 5) |

**Ví dụ:**
```bash
curl -X POST http://localhost:8000/predict/url -H "Content-Type: application/json" -d "{\"url\":\"https://upload.wikimedia.org/wikipedia/commons/thumb/1/14/Gatto_europeo4.jpg/1200px-Gatto_europeo4.jpg\",\"top_k\":3}"
```

**Response:**
```json
{"input_url":"https://upload.wikimedia.org/wikipedia/commons/thumb/1/14/Gatto_europeo4.jpg/1200px-Gatto_europeo4.jpg","top_k":3,"predictions":[{"label":"Egyptian cat","score":0.5545},{"label":"tiger cat","score":0.2216},{"label":"tabby, tabby cat","score":0.2158}]}
```

---

### POST /predict/base64 — Phân loại ảnh từ Base64

**Request body:**

| Trường | Kiểu | Bắt buộc | Mô tả |
|--------|------|----------|-------|
| `image_base64` | string | Có | Ảnh được mã hóa dạng base64 |
| `top_k` | integer | Không | Số lượng kết quả (1–10, mặc định: 5) |

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
{'top_k': 3, 'predictions': [{'label': 'Egyptian cat', 'score': 0.5545}, {'label': 'tiger cat', 'score': 0.2216}, {'label': 'tabby, tabby cat', 'score': 0.2158}]}
```

---

### Các trường hợp lỗi

| HTTP Status | Nguyên nhân | Response |
|-------------|-------------|----------|
| `400` | URL rỗng | `{"detail": "Truong url khong duoc de trong."}` |
| `400` | top_k ngoài khoảng 1–10 | `{"detail": "top_k phai tu 1 den 10."}` |
| `400` | Base64 không hợp lệ | `{"detail": "Chuoi base64 khong hop le."}` |
| `400` | Không tải được ảnh từ URL | `{"detail": "Khong the tai anh tu URL."}` |
| `503` | Model chưa được tải | `{"detail": "Model chua duoc tai."}` |
| `500` | Lỗi trong quá trình suy luận | `{"detail": "Loi khi phan loai: ..."}` |

---

## Chạy file kiểm thử

Đảm bảo server đang chạy, sau đó mở terminal mới và chạy:

```bash
# Kích hoạt môi trường ảo trước
source venv/bin/activate    # macOS/Linux
venv\Scripts\activate       # Windows

# Chạy file test
python test_api.py
```

---

## Video Demo


[![Watch the demo](https://img.youtube.com/vi/ACGv7_nw-Dc/0.jpg)](https://youtu.be/ACGv7_nw-Dc)
---

## Thư viện sử dụng

| Thư viện | Mục đích |
|----------|----------|
| `fastapi` | Framework xây dựng Web API |
| `uvicorn` | ASGI server để chạy FastAPI |
| `transformers` | Tải và sử dụng mô hình Hugging Face |
| `torch` | Backend tính toán cho mô hình |
| `Pillow` | Xử lý ảnh |
| `requests` | Gọi HTTP request |
| `pydantic` | Validate dữ liệu đầu vào |
