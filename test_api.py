import requests
import base64

BASE = "http://localhost:8000"

print("=" * 50)

# ── Test GET / ──────────────────────────────────────
r = requests.get(f"{BASE}/")
print("GET /")
print(r.json())
print()

# ── Test GET /health ────────────────────────────────
r = requests.get(f"{BASE}/health")
print("GET /health")
print(r.json())
print()

# ── Test POST /predict/url ──────────────────────────
print("POST /predict/url — Ảnh con chó")
r = requests.post(f"{BASE}/predict/url", json={
    "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/2/26/YellowLabradorLooking_new.jpg/1200px-YellowLabradorLooking_new.jpg",
    "top_k": 3
})
print(r.json())
print()

print("POST /predict/url — Ảnh con mèo")
r = requests.post(f"{BASE}/predict/url", json={
    "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Cat_November_2010-1a.jpg/1200px-Cat_November_2010-1a.jpg",
    "top_k": 3
})
print(r.json())
print()

# ── Test lỗi: URL rỗng ──────────────────────────────
print("POST /predict/url — Lỗi URL rỗng")
r = requests.post(f"{BASE}/predict/url", json={"url": ""})
print(f"Status: {r.status_code} | {r.json()}")
print()

# ── Test lỗi: URL sai ───────────────────────────────
print("POST /predict/url — Lỗi URL không tồn tại")
r = requests.post(f"{BASE}/predict/url", json={"url": "https://example.com/khong-ton-tai.jpg"})
print(f"Status: {r.status_code} | {r.json()}")
print()

# ── Test POST /predict/base64 ────────────────────────
print("POST /predict/base64 — Ảnh từ file local")
with open("test_image.webp", "rb") as f:        # chuẩn bị sẵn 1 ảnh tên test_image.webp
    encoded = base64.b64encode(f.read()).decode("utf-8")
r = requests.post(f"{BASE}/predict/base64", json={
    "image_base64": encoded,
    "top_k": 5
})
print(r.json())