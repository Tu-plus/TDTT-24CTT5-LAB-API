
import requests
import base64
import json

BASE = "https://your-pinggy-url.run.pinggy-free.link"  # Thay URL mới khi chạy Colab
def pretty(response):
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2, ensure_ascii=False))
    print()

print("=" * 50)

# Test 1: GET /
print("TEST 1 — GET /")
r = requests.get(f"{BASE}/")
pretty(r)

# Test 2: GET /health
print("TEST 2 — GET /health")
r = requests.get(f"{BASE}/health")
pretty(r)

# Test 3: POST /predict/url - ảnh chó (base64)
print("TEST 3 — POST /predict/base64 (ảnh chó)")
headers = {"User-Agent": "Mozilla/5.0"}
img_bytes = requests.get(
    "https://upload.wikimedia.org/wikipedia/commons/thumb/2/26/YellowLabradorLooking_new.jpg/1200px-YellowLabradorLooking_new.jpg",
    headers=headers
).content
encoded = base64.b64encode(img_bytes).decode("utf-8")
r = requests.post(f"{BASE}/predict/base64", json={"image_base64": encoded, "top_k": 3})
pretty(r)

# Test 4: POST /predict/base64 - ảnh mèo
print("TEST 4 — POST /predict/base64 (ảnh mèo)")
img_bytes = requests.get(
    "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Cat_November_2010-1a.jpg/1200px-Cat_November_2010-1a.jpg",
    headers=headers
).content
encoded = base64.b64encode(img_bytes).decode("utf-8")
r = requests.post(f"{BASE}/predict/base64", json={"image_base64": encoded, "top_k": 3})
pretty(r)

# Test 5: Lỗi - URL rỗng
print("TEST 5 — Lỗi: URL rỗng (expect 400)")
r = requests.post(f"{BASE}/predict/url", json={"url": ""})
pretty(r)

# Test 6: Lỗi - top_k sai
print("TEST 6 — Lỗi: top_k = 99 (expect 400)")
r = requests.post(f"{BASE}/predict/url", json={
    "url": "https://images.dog.ceo/breeds/labrador/n02099712_7003.jpg",
    "top_k": 99
})
pretty(r)
