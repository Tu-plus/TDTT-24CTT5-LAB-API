import requests
import base64
import json

# ================= CONFIG =================
BASE = "http://127.0.0.1:8000"  
# BASE = "https://your-pinggy-url.run.pinggy-free.link"

def pretty(response):
    print(f"Status: {response.status_code}")
    try:
        print(json.dumps(response.json(), indent=2, ensure_ascii=False))
    except:
        print(response.text)
    print()

print("=" * 50)

# ================= TEST 1 =================
print("TEST 1 — GET /")
r = requests.get(f"{BASE}/")
assert r.status_code == 200
data = r.json()
assert "name" in data
assert "model" in data
pretty(r)

# ================= TEST 2 =================
print("TEST 2 — GET /health")
r = requests.get(f"{BASE}/health")
assert r.status_code == 200
data = r.json()
assert "status" in data
assert "model_loaded" in data
pretty(r)

# ================= TEST 3 =================
print("TEST 3 — POST /predict/base64 (ảnh chó)")
headers = {"User-Agent": "Mozilla/5.0"}
img_bytes = requests.get(
    "https://upload.wikimedia.org/wikipedia/commons/thumb/2/26/YellowLabradorLooking_new.jpg/1200px-YellowLabradorLooking_new.jpg",
    headers=headers
).content

encoded = base64.b64encode(img_bytes).decode("utf-8")

r = requests.post(f"{BASE}/predict/base64", json={
    "image_base64": encoded,
    "top_k": 3
})

assert r.status_code == 200
data = r.json()
assert "predictions" in data
assert len(data["predictions"]) == 3
pretty(r)

# ================= TEST 4 =================
print("TEST 4 — POST /predict/base64 (ảnh mèo)")
img_bytes = requests.get(
    "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Cat_November_2010-1a.jpg/1200px-Cat_November_2010-1a.jpg",
    headers=headers
).content

encoded = base64.b64encode(img_bytes).decode("utf-8")

r = requests.post(f"{BASE}/predict/base64", json={
    "image_base64": encoded,
    "top_k": 3
})

assert r.status_code == 200
data = r.json()
assert "predictions" in data
assert len(data["predictions"]) == 3
pretty(r)

# ================= TEST 5 =================
print("TEST 5 — Lỗi: URL rỗng (expect 400)")
r = requests.post(f"{BASE}/predict/url", json={"url": ""})
assert r.status_code == 400
pretty(r)

# ================= TEST 6 =================
print("TEST 6 — Lỗi: top_k = 99 (expect 400)")
r = requests.post(f"{BASE}/predict/url", json={
    "url": "https://www.naturalhistoryonthenet.com/wp-content/uploads/2016/12/Domestic-Dog.jpg",
    "top_k": 99
})
assert r.status_code == 400
pretty(r)

# ================= TEST 7 =================
print("TEST — top_k = 1")
r = requests.post(f"{BASE}/predict/base64", json={
    "image_base64": encoded,
    "top_k": 1
})
assert r.status_code == 200
assert len(r.json()["predictions"]) == 1
pretty(r)

# ================= TEST 8 =================
print("TEST — top_k = 10")
r = requests.post(f"{BASE}/predict/base64", json={
    "image_base64": encoded,
    "top_k": 10
})
assert r.status_code == 200
assert len(r.json()["predictions"]) == 10
pretty(r)

# ================= TEST 9 =================
print("TEST — Ảnh xe hơi")
img_bytes = requests.get(
    "https://tse2.mm.bing.net/th/id/OIP.5VY1ONHuCPuHZRv-10n_ywHaFj",
    headers=headers
).content

encoded_car = base64.b64encode(img_bytes).decode("utf-8")

r = requests.post(f"{BASE}/predict/base64", json={
    "image_base64": encoded_car,
    "top_k": 3
})

assert r.status_code == 200
assert "predictions" in r.json()
pretty(r)

# ================= TEST 10 =================
print("TEST — Lỗi: base64 rỗng (expect 400)")
r = requests.post(f"{BASE}/predict/base64", json={
    "image_base64": ""
})
assert r.status_code == 400
pretty(r)

# ================= TEST 11 =================
print("TEST — Lỗi: base64 không phải ảnh (expect 400)")
fake = base64.b64encode(b"day khong phai anh").decode("utf-8")

r = requests.post(f"{BASE}/predict/base64", json={
    "image_base64": fake
})
assert r.status_code == 400
pretty(r)

# ================= TEST 12 =================
print("TEST — Lỗi: top_k = 0 (expect 400)")
r = requests.post(f"{BASE}/predict/url", json={
    "url": "https://images.dog.ceo/breeds/labrador/n02099712_7003.jpg",
    "top_k": 0
})
assert r.status_code == 400
pretty(r)

# ================= TEST 13 =================
print("TEST — Ảnh nhiều hoa")
img_bytes = requests.get(
    "https://tse2.mm.bing.net/th/id/OIP.tVDOjEq-v8EMzmbOGWIjNQHaEo",
    headers=headers
).content

encoded_flower = base64.b64encode(img_bytes).decode("utf-8")

r = requests.post(f"{BASE}/predict/base64", json={
    "image_base64": encoded_flower,
    "top_k": 3
})

assert r.status_code == 200
data = r.json()
assert "predictions" in data
assert len(data["predictions"]) == 3
pretty(r)

print("=" * 50)
print("ALL TESTS PASSED!")