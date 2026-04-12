
import requests
import base64
import json

BASE = "http://127.0.0.1:8000"  # local
# BASE = "https://...pinggy..." # public (thay URL mới khi chạy collab)
def pretty(response):
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2, ensure_ascii=False))
    print()

print("=" * 50)

#-----------------------------------------------------------------------------------

# Test 1: GET /
print("TEST 1 — GET /")
r = requests.get(f"{BASE}/")
pretty(r)

#-----------------------------------------------------------------------------------

# Test 2: GET /health
print("TEST 2 — GET /health")
r = requests.get(f"{BASE}/health")
pretty(r)

#-----------------------------------------------------------------------------------

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

#-----------------------------------------------------------------------------------

# Test 4: POST /predict/base64 - ảnh mèo
print("TEST 4 — POST /predict/base64 (ảnh mèo)")
img_bytes = requests.get(
    "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Cat_November_2010-1a.jpg/1200px-Cat_November_2010-1a.jpg",
    headers=headers
).content
encoded = base64.b64encode(img_bytes).decode("utf-8")
r = requests.post(f"{BASE}/predict/base64", json={"image_base64": encoded, "top_k": 3})
pretty(r)

#-----------------------------------------------------------------------------------


# Test 5: Lỗi - URL rỗng
print("TEST 5 — Lỗi: URL rỗng (expect 400)")
r = requests.post(f"{BASE}/predict/url", json={"url": ""})
pretty(r)

#-----------------------------------------------------------------------------------


# Test 6: Lỗi - top_k sai
print("TEST 6 — Lỗi: top_k = 99 (expect 400)")
r = requests.post(f"{BASE}/predict/url", json={
    "url": "https://www.naturalhistoryonthenet.com/wp-content/uploads/2016/12/Domestic-Dog.jpg",
    "top_k": 99
})
pretty(r)

#-----------------------------------------------------------------------------------

# Test top_k = 1 — chỉ lấy kết quả tốt nhất
print("TEST — top_k = 1")
r = requests.post(f"{BASE}/predict/base64", json={
    "image_base64": encoded,
    "top_k": 1
})
pretty(r)

#-----------------------------------------------------------------------------------

# Test top_k = 10 — lấy tối đa
print("TEST — top_k = 10")
r = requests.post(f"{BASE}/predict/base64", json={
    "image_base64": encoded,
    "top_k": 10
})
pretty(r)

#-----------------------------------------------------------------------------------

# Test ảnh không phải động vật — ví dụ xe hơi
print("TEST — Ảnh xe hơi")
img_bytes = requests.get(
    "https://tse2.mm.bing.net/th/id/OIP.5VY1ONHuCPuHZRv-10n_ywHaFj?rs=1&pid=ImgDetMain&o=7&rm=3",
    headers={"User-Agent": "Mozilla/5.0"}
).content
encoded_car = base64.b64encode(img_bytes).decode("utf-8")
r = requests.post(f"{BASE}/predict/base64", json={
    "image_base64": encoded_car,
    "top_k": 3
})
pretty(r)

#-----------------------------------------------------------------------------------

# Test base64 rỗng
print("TEST — Lỗi: base64 rỗng (expect 400)")
r = requests.post(f"{BASE}/predict/base64", json={
    "image_base64": ""
})
pretty(r)

#-----------------------------------------------------------------------------------

# Test base64 không phải ảnh — chuỗi bình thường encode lên
print("TEST — Lỗi: base64 không phải ảnh (expect 400)")
fake = base64.b64encode(b"day khong phai anh").decode("utf-8")
r = requests.post(f"{BASE}/predict/base64", json={
    "image_base64": fake
})
pretty(r)

#-----------------------------------------------------------------------------------

# Test top_k = 0 — dưới giới hạn
print("TEST — Lỗi: top_k = 0 (expect 400)")
r = requests.post(f"{BASE}/predict/url", json={
    "url": "https://images.dog.ceo/breeds/labrador/n02099712_7003.jpg",
    "top_k": 0
})
pretty(r)

#-----------------------------------------------------------------------------------

# Test ảnh nhận sai — ảnh hoa (không có trong 1000 lớp ImageNet)
print("TEST — ảnh hoa (model sẽ nhận nhãn gần nhất)")
img_bytes = requests.get(
    "https://tse2.mm.bing.net/th/id/OIP.tVDOjEq-v8EMzmbOGWIjNQHaEo?o=7rm=3&rs=1&pid=ImgDetMain&o=7&rm=3",
    headers={"User-Agent": "Mozilla/5.0"}
).content
encoded_person = base64.b64encode(img_bytes).decode("utf-8")
r = requests.post(f"{BASE}/predict/base64", json={
    "image_base64": encoded_person,
    "top_k": 3
})
pretty(r)

#-----------------------------------------------------------------------------------

# Test ảnh có nhiều hơn 2 vật thể — ảnh gồm chó, khăn và mèo (đều có trong 1000 lớp ImageNet)
# Chú chó có diện tích lớn hơn chú mèo trong ảnh

print("TEST — ảnh có nhiều hơn 2 vật thể")
img_bytes = requests.get(
    "https://kingspet.vn/wp-content/uploads/2023/07/dog-cat-under-sheet-1400x788.jpg",
    headers={"User-Agent": "Mozilla/5.0"}
).content
encoded_person = base64.b64encode(img_bytes).decode("utf-8")
r = requests.post(f"{BASE}/predict/base64", json={
    "image_base64": encoded_person,
    "top_k": 3
})
pretty(r)