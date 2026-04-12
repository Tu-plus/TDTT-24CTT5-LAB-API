import requests
import base64
import json

# ================= CONFIG =================
BASE = "http://127.0.0.1:8000"
# BASE = "https://your-pinggy-url.run.pinggy-free.link"  # thay URL mới khi chạy Colab

HEADERS = {"User-Agent": "Mozilla/5.0"}

passed = 0
failed = 0

def pretty(response):
    print(f"Status: {response.status_code}")
    try:
        print(json.dumps(response.json(), indent=2, ensure_ascii=False))
    except:
        print(response.text)
    print()

def check(condition, msg=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS {msg}")
    else:
        failed += 1
        print(f"  FAIL {msg}")

def load_image(url):
    """Tải ảnh từ URL về bytes"""
    return requests.get(url, headers=HEADERS).content

def to_base64(img_bytes):
    """Chuyển bytes sang chuỗi base64"""
    return base64.b64encode(img_bytes).decode("utf-8")

print("=" * 50)
print("CHẠY FILE KIỂM THỬ API")
print(f"Server: {BASE}")
print("=" * 50)
print()

# ================= CHUẨN BỊ ẢNH =================
print("Đang tải ảnh...")
encoded_dog    = to_base64(load_image("https://upload.wikimedia.org/wikipedia/commons/thumb/2/26/YellowLabradorLooking_new.jpg/1200px-YellowLabradorLooking_new.jpg"))
encoded_cat    = to_base64(load_image("https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Cat_November_2010-1a.jpg/1200px-Cat_November_2010-1a.jpg"))
encoded_car    = to_base64(load_image("https://tse2.mm.bing.net/th/id/OIP.5VY1ONHuCPuHZRv-10n_ywHaFj?rs=1&pid=ImgDetMain&o=7&rm=3"))
encoded_flower = to_base64(load_image("https://upload.wikimedia.org/wikipedia/commons/thumb/4/41/Sunflower_from_Silesia2.jpg/1200px-Sunflower_from_Silesia2.jpg"))
encoded_multi  = to_base64(load_image("https://kingspet.vn/wp-content/uploads/2023/07/dog-cat-under-sheet-1400x788.jpg"))
print("Tải ảnh xong!")
print()

# ================= TEST 1 =================
print("TEST 1 — GET /")
r = requests.get(f"{BASE}/")
pretty(r)
check(r.status_code == 200, "status 200")
check("name" in r.json(), "có trường 'name'")
check("model" in r.json(), "có trường 'model'")

# ================= TEST 2 =================
print("TEST 2 — GET /health")
r = requests.get(f"{BASE}/health")
pretty(r)
check(r.status_code == 200, "status 200")
check("status" in r.json(), "có trường 'status'")
check("model_loaded" in r.json(), "có trường 'model_loaded'")
check(r.json()["model_loaded"] == True, "model đã load")

# ================= TEST 3 =================
print("TEST 3 — POST /predict/base64 (ảnh chó)")
r = requests.post(f"{BASE}/predict/base64", json={"image_base64": encoded_dog, "top_k": 3})
pretty(r)
check(r.status_code == 200, "status 200")
check("predictions" in r.json(), "có trường 'predictions'")
check(len(r.json()["predictions"]) == 3, "trả về đúng 3 kết quả")

# ================= TEST 4 =================
print("TEST 4 — POST /predict/base64 (ảnh mèo)")
r = requests.post(f"{BASE}/predict/base64", json={"image_base64": encoded_cat, "top_k": 3})
pretty(r)
check(r.status_code == 200, "status 200")
check("predictions" in r.json(), "có trường 'predictions'")
check(len(r.json()["predictions"]) == 3, "trả về đúng 3 kết quả")

# ================= TEST 5 =================
print("TEST 5 — Lỗi: URL rỗng (expect 400)")
r = requests.post(f"{BASE}/predict/url", json={"url": ""})
pretty(r)
check(r.status_code == 400, "status 400")

# ================= TEST 6 =================
print("TEST 6 — Lỗi: top_k = 99 (expect 400)")
r = requests.post(f"{BASE}/predict/url", json={
    "url": "https://images.dog.ceo/breeds/labrador/n02099712_7003.jpg",
    "top_k": 99
})
pretty(r)
check(r.status_code == 400, "status 400")

# ================= TEST 7 =================
print("TEST 7 — top_k = 1 (chỉ lấy kết quả tốt nhất)")
r = requests.post(f"{BASE}/predict/base64", json={"image_base64": encoded_dog, "top_k": 1})
pretty(r)
check(r.status_code == 200, "status 200")
check(len(r.json()["predictions"]) == 1, "trả về đúng 1 kết quả")

# ================= TEST 8 =================
print("TEST 8 — top_k = 10 (lấy tối đa)")
r = requests.post(f"{BASE}/predict/base64", json={"image_base64": encoded_dog, "top_k": 10})
pretty(r)
check(r.status_code == 200, "status 200")
check(len(r.json()["predictions"]) == 10, "trả về đúng 10 kết quả")

# ================= TEST 9 =================
print("TEST 9 — Ảnh xe hơi (vật thể không phải động vật)")
r = requests.post(f"{BASE}/predict/base64", json={"image_base64": encoded_car, "top_k": 3})
pretty(r)
check(r.status_code == 200, "status 200")
check("predictions" in r.json(), "có trường 'predictions'")

# ================= TEST 10 =================
print("TEST 10 — Lỗi: base64 rỗng (expect 400)")
r = requests.post(f"{BASE}/predict/base64", json={"image_base64": ""})
pretty(r)
check(r.status_code == 400, "status 400")

# ================= TEST 11 =================
print("TEST 11 — Lỗi: base64 không phải ảnh (expect 400)")
fake = base64.b64encode(b"day khong phai anh").decode("utf-8")
r = requests.post(f"{BASE}/predict/base64", json={"image_base64": fake})
pretty(r)
check(r.status_code == 400, "status 400")

# ================= TEST 12 =================
print("TEST 12 — Lỗi: top_k = 0 (expect 400)")
r = requests.post(f"{BASE}/predict/url", json={
    "url": "https://images.dog.ceo/breeds/labrador/n02099712_7003.jpg",
    "top_k": 0
})
pretty(r)
check(r.status_code == 400, "status 400")

# ================= TEST 13 =================
print("TEST 13 — Ảnh hoa (model nhận nhãn gần nhất vì hoa không có trong ImageNet)")
r = requests.post(f"{BASE}/predict/base64", json={"image_base64": encoded_flower, "top_k": 3})
pretty(r)
check(r.status_code == 200, "status 200")
check("predictions" in r.json(), "có trường 'predictions'")

# ================= TEST 14 =================
print("TEST 14 — Ảnh nhiều vật thể (chó + mèo + khăn)")
print("         Model sẽ nhận vật thể chiếm diện tích lớn nhất")
r = requests.post(f"{BASE}/predict/base64", json={"image_base64": encoded_multi, "top_k": 3})
pretty(r)
check(r.status_code == 200, "status 200")
check("predictions" in r.json(), "có trường 'predictions'")

# ================= TỔNG KẾT =================
print("=" * 50)
print(f"TONG KET: {passed} PASS — {failed} FAIL")
if failed == 0:
    print("ALL TESTS PASSED!")
else:
    print(f"CO {failed} TEST BI LOI, kiem tra lai server.")
print("=" * 50)
