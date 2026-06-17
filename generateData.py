import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import csv
import random
import os

# Tạo thư mục chứa ảnh
output_dir = "dataset"
os.makedirs(output_dir, exist_ok=True)

# Danh sách từ vựng ngẫu nhiên để tạo hóa đơn
items = ["Ca phe sua", "Tra da", "Banh mi", "Pho bo", "Com tam", "Sinh to bo", "Sinh sau rieng", "Ca phe da", "Pho ga", "Com tam", "Banh bao", "Xuc xich", " Sinh to dau"]
csv_data = [["filename", "text"]]

def generate_base_receipt(text):
    """Tạo một ảnh hóa đơn cơ bản nền trắng chữ đen."""
    img = Image.new('RGB', (400, 300), color=(255, 255, 255))
    d = ImageDraw.Draw(img)
    # Lưu ý: Nếu muốn test tiếng Việt có dấu tốt hơn, bạn hãy tải một file font .ttf (ví dụ arial.ttf) và đổi đường dẫn dưới đây.
    # font = ImageFont.truetype("arial.ttf", 20) 
    font = ImageFont.load_default()
    
    d.text((20, 20), "BIEN LAI THANH TOAN", fill=(0,0,0), font=font)
    d.text((20, 60), text, fill=(0,0,0), font=font)
    d.text((20, 250), "Cam on quy khach!", fill=(0,0,0), font=font)
    
    return np.array(img)

def apply_blur(img):
    """Làm mờ ảnh ngẫu nhiên."""
    k = random.choice([3, 5, 7])
    return cv2.GaussianBlur(img, (k, k), 0)

def apply_low_light(img):
    """Giảm độ sáng (thiếu sáng)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hsv = np.array(hsv, dtype = np.float64)
    hsv[:,:,2] = hsv[:,:,2] * random.uniform(0.3, 0.6) # Giảm Value
    hsv[:,:,2][hsv[:,:,2] > 255]  = 255
    hsv = np.array(hsv, dtype = np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

def apply_glare(img):
    """Thêm vệt sáng chói ngẫu nhiên."""
    overlay = img.copy()
    output = img.copy()
    h, w = img.shape[:2]
    # Vẽ một hình tròn trắng lớn với viền mờ để giả lập đèn flash/chói
    cx, cy = random.randint(0, w), random.randint(0, h)
    radius = random.randint(50, 150)
    cv2.circle(overlay, (cx, cy), radius, (255, 255, 255), -1)
    
    cv2.addWeighted(overlay, 0.5, output, 0.5, 0, output)
    return output

def apply_perspective_distortion(img):
    """Làm méo ảnh (perspective)."""
    h, w = img.shape[:2]
    
    # Tọa độ gốc
    pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    
    # Tọa độ ngẫu nhiên bị xô lệch
    pad = random.randint(10, 40)
    pts2 = np.float32([
        [0 + random.randint(0, pad), 0 + random.randint(0, pad)], 
        [w - random.randint(0, pad), 0 + random.randint(0, pad)], 
        [0 + random.randint(0, pad), h - random.randint(0, pad)], 
        [w - random.randint(0, pad), h - random.randint(0, pad)]
    ])
    
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(img, matrix, (w, h), borderValue=(200, 200, 200))
    return result

# Tạo 200 ảnh
for i in range(1, 201):
    # 1. Sinh nội dung text ngẫu nhiên
    item_name = random.choice(items)
    price = random.randint(15, 100) * 1000
    receipt_text = f"Mon: {item_name}\nGia: {price} VND\nNgay: {random.randint(1,28)}/10/2026"
    
    # Lưu text đầy đủ cho CSV (bạn có thể điều chỉnh để chỉ lấy nội dung cần thiết)
    full_text_for_csv = f"BIEN LAI THANH TOAN | Mon: {item_name} | Gia: {price} VND | Ngay: 17/06/2026 | Cam on quy khach!"
    
    # 2. Tạo ảnh base
    img = generate_base_receipt(receipt_text)
    
    # 3. Áp dụng hiệu ứng ngẫu nhiên
    effects = random.sample(["blur", "dark", "glare", "distort"], k=random.randint(1, 3))
    
    if "blur" in effects:
        img = apply_blur(img)
    if "dark" in effects:
        img = apply_low_light(img)
    if "glare" in effects:
        img = apply_glare(img)
    if "distort" in effects:
        img = apply_perspective_distortion(img)
        
    # 4. Lưu ảnh
    filename = f"receipt_{i:03d}.jpg"
    filepath = os.path.join(output_dir, "images", filename)
    cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    
    # 5. Thêm vào data CSV
    csv_data.append([filename, full_text_for_csv])

# Ghi ra file CSV
csv_path = os.path.join(output_dir, "labels.csv")
with open(csv_path, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerows(csv_data)

print(f"Đã tạo thành công 200 ảnh và file labels.csv tại thư mục: {output_dir}")