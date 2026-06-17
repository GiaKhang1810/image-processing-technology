# Image Processing Technology - OCR Receipt Recognition

## 1. Giới thiệu đề tài

Project này xây dựng một hệ thống **OCR nhận dạng chữ trên ảnh biên lai/hóa đơn**. Ảnh đầu vào được tiền xử lý bằng thư viện **Pillow**, sau đó đưa vào model OCR để trích xuất nội dung văn bản.

Mục tiêu chính của project:

- Đọc chữ từ ảnh biên lai/hóa đơn.
- So sánh kết quả OCR với dữ liệu nhãn đúng trong `labels.csv`.
- Thử nhiều kiểu tiền xử lý ảnh để tìm cấu hình tốt nhất.
- Đánh giá kết quả bằng chỉ số **CER** và **Accuracy**.
- So sánh hai model OCR: **EasyOCR** và **Tesseract OCR**.

> Lưu ý: Trong project này, bước `training` không phải là fine-tune model deep learning thật. Bước này dùng tập train để tìm cấu hình tiền xử lý ảnh tốt nhất, ví dụ `raw`, `gray`, `contrast`, `binary` và các giá trị `threshold` khác nhau.

---

## 2. Công nghệ sử dụng

Project sử dụng các thư viện chính:

| Thư viện | Chức năng |
|---|---|
| Pillow | Đọc ảnh, resize, grayscale, tăng tương phản, làm nét, threshold |
| EasyOCR | Nhận dạng ký tự tiếng Việt và tiếng Anh |
| pytesseract | Kết nối Python với Tesseract OCR |
| pandas | Đọc và ghi file CSV |
| scikit-learn | Chia dữ liệu train/test |
| numpy | Chuyển ảnh Pillow sang array cho EasyOCR |

---

## 3. Cấu trúc thư mục

```text
image_processing_technology/
│
├── dataset/
│   ├── images/
│   │   ├── receipt_001.jpg
│   │   ├── receipt_002.jpg
│   │   └── ...
│   │
│   └── labels.csv
│
├── output/
│   ├── processed/
│   ├── easyocr.csv
│   ├── easyocr_summary.csv
│   ├── tesseract.csv
│   └── tesseract_summary.csv
│
├── errorlog/
│   └── yyyy-mm-dd_hh-mm-ss.log
│
├── gendata.py
├── main.py
├── metrics.py
├── model.py
├── preprocess.py
├── options.json
├── requirements.txt
└── README.md
```

Ý nghĩa các file chính:

| File | Chức năng |
|---|---|
| `main.py` | Chạy toàn bộ pipeline OCR, chia train/test, tìm cấu hình xử lý ảnh tốt nhất và xuất kết quả |
| `model.py` | Quản lý các model OCR gồm EasyOCR và Tesseract |
| `preprocess.py` | Tiền xử lý ảnh bằng Pillow |
| `metrics.py` | Tính CER và Accuracy |
| `gendata.py` | Sinh dữ liệu biên lai mẫu |
| `options.json` | Danh sách các cấu hình xử lý ảnh cần thử nghiệm |
| `dataset/labels.csv` | Chứa nhãn đúng của từng ảnh |
| `output/*.csv` | Chứa kết quả OCR sau khi chạy |
| `errorlog/` | Lưu lỗi nếu chương trình chạy thất bại |

---

## 4. Dataset

Dataset nằm trong thư mục:

```text
dataset/
├── images/
└── labels.csv
```

Trong đó:

- `dataset/images/` chứa ảnh biên lai/hóa đơn.
- `dataset/labels.csv` chứa nội dung đúng tương ứng với từng ảnh.

Cấu trúc file `labels.csv`:

```csv
filename,text
receipt_001.jpg,"BIEN LAI THANH TOAN | Mon: Ca phe sua | Gia: 56000 VND | Ngay: 17/06/2026 | Cam on quy khach!"
receipt_002.jpg,"BIEN LAI THANH TOAN | Mon: Tra da | Gia: 18000 VND | Ngay: 17/06/2026 | Cam on quy khach!"
```

Ý nghĩa:

| Cột | Ý nghĩa |
|---|---|
| `filename` | Tên ảnh trong thư mục `dataset/images/` |
| `text` | Nội dung đúng dùng để so sánh với kết quả OCR |

---

## 5. Cài đặt môi trường

Nên dùng Python **3.11** để tránh lỗi tương thích thư viện OCR.

### 5.1. Tạo môi trường ảo

```powershell
py -3.11 -m venv .venv
```

Kích hoạt môi trường ảo trên Windows PowerShell:

```powershell
.\.venv\Scripts\activate
```

Kiểm tra Python đang dùng:

```powershell
python --version
where python
```

Đường dẫn Python nên trỏ về thư mục `.venv` của project.

---

### 5.2. Cài thư viện

Nếu đã có `requirements.txt`, chạy:

```powershell
python -m pip install -r requirements.txt
```

Nếu chưa có `requirements.txt`, có thể cài thủ công:

```powershell
python -m pip install pillow numpy pandas scikit-learn easyocr pytesseract opencv-python-headless
```

Nếu muốn EasyOCR dùng GPU NVIDIA, cài PyTorch bản CUDA:

```powershell
python -m pip uninstall torch torchvision torchaudio -y
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Kiểm tra PyTorch có nhận GPU không:

```powershell
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"
```

Nếu kết quả là `True` và hiện tên GPU thì EasyOCR có thể chạy bằng GPU. Nếu không, chương trình vẫn chạy được bằng CPU nhưng tốc độ sẽ chậm hơn.

---

## 6. Cài đặt Tesseract OCR

Model Tesseract cần cài thêm phần mềm **Tesseract OCR** trên Windows. Việc cài `pytesseract` bằng pip chỉ là thư viện Python để gọi Tesseract, chưa phải OCR engine thật.

Sau khi cài Tesseract OCR, cần đảm bảo lệnh này chạy được trong terminal:

```powershell
tesseract --version
```

Nếu dùng tiếng Việt, cần có dữ liệu ngôn ngữ `vie.traineddata`. Khi chạy Tesseract trong project, code sử dụng:

```python
lang="vie+eng"
```

Nghĩa là Tesseract sẽ đọc cả tiếng Việt và tiếng Anh.

---

## 7. Cách chạy project

### 7.1. Chạy với EasyOCR

```powershell
python main.py --model easyocr
```

Kết quả sẽ được lưu vào:

```text
output/easyocr.csv
output/easyocr_summary.csv
```

---

### 7.2. Chạy với Tesseract OCR

```powershell
python main.py --model tesseract
```

Kết quả sẽ được lưu vào:

```text
output/tesseract.csv
output/tesseract_summary.csv
```

---

## 8. Quy trình hoạt động

Khi chạy `main.py`, chương trình sẽ thực hiện các bước:

1. Đọc dữ liệu từ `dataset/labels.csv`.
2. Chia dữ liệu thành 80% train và 20% test.
3. Đọc danh sách cấu hình xử lý ảnh trong `options.json`.
4. Với từng cấu hình, chương trình xử lý ảnh bằng Pillow.
5. Đưa ảnh đã xử lý vào model OCR.
6. So sánh kết quả OCR với nhãn đúng.
7. Tính CER và Accuracy.
8. Chọn cấu hình xử lý ảnh tốt nhất trên tập train.
9. Đánh giá lại trên tập test.
10. Xuất kết quả chi tiết và kết quả tổng kết ra thư mục `output`.

---

## 9. Các chế độ tiền xử lý ảnh

Các chế độ xử lý ảnh được định nghĩa trong `preprocess.py` và cấu hình trong `options.json`.

| Mode | Ý nghĩa |
|---|---|
| `raw` | Giữ ảnh gốc, không xử lý thêm |
| `gray` | Resize ảnh và chuyển sang ảnh xám |
| `contrast` | Resize, grayscale, tăng tương phản và làm nét |
| `binary` | Resize, grayscale, tăng tương phản, làm nét, khử nhiễu và threshold trắng đen |

Ví dụ cấu hình trong `options.json`:

```json
{
  "mode": "contrast",
  "threshold": 100
}
```

Với `mode` là `binary`, giá trị `threshold` sẽ quyết định điểm cắt trắng đen của ảnh. Với các mode khác, `threshold` chủ yếu được dùng để ghi nhận cấu hình khi xuất kết quả.

---

## 10. Metrics đánh giá

Project sử dụng hai chỉ số chính:

### 10.1. CER

CER là viết tắt của **Character Error Rate**, tức tỷ lệ lỗi theo ký tự.

Công thức:

```text
CER = số lỗi ký tự / tổng số ký tự đúng
```

CER càng thấp thì OCR càng tốt.

Ví dụ:

```text
Expected: BIEN LAI THANH TOAN
Predicted: BIEN LAI THANH TOAM
```

Model chỉ sai một ký tự, nên CER sẽ thấp.

---

### 10.2. Accuracy

Accuracy trong project được tính gần đúng từ CER:

```text
Accuracy = max(0, 1 - CER)
```

Accuracy càng cao thì model đọc càng tốt.

---

## 11. Chuẩn hóa text khi đánh giá

Trong `metrics.py`, text được chuẩn hóa trước khi tính CER:

- Xóa dấu tiếng Việt.
- Chuyển về chữ thường.
- Xóa ký tự đặc biệt.
- Xóa khoảng trắng dư.

Việc này giúp đánh giá công bằng hơn, vì OCR có thể đọc đúng nội dung nhưng khác dấu, khác chữ hoa/thường hoặc khác ký tự phân tách.

Ví dụ:

```text
Cà phê sữa
ca phe sua
```

Sau chuẩn hóa, hai chuỗi trên được xem là gần giống nhau hơn.

---

## 12. Kết quả đầu ra

Sau khi chạy model, thư mục `output/` sẽ có các file:

```text
output/
├── easyocr.csv
├── easyocr_summary.csv
├── tesseract.csv
├── tesseract_summary.csv
└── processed/
```

### 12.1. File kết quả chi tiết

Ví dụ `output/easyocr.csv` hoặc `output/tesseract.csv`:

| Cột | Ý nghĩa |
|---|---|
| `filename` | Tên ảnh test |
| `model` | Model OCR đã dùng |
| `mode` | Kiểu xử lý ảnh tốt nhất |
| `threshold` | Giá trị threshold |
| `expected` | Text đúng từ labels.csv |
| `predicted` | Text OCR đọc được |
| `cer` | Character Error Rate |
| `accuracy` | Độ chính xác gần đúng |

### 12.2. File tổng kết

Ví dụ `output/easyocr_summary.csv`:

| Cột | Ý nghĩa |
|---|---|
| `model` | Model OCR |
| `best_mode` | Mode xử lý ảnh tốt nhất |
| `best_threshold` | Threshold tốt nhất |
| `train_accuracy` | Accuracy trung bình trên tập train |
| `test_cer` | CER trung bình trên tập test |
| `test_accuracy` | Accuracy trung bình trên tập test |

File summary là file nên dùng để đưa vào báo cáo vì gọn và dễ đọc.

---

## 13. Sinh lại dataset mẫu

Project có file `gendata.py` để sinh ảnh biên lai mẫu.

Trước khi chạy, đảm bảo thư mục `dataset/images` đã tồn tại:

```powershell
mkdir dataset\images
```

Chạy lệnh:

```powershell
python gendata.py
```

Script sẽ tạo 100 ảnh biên lai mẫu và file `dataset/labels.csv`.

Các hiệu ứng ảnh được tạo ngẫu nhiên gồm:

- Làm mờ ảnh.
- Giảm sáng.
- Thêm vùng chói sáng.
- Làm méo phối cảnh.

Mục đích là mô phỏng ảnh hóa đơn/biên lai chụp trong nhiều điều kiện khác nhau.

---

## 14. Xử lý lỗi

Nếu chương trình gặp lỗi, lỗi sẽ được ghi vào thư mục:

```text
errorlog/
```

Tên file lỗi có dạng:

```text
2026-06-18_01-50-20.log
```

Mở file log để xem traceback chi tiết.

---

## 15. Một số lỗi thường gặp

### 15.1. EasyOCR báo chạy CPU

Thông báo thường gặp:

```text
Neither CUDA nor MPS are available - defaulting to CPU
```

Nguyên nhân là PyTorch chưa nhận GPU CUDA. Kiểm tra bằng lệnh:

```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

Nếu kết quả là `False`, cần cài lại PyTorch bản CUDA.

---

### 15.2. TesseractNotFoundError

Nguyên nhân là máy chưa cài phần mềm Tesseract OCR hoặc chưa thêm Tesseract vào PATH.

Cách kiểm tra:

```powershell
tesseract --version
```

Nếu lệnh không chạy, cần cài Tesseract OCR trên Windows.

---

### 15.3. CSV bị xuống dòng lộn xộn

Kết quả OCR có thể chứa ký tự `\n`, `\r`, `\t`, làm file CSV hiển thị lộn xộn. Project đã xử lý bằng hàm `clean_text_for_csv()` trong `main.py`, sau đó lưu CSV với `quoting=QUOTE_ALL` để tránh vỡ dòng và vỡ cột.

---

## 16. Xuất requirements.txt

Sau khi cài đủ thư viện và chạy ổn định, xuất file requirements:

```powershell
python -m pip freeze > requirements.txt
```

Người khác có thể cài lại bằng:

```powershell
python -m pip install -r requirements.txt
```

Không nên đưa thư mục `.venv` vào file nộp.

---

## 17. Các file không nên nộp

Khi nén project để nộp, có thể bỏ các thư mục/file sau:

```text
.venv/
__pycache__/
*.pyc
output/processed/
errorlog/
.git/
```

Các file nên nộp:

```text
main.py
model.py
preprocess.py
metrics.py
gendata.py
options.json
requirements.txt
README.md
dataset/labels.csv
dataset/images/ một số ảnh mẫu
output/easyocr_summary.csv
output/tesseract_summary.csv
```

---

## 18. Kết luận

Project đã xây dựng được pipeline OCR cơ bản cho ảnh biên lai/hóa đơn. Hệ thống có thể tiền xử lý ảnh bằng Pillow, chạy OCR bằng EasyOCR hoặc Tesseract, sau đó đánh giá kết quả bằng CER và Accuracy.

Kết quả có thể tiếp tục cải thiện bằng cách:

- Tăng số lượng ảnh trong dataset.
- Dùng ảnh hóa đơn thật thay vì ảnh sinh tự động.
- Gán nhãn dữ liệu chính xác hơn.
- Tách riêng các trường như tên món, giá tiền, ngày tháng.
- Fine-tune model OCR chuyên cho tiếng Việt nếu có đủ dữ liệu.

## Thông tin tác giả

* **Tác giả:** [GiaKhang](https://github.com/GiaKhang1810), ThanhTan, MinhQuan, KhanhDuy, CongHuy
* **Tên project:** Image Processing Technology - OCR Receipt Recognition
* **Mục đích:** Project được xây dựng phục vụ học tập, nghiên cứu và demo bài toán OCR nhận dạng chữ trên ảnh hóa đơn/biên lai.
* **Công nghệ sử dụng:** Python, Pillow, EasyOCR, Tesseract OCR, Pandas, Scikit-learn.

## Quyền sử dụng và bản quyền

  Project này được phát triển cho mục đích học tập và báo cáo môn học.
  Toàn bộ source code, cấu trúc xử lý ảnh và quy trình đánh giá model thuộc về tác giả của project.

  Người dùng có thể tham khảo, chỉnh sửa và chạy lại project cho mục đích học tập hoặc nghiên cứu cá nhân.
  Không được sử dụng project này cho mục đích thương mại, sao chép để nộp lại như sản phẩm của người khác hoặc phân phối lại khi chưa có sự đồng ý của tác giả.

  Dataset mẫu trong project chỉ dùng để thử nghiệm mô hình OCR.
  Nếu sử dụng ảnh hóa đơn/biên lai thật, cần đảm bảo không chứa thông tin cá nhân nhạy cảm hoặc dữ liệu riêng tư của người khác.

  © 2026 Gia Khang, ThanhTan, MinhQuan, KhanhDuy, CongHuy. All rights reserved.