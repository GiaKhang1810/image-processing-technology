# Image Processing Technology - OCR Receipt Recognition

**Version:** 1.0.5
**Status:** Stable
**Author:** [Gia Khang](https://github.com/GiaKhang1810)
**Purpose:** Course project, study, research and OCR demonstration.

---

## 1. Giới thiệu

Project này xây dựng một hệ thống **OCR nhận dạng chữ trên ảnh biên lai/hóa đơn**.

Ảnh đầu vào được tiền xử lý bằng **Pillow**, sau đó đưa vào model OCR để trích xuất nội dung văn bản. Project hỗ trợ thử nghiệm nhiều phương pháp tiền xử lý ảnh như `raw`, `gray`, `contrast`, `binary`, `background`, từ đó chọn ra cấu hình có kết quả tốt nhất.

Project hiện hỗ trợ hai model OCR:

* **EasyOCR**
* **Tesseract OCR**

PaddleOCR không được dùng trong bản cuối để tránh lỗi tương thích môi trường trên Windows.

---

## 2. Mục tiêu đề tài

Mục tiêu chính của project:

* Tạo dataset ảnh biên lai/hóa đơn mẫu.
* Đọc chữ từ ảnh biên lai/hóa đơn.
* Tiền xử lý ảnh bằng Pillow trước khi OCR.
* Thử nhiều cấu hình tiền xử lý ảnh.
* So sánh kết quả OCR với nhãn đúng trong `labels.csv`.
* Đánh giá model bằng **CER** và **Accuracy**.
* Đo thời gian xử lý ảnh và thời gian OCR.
* Xuất kết quả chi tiết và kết quả tổng kết ra file CSV.

Lưu ý: Trong project này, bước `training` không phải là fine-tune model deep learning thật. Bước này dùng tập train để tìm cấu hình tiền xử lý ảnh tốt nhất.

---

## 3. Công nghệ sử dụng

| Công nghệ         | Chức năng                                |
| ----------------- | ---------------------------------------- |
| Python            | Ngôn ngữ chính của project               |
| Pillow            | Đọc ảnh và tiền xử lý ảnh                |
| NumPy             | Chuyển đổi ảnh sang mảng số              |
| OpenCV            | Tạo dataset ảnh biên lai giả lập         |
| EasyOCR           | Model OCR hỗ trợ tiếng Việt và tiếng Anh |
| Tesseract OCR     | OCR engine truyền thống                  |
| pytesseract       | Thư viện Python để gọi Tesseract OCR     |
| pandas            | Đọc và ghi dữ liệu CSV                   |
| scikit-learn      | Chia dữ liệu train/test                  |
| pathlib           | Quản lý đường dẫn file/thư mục           |
| csv               | Ghi CSV có quote để tránh vỡ dòng        |
| time.perf_counter | Đo thời gian preprocess và OCR           |

---

## 4. Cấu trúc thư mục

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
│   ├──errorlog/
│   │  └── yyyy-mm-dd_HH-MM-SS.log
│   ├── processed/
│   ├── easyocr.csv
│   ├── easyocr_summary.csv
│   ├── tesseract.csv
│   └── tesseract_summary.csv
│
├── generateData.py
├── main.py
├── metrics.py
├── model.py
├── preprocess.py
├── utils.py
├── options.json
├── requirements.txt
├── LICENSE
├── .gitignore
└── README.md
```

---

## 5. Chức năng từng file

| File               | Chức năng                                                              |
| ------------------ | ---------------------------------------------------------------------- |
| `generateData.py`  | Sinh dataset ảnh biên lai giả lập                                      |
| `main.py`          | Chạy pipeline OCR, chia train/test, tìm option tốt nhất và lưu kết quả |
| `model.py`         | Quản lý model EasyOCR và Tesseract                                     |
| `preprocess.py`    | Tiền xử lý ảnh bằng Pillow                                             |
| `metrics.py`       | Tính Levenshtein Distance, CER và Accuracy                             |
| `utils.py`         | Đọc cấu hình, chuẩn hóa text, làm sạch text khi ghi CSV                |
| `options.json`     | Cấu hình dataset, output, preprocess và các mode cần thử               |
| `requirements.txt` | Danh sách thư viện cần cài                                             |
| `LICENSE`          | Thông tin quyền sử dụng và bản quyền                                   |
| `README.md`        | Tài liệu hướng dẫn project                                             |

---

## 6. Dataset

Dataset nằm trong thư mục:

```text
dataset/
├── images/
└── labels.csv
```

Trong đó:

* `dataset/images/` chứa ảnh biên lai/hóa đơn.
* `dataset/labels.csv` chứa text đúng tương ứng với từng ảnh.

Cấu trúc `labels.csv`:

```csv
filename,text
receipt_001.jpg,"BIEN LAI THANH TOAN | Mon: Ca phe sua | Gia: 56000 VND | Ngay: 17/06/2026 | Cam on quy khach!"
receipt_002.jpg,"BIEN LAI THANH TOAN | Mon: Tra da | Gia: 18000 VND | Ngay: 17/06/2026 | Cam on quy khach!"
```

Ý nghĩa:

| Cột        | Ý nghĩa                                       |
| ---------- | --------------------------------------------- |
| `filename` | Tên ảnh trong thư mục `dataset/images/`       |
| `text`     | Nội dung đúng dùng để so sánh với kết quả OCR |

Các ký tự như `|`, `:`, `/`, `!` sẽ được loại bỏ khi tính điểm nhờ hàm chuẩn hóa trong `utils.py`, nên kết quả đánh giá công bằng hơn.

---

## 7. Tạo dataset mẫu

Project có file `generateData.py` để tự tạo ảnh biên lai giả lập.

Chạy tạo 200 ảnh mặc định:

```powershell
python generateData.py --len 200
```

Chạy tạo 500 ảnh:

```powershell
python generateData.py --len 500
```

Đổi thư mục output:

```powershell
python generateData.py --len 200 --output dataset
```

Sau khi chạy, chương trình sẽ tạo:

```text
dataset/images/
dataset/labels.csv
```

Ảnh được tạo có thể có một số hiệu ứng ngẫu nhiên như:

* Làm mờ ảnh.
* Làm tối ảnh.
* Thêm vùng chói sáng.
* Làm méo phối cảnh.

Mục đích là mô phỏng ảnh hóa đơn/biên lai chụp trong điều kiện thực tế.

---

## 8. Cài đặt môi trường

Nên dùng Python 3.11 để tránh lỗi tương thích thư viện OCR.

### 8.1. Tạo môi trường ảo

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

Đường dẫn Python nên trỏ vào thư mục `.venv` của project.

---

### 8.2. Cài thư viện

Cài từ `requirements.txt`:

```powershell
python -m pip install -r requirements.txt
```

Nếu `requirements.txt` bị quá nhiều thư viện phụ hoặc bị lỗi encoding, có thể cài thủ công các thư viện chính:

```powershell
python -m pip install pillow numpy pandas scikit-learn easyocr pytesseract opencv-python-headless
```

Nếu muốn dùng EasyOCR với GPU NVIDIA, cài PyTorch bản CUDA:

```powershell
python -m pip uninstall torch torchvision torchaudio -y
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

Hoặc dùng CUDA 12.8:

```powershell
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Kiểm tra GPU:

```powershell
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"
```

Nếu kết quả có `True` và hiện tên GPU, EasyOCR có thể chạy bằng GPU.

---

## 9. Cài đặt Tesseract OCR

Nếu dùng model `tesseract`, cần cài thêm phần mềm **Tesseract OCR** trên Windows.

`pytesseract` chỉ là thư viện Python để gọi Tesseract, không phải OCR engine.

Sau khi cài Tesseract OCR, kiểm tra:

```powershell
tesseract --version
```

Nếu dùng tiếng Việt, cần đảm bảo Tesseract có file ngôn ngữ:

```text
vie.traineddata
```

Trong code, Tesseract được gọi với cấu hình:

```python
lang="vie+eng"
config="--psm 6"
```

Nghĩa là Tesseract sẽ đọc cả tiếng Việt và tiếng Anh.

---

## 10. Cấu hình `options.json`

File `options.json` điều khiển toàn bộ đường dẫn và cách chạy preprocess.

Ví dụ cấu hình hiện tại:

```json
{
    "dataset": {
        "dataset_dir": "dataset",
        "image_dir": "images",
        "label_path": "labels.csv",
        "test_size": 0.2,
        "random_state": 42,
        "shuffle": true
    },
    "output": {
        "output_dir": "output",
        "processed_dir": "processed",
        "errorlog_dir": "errorlog"
    },
    "preprocesser": {
        "save_processed": true,
        "auto_rotate": false,
        "resize_scale": 2,
        "contrast_factor": 2.0,
        "median_filter_size": 3,
        "background_blur_radius": 25,
        "modes": {
            "raw": {
                "enable": true,
                "threshold": 0
            },
            "gray": {
                "enable": true,
                "threshold": 0
            },
            "contrast": {
                "enable": true,
                "threshold": 0
            },
            "binary": {
                "enable": true,
                "thresholds": [100, 120, 140, 150, 160, 180, 200]
            },
            "background": {
                "enable": true,
                "thresholds": [100, 120, 140, 150, 160, 180, 200]
            }
        }
    },
    "main": {
        "multithreading": true
    }
}
```

Ý nghĩa chính:

| Key                      | Ý nghĩa                                  |
| ------------------------ | ---------------------------------------- |
| `dataset_dir`            | Thư mục dataset                          |
| `image_dir`              | Thư mục ảnh bên trong dataset            |
| `label_path`             | File nhãn bên trong dataset              |
| `test_size`              | Tỉ lệ dữ liệu test                       |
| `random_state`           | Seed để chia dữ liệu ổn định             |
| `shuffle`                | Có trộn dữ liệu trước khi chia hay không |
| `output_dir`             | Thư mục lưu kết quả                      |
| `processed_dir`          | Thư mục lưu ảnh đã xử lý                 |
| `errorlog_dir`           | Thư mục lưu log lỗi                      |
| `save_processed`         | Có lưu ảnh sau xử lý hay không           |
| `auto_rotate`            | Có tự xoay ảnh bị nghiêng hay không      |
| `resize_scale`           | Tỉ lệ phóng to ảnh                       |
| `contrast_factor`        | Mức tăng tương phản                      |
| `median_filter_size`     | Kích thước bộ lọc khử nhiễu              |
| `background_blur_radius` | Bán kính blur để ước lượng nền           |
| `modes`                  | Các mode preprocess cần thử nghiệm       |
| `multithreading`         | Sử dụng 40% số luồng CPU                 |

Lưu ý: Key đang dùng trong project là `preprocesser`, không phải `preprocessor`. Không nên tự đổi tên key nếu chưa sửa code.

---

## 11. Các mode tiền xử lý ảnh

Project hỗ trợ các mode sau:

| Mode         | Ý nghĩa                                                           |
| ------------ | ----------------------------------------------------------------- |
| `raw`        | Giữ nguyên ảnh gốc                                                |
| `gray`       | Resize và chuyển ảnh sang grayscale                               |
| `contrast`   | Resize, grayscale, tăng tương phản và làm nét                     |
| `binary`     | Xử lý thêm khử nhiễu và chuyển ảnh thành đen trắng bằng threshold |
| `background` | Giảm nền giấy, bóng đổ và nền xám bằng cách ước lượng background  |

Với `raw`, `gray`, `contrast`, threshold thường không ảnh hưởng nhiều.
Với `binary` và `background`, threshold ảnh hưởng trực tiếp đến kết quả.

---

## 12. Chạy project

### 12.1. Chạy với EasyOCR

```powershell
python main.py --model easyocr
```

Kết quả sẽ được lưu vào:

```text
output/easyocr.csv
output/easyocr_summary.csv
```

---

### 12.2. Chạy với Tesseract OCR

```powershell
python main.py --model tesseract
```

Kết quả sẽ được lưu vào:

```text
output/tesseract.csv
output/tesseract_summary.csv
```

---

## 13. Quy trình hoạt động

Khi chạy `main.py`, chương trình thực hiện các bước sau:

1. Đọc cấu hình từ `options.json`.
2. Tạo các thư mục cần thiết như `output`, `output/processed`, `output/errorlog`.
3. Đọc dữ liệu từ `dataset/labels.csv`.
4. Chia dữ liệu thành train/test theo cấu hình.
5. Khởi tạo model OCR.
6. Khởi tạo bộ tiền xử lý ảnh `Preprocesser`.
7. Tạo pipeline xử lý.
8. Dùng tập train để thử các mode preprocess.
9. Tính Accuracy và CER cho từng mode.
10. Chọn mode và threshold tốt nhất.
11. Dùng cấu hình tốt nhất để đánh giá trên tập test.
12. Lưu kết quả chi tiết và kết quả tổng kết.
13. Nếu có lỗi, ghi traceback vào thư mục log.

---

## 14. Kết quả đầu ra

Sau khi chạy, project tạo các file trong thư mục `output/`.

### 14.1. File chi tiết

Ví dụ:

```text
output/easyocr.csv
output/tesseract.csv
```

Các cột chính:

| Cột               | Ý nghĩa                          |
| ----------------- | -------------------------------- |
| `filename`        | Tên ảnh                          |
| `model`           | Model OCR đang dùng              |
| `mode`            | Mode preprocess                  |
| `threshold`       | Threshold đang dùng              |
| `expected`        | Text đúng                        |
| `predicted`       | Text OCR dự đoán                 |
| `cer`             | Character Error Rate             |
| `accuracy`        | Độ chính xác gần đúng            |
| `preprocess_time` | Thời gian xử lý ảnh              |
| `ocr_time`        | Thời gian OCR                    |
| `total_time`      | Tổng thời gian preprocess và OCR |

### 14.2. File tổng kết

Ví dụ:

```text
output/easyocr_summary.csv
output/tesseract_summary.csv
```

Các cột chính:

| Cột              | Ý nghĩa                  |
| ---------------- | ------------------------ |
| `model`          | Model OCR                |
| `best_mode`      | Mode preprocess tốt nhất |
| `best_threshold` | Threshold tốt nhất       |
| `train_accuracy` | Accuracy trên tập train  |
| `test_cer`       | CER trên tập test        |
| `test_accuracy`  | Accuracy trên tập test   |

### 14.3. Ảnh đã xử lý

Nếu `save_processed` là `true`, ảnh sau preprocess sẽ được lưu vào:

```text
output/processed/
```

Các ảnh này dùng để kiểm tra xem bước xử lý ảnh có làm chữ rõ hơn hay không.

---

## 15. Metrics đánh giá

Project dùng **CER** và **Accuracy**.

### 15.1. CER

CER là viết tắt của **Character Error Rate**.

Công thức:

```text
CER = Levenshtein Distance / số ký tự của text đúng
```

CER càng thấp thì model OCR càng tốt.

Ví dụ:

```text
CER = 0.10
```

Nghĩa là sai khoảng 10% ký tự.

### 15.2. Accuracy

Accuracy được tính gần đúng từ CER:

```text
Accuracy = max(0, 1 - CER)
```

Accuracy càng cao thì model OCR càng tốt.

---

## 16. Chuẩn hóa text khi đánh giá

Trước khi tính điểm, text được chuẩn hóa trong `utils.py`.

Các bước chuẩn hóa:

* Chuyển về string.
* Xóa dấu tiếng Việt.
* Chuyển về chữ thường.
* Xóa ký tự đặc biệt.
* Chỉ giữ chữ cái, số và khoảng trắng.
* Xóa khoảng trắng dư.

Ví dụ:

```text
BIEN LAI THANH TOAN | Mon: Trà đá | Giá: 18000 VND!
```

sau chuẩn hóa thành:

```text
bien lai thanh toan mon tra da gia 18000 vnd
```

Nhờ vậy, các ký tự như `|`, `:`, `!`, `/` không làm điểm đánh giá bị sai lệch quá nhiều.

---

## 17. Đo thời gian xử lý

Trong `main.py`, hàm `evaluate()` đo ba loại thời gian:

| Cột               | Ý nghĩa                         |
| ----------------- | ------------------------------- |
| `preprocess_time` | Thời gian xử lý ảnh bằng Pillow |
| `ocr_time`        | Thời gian model OCR đọc ảnh     |
| `total_time`      | Tổng thời gian của từng ảnh     |

Ngoài ra, khi chạy terminal sẽ in tổng thời gian theo từng mode:

```text
Evaluate mode=contrast, threshold=0 | preprocess=1.24s | ocr=12.50s | total=13.90s
```

Thông tin này giúp xác định phần chậm nằm ở preprocess hay OCR.

---

## 18. Sử dụng GPU

EasyOCR có thể dùng GPU nếu PyTorch nhận CUDA.

Trong `model.py`, EasyOCR được khởi tạo với:

```python
Reader(["vi", "en"], gpu=True)
```

Kiểm tra GPU:

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"
```

Nếu Task Manager không hiện mục CUDA riêng thì vẫn có thể kiểm tra bằng:

```powershell
nvidia-smi
```

Nếu thấy `python.exe` dùng VRAM, nghĩa là GPU đang được dùng.

Lưu ý: GPU chỉ tăng tốc phần OCR của EasyOCR. Các bước như resize, rotate, background removal, ghi CSV vẫn chạy bằng CPU.

---

## 19. Tối ưu tốc độ

Nếu project chạy chậm, có thể chỉnh `options.json`.

Tắt lưu ảnh processed:

```json
"save_processed": false
```

Tắt tự xoay ảnh:

```json
"auto_rotate": false
```

Giảm số threshold cần thử:

```json
"binary": {
    "enable": true,
    "thresholds": [120, 150, 180]
}
```

Tắt mode background nếu không cần:

```json
"background": {
    "enable": false,
    "thresholds": [120, 150, 180]
}
```

Gợi ý cấu hình nhanh để benchmark:

```json
"modes": {
    "raw": {
        "enable": true,
        "threshold": 0
    },
    "gray": {
        "enable": true,
        "threshold": 0
    },
    "contrast": {
        "enable": true,
        "threshold": 0
    },
    "binary": {
        "enable": true,
        "thresholds": [120, 150, 180]
    },
    "background": {
        "enable": false,
        "thresholds": [120, 150, 180]
    }
}
```

---

## 20. Lỗi thường gặp

### 20.1. EasyOCR chạy CPU thay vì GPU

Kiểm tra PyTorch:

```powershell
python -c "import torch; print(torch.version.cuda); print(torch.cuda.is_available())"
```

Nếu `False`, cần cài lại PyTorch bản CUDA.

---

### 20.2. TesseractNotFoundError

Nguyên nhân: Chưa cài phần mềm Tesseract OCR hoặc chưa thêm Tesseract vào PATH.

Cách kiểm tra:

```powershell
tesseract --version
```

---

### 20.3. File CSV bị vỡ dòng

Nguyên nhân: Text OCR có ký tự xuống dòng.

Project đã xử lý bằng `clean_text_for_csv()` và lưu CSV với `QUOTE_ALL`, nên file CSV sẽ dễ đọc hơn.

---

### 20.4. Không tìm thấy `labels.csv`

Kiểm tra cấu trúc thư mục:

```text
dataset/
├── images/
└── labels.csv
```

Nếu chưa có dataset, chạy:

```powershell
python generateData.py --len 200
```

---

### 20.5. Output quá nặng

Nếu `output/processed/` có quá nhiều ảnh, chỉnh trong `options.json`:

```json
"save_processed": false
```

---

## 21. Tác giả

* Gia Khang

Project được phát triển phục vụ học tập, nghiên cứu và báo cáo môn học.

---

## 22. Quyền sử dụng và bản quyền

Project này được phát triển cho mục đích học tập và báo cáo môn học.

Toàn bộ source code, cấu trúc xử lý ảnh và quy trình đánh giá model thuộc về nhóm tác giả của project.

Người dùng có thể tham khảo, chỉnh sửa và chạy lại project cho mục đích học tập hoặc nghiên cứu cá nhân.

Không được sử dụng project này cho mục đích thương mại, sao chép để nộp lại như sản phẩm của người khác hoặc phân phối lại khi chưa có sự đồng ý của nhóm tác giả.

Dataset mẫu trong project chỉ dùng để thử nghiệm mô hình OCR.

Nếu sử dụng ảnh hóa đơn/biên lai thật, cần đảm bảo không chứa thông tin cá nhân nhạy cảm hoặc dữ liệu riêng tư của người khác.

© 2026 Gia Khang. All rights reserved.
