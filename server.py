"""
BillScan — server.py
FastAPI server nhận ảnh, chạy preprocess + OCR, trả về JSON.
Dùng chung Preprocesser và Model từ code sẵn có.

Deploy: Railway / Render / VPS
    railway up
    hoặc: uvicorn server:app --host 0.0.0.0 --port $PORT
"""

import os
import io
import re
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

from model import Model
from preprocess import Preprocesser
from utils import read_options

# ── Khởi tạo model và preprocesser một lần duy nhất khi server start ──
ocr_model: Model | None = None
preprocesser: Preprocesser | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global ocr_model, preprocesser

    options = read_options()

    model_name = os.environ.get("OCR_MODEL", "easyocr")   # easyocr | tesseract
    print(f"[BillScan] Loading model: {model_name}")

    ocr_model = Model(model_name)

    preprocesser = Preprocesser(
        processed_dir=None,                              # Không lưu ảnh trên server
        resize_scale=options["preprocesser"]["resize_scale"],
        median_filter_size=options["preprocesser"]["median_filter_size"],
        contrast_factor=options["preprocesser"]["contrast_factor"],
        background_blur_radius=options["preprocesser"]["background_blur_radius"],
        auto_rotate=options["preprocesser"]["auto_rotate"],
        auto_sharpen=options["preprocesser"]["auto_sharpen"],
        save_processed=False,
    )

    # Mode và threshold tốt nhất — đọc từ biến môi trường hoặc dùng mặc định
    app.state.best_mode      = os.environ.get("OCR_MODE", "contrast")
    app.state.best_threshold = int(os.environ.get("OCR_THRESHOLD", "0"))

    print(f"[BillScan] Mode={app.state.best_mode}, Threshold={app.state.best_threshold}")
    print("[BillScan] Server ready.")
    yield


app = FastAPI(title="BillScan OCR API", version="1.0.0", lifespan=lifespan)

# ── CORS: cho phép frontend gọi từ bất kỳ origin ──────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif", "image/bmp", "image/tiff"}
MAX_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB


@app.get("/health")
def health():
    """Endpoint kiểm tra server đang sống — frontend ping định kỳ."""
    return {"status": "ok", "model": os.environ.get("OCR_MODEL", "easyocr")}


@app.post("/ocr")
async def ocr(file: UploadFile = File(...)):
    """
    Nhận ảnh hóa đơn → preprocess → OCR → trả JSON.

    Response:
        {
            "text": "toàn bộ text OCR đọc được",
            "items": [
                {"mon": "...", "gia": "...", "ngay": "..."}
            ],
            "tong": "...",
            "ngay_hoadon": "..."
        }
    """
    # Kiểm tra loại file
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Định dạng không hỗ trợ: {file.content_type}. Dùng JPG, PNG hoặc WEBP."
        )

    raw_bytes = await file.read()

    if len(raw_bytes) > MAX_SIZE_BYTES:
        raise HTTPException(status_code=413, detail="File quá lớn. Tối đa 10MB.")

    try:
        # Preprocesser.modify() nhận đường dẫn file — ta ghi tạm vào /tmp
        tmp_path = Path("/tmp") / (file.filename or "upload.jpg")
        tmp_path.write_bytes(raw_bytes)

        image = preprocesser.modify(
            image_path=str(tmp_path),
            mode=app.state.best_mode,
            threshold=app.state.best_threshold,
            name=None,
        )

        raw_text: str = ocr_model.read(image)

        # Cố gắng parse cấu trúc đơn giản từ text
        items, tong, ngay_hoadon = parse_receipt_text(raw_text)

        return JSONResponse({
            "text":        raw_text,
            "items":       items,
            "tong":        tong,
            "ngay_hoadon": ngay_hoadon,
        })

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Lỗi OCR: {str(exc)}")
    finally:
        # Xóa file tạm
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass


import re

def parse_receipt_text(text: str):
    items = []
    tong = ""
    ngay_hoadon = ""

    # ── Định dạng dataset ──────────────────────────────────────────────
    mon_match  = re.search(r"Mon[:\s]+([^\|]+)", text, re.IGNORECASE)
    gia_match  = re.search(r"Gia[:\s]+([^\|]+)", text, re.IGNORECASE)
    ngay_match = re.search(r"Ngay[:\s]+([^\|]+)", text, re.IGNORECASE)
    if mon_match:
        items.append({
            "mon":      mon_match.group(1).strip(),
            "gia":      gia_match.group(1).strip()  if gia_match  else "",
            "ngay":     ngay_match.group(1).strip() if ngay_match else "",
            "so_luong": 1,
        })
        if ngay_match:
            ngay_hoadon = ngay_match.group(1).strip()
        return items, tong, ngay_hoadon

    # ── Helpers ────────────────────────────────────────────────────────
    def clean_num(s):
        s = re.sub(r"[oO]", "0", s)
        s = re.sub(r"[^\d.,]", "", s)
        if "." in s and "," in s:
            s = s.replace(",", "")
        s = s.replace(",", ".")
        parts = s.split(".")
        if len(parts) > 2:
            s = "".join(parts[:-1]) + "." + parts[-1]
        return s.rstrip(".")

    def to_int_vnd(s):
        try:
            f = float(clean_num(s))
            return int(f * 1000) if 0 < f < 1000 else int(f)
        except Exception:
            return 0

    def fmt_vnd(n):
        return f"{n:,} VND".replace(",", ".")

    # ── Ngày & tổng ───────────────────────────────────────────────────
    date_pat  = re.compile(r"\b(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})\b")
    total_pat = re.compile(
        r"(?:t[oổ]ng|c[oộ]ng|total|thanh\s*to[aá]n)[^\d]*(\d[\d.,]+)",
        re.IGNORECASE,
    )
    dm = date_pat.search(text)
    if dm:
        ngay_hoadon = dm.group(1)
    tm = total_pat.search(text)
    if tm:
        v = to_int_vnd(tm.group(1))
        if v > 0:
            tong = fmt_vnd(v)

    skip_kw = {"tên", "ten", "hang", "tong", "cong", "sl", "đg", "tt",
               "stt", "món", "mon", "uống", "uong", "ăn", "an"}

    # ── Tách theo STT tăng dần ────────────────────────────────────────
    stt_pat = re.compile(r"(?<!\d)(\d{1,2})\s+([A-ZĐa-zđÀ-ỹ])")
    stt_positions = [(m.start(), int(m.group(1))) for m in stt_pat.finditer(text)]

    filtered = []
    expected = 1
    for pos, stt in stt_positions:
        if stt == expected:
            filtered.append((pos, stt))
            expected += 1
        elif stt > expected and stt <= expected + 2:
            filtered.append((pos, stt))
            expected = stt + 1

    money_pat = re.compile(r"\d[\d.,]{2,}")

    if len(filtered) >= 2:
        for i, (pos, stt) in enumerate(filtered):
            end = filtered[i + 1][0] if i + 1 < len(filtered) else len(text)
            seg = text[pos:end]
            seg = re.sub(r"^\d{1,2}\s+", "", seg).strip()

            numbers = money_pat.findall(seg)
            if not numbers:
                continue

            # Số tiền: >= 3 chữ số
            money_nums = [n for n in numbers if len(re.sub(r"[^\d]", "", n)) >= 3]

            if len(money_nums) >= 2:
                don_gia_str    = money_nums[-2]
                thanh_tien_str = money_nums[-1]
            elif len(money_nums) == 1:
                don_gia_str    = money_nums[0]
                thanh_tien_str = money_nums[0]
            else:
                continue

            # Tên món: text trước số đầu tiên
            fm = money_pat.search(seg)
            ten_mon = seg[:fm.start()].strip() if fm else seg
            ten_mon = re.sub(r"[^\w\sÀ-ỹĐđ]", " ", ten_mon)
            ten_mon = " ".join(ten_mon.split())
            if not ten_mon or len(ten_mon) < 2:
                continue
            if ten_mon.lower() in skip_kw:
                continue

            # SL: số nhỏ không nằm trong money_nums
            sl_val = 1
            small_nums = [n for n in numbers if n not in money_nums]
            if small_nums:
                try:
                    sl_val = max(1, round(float(clean_num(small_nums[0]))))
                except Exception:
                    sl_val = 1

            don_val = to_int_vnd(don_gia_str)
            tt_val  = to_int_vnd(thanh_tien_str)
            if tt_val > 0 and don_val > 0 and tt_val < don_val / 2:
                don_val, tt_val = tt_val, don_val

            items.append({
                "mon":      ten_mon,
                "gia":      fmt_vnd(don_val) if don_val else "",
                "ngay":     "",
                "so_luong": sl_val,
            })

        if items:
            return items, tong, ngay_hoadon

    # ── Fallback regex ─────────────────────────────────────────────────
    num_re  = r"\d[\d.,]*"
    row_pat = re.compile(
        r"(?<!\d)(\d{1,2})\s+"
        r"([A-ZĐa-zđÀ-ỹ][^0-9\n]{1,50}?)\s+"
        r"(?:(" + num_re + r")\s+)?"
        r"(" + num_re + r")\s+"
        r"(" + num_re + r")"
    )
    for m in row_pat.finditer(text):
        ten_mon = m.group(2).strip()
        if any(kw in ten_mon.lower() for kw in skip_kw) or len(ten_mon) < 2:
            continue
        sl_val = 1
        if m.group(3):
            try:
                sl_val = max(1, round(float(clean_num(m.group(3)))))
            except Exception:
                sl_val = 1
        don_val = to_int_vnd(m.group(4))
        tt_val  = to_int_vnd(m.group(5))
        if tt_val > 0 and don_val > 0 and tt_val < don_val / 2:
            don_val, tt_val = tt_val, don_val
        items.append({
            "mon":      ten_mon,
            "gia":      fmt_vnd(don_val) if don_val else "",
            "ngay":     "",
            "so_luong": sl_val,
        })

    return items, tong, ngay_hoadon



if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)
