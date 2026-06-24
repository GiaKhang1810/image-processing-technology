"""
FastAPI backend for Vietnamese Receipt OCR
Run: uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pathlib import Path
from io import BytesIO
from time import perf_counter
import tempfile
import os

from PIL import Image
from model import Model
from preprocess import Preprocesser

# ── App setup ──────────────────────────────────────────────────────────────────
app = FastAPI(title="Vietnamese Receipt OCR API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static frontend files (index.html, etc.)
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)

# ── Load model & preprocesser once at startup ──────────────────────────────────
print("Loading OCR model...")

MODEL_NAME = os.getenv("OCR_MODEL", "easyocr")  # or "tesseract"

model = Model(MODEL_NAME)

preprocesser = Preprocesser(
    processed_dir=None,   # Don't save processed images in API mode
    resize_scale=2,
    median_filter_size=3,
    contrast_factor=2.0,
    background_blur_radius=25,
    auto_rotate=True,
    auto_sharpen=True,
    save_processed=False,
)

# Best preprocessing config (from your training results).
# Override via env vars: BEST_MODE, BEST_THRESHOLD
BEST_MODE      = os.getenv("BEST_MODE", "contrast")
BEST_THRESHOLD = int(os.getenv("BEST_THRESHOLD", "150"))

print(f"Model: {MODEL_NAME} | Mode: {BEST_MODE} | Threshold: {BEST_THRESHOLD}")

# ── Response schema ─────────────────────────────────────────────────────────────
class OCRResult(BaseModel):
    text: str
    model: str
    mode: str
    threshold: int
    preprocess_ms: float
    ocr_ms: float
    total_ms: float

# ── Routes ──────────────────────────────────────────────────────────────────────
@app.get("/")
def serve_frontend():
    """Serve the frontend HTML."""
    frontend = STATIC_DIR / "index.html"
    if frontend.exists():
        return FileResponse(frontend)
    return {"message": "OCR API is running. Place index.html in ./static/"}


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME}


@app.post("/ocr", response_model=OCRResult)
async def run_ocr(file: UploadFile = File(...)):
    """
    Upload an image (jpg/png/webp) and get OCR text back.
    """
    # Validate file type
    allowed = {"image/jpeg", "image/png", "image/webp", "image/bmp", "image/tiff"}
    if file.content_type not in allowed:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {file.content_type}. Use JPG, PNG, or WebP.",
        )

    contents = await file.read()

    if len(contents) > 20 * 1024 * 1024:  # 20 MB cap
        raise HTTPException(status_code=413, detail="File too large (max 20 MB).")

    # Save to temp file — Preprocesser expects a path
    suffix = Path(file.filename or "upload.jpg").suffix or ".jpg"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        # ── Preprocess ──
        t0 = perf_counter()
        processed_image = preprocesser.modify(
            image_path=tmp_path,
            mode=BEST_MODE,
            threshold=BEST_THRESHOLD,
            name=None,
        )
        preprocess_ms = (perf_counter() - t0) * 1000

        # ── OCR ──
        t1 = perf_counter()
        text = model.read(processed_image)
        ocr_ms = (perf_counter() - t1) * 1000

    finally:
        os.unlink(tmp_path)

    return OCRResult(
        text=text.strip(),
        model=MODEL_NAME,
        mode=BEST_MODE,
        threshold=BEST_THRESHOLD,
        preprocess_ms=round(preprocess_ms, 1),
        ocr_ms=round(ocr_ms, 1),
        total_ms=round(preprocess_ms + ocr_ms, 1),
    )


# Mount static dir last so API routes take priority
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
