from re import sub
from unicodedata import category, normalize
from json import load, dump
from typing import TypedDict


class DatasetDict(TypedDict):
    dataset_dir: str
    image_dir: str
    label_path: str
    test_size: float
    random_state: int
    shuffle: bool


class OutputDict(TypedDict):
    output_dir: str
    processed_dir: str
    errorlog_dir: str


class ThresholdDict(TypedDict):
    enable: bool
    threshold: int


class ThresholdsDict(TypedDict):
    enable: bool
    thresholds: list[int]


class ModesDict(TypedDict):
    raw: ThresholdDict
    gray: ThresholdDict
    contrast: ThresholdDict
    binary: ThresholdsDict
    background: ThresholdsDict


class PreprocesserDict(TypedDict):
    save_processed: bool
    auto_rotate: bool
    resize_scale: int
    contrast_factor: float
    median_filter_size: int
    background_blur_radius: int
    modes: ModesDict


class OptionsDict(TypedDict):
    dataset: DatasetDict
    output: OutputDict
    preprocesser: PreprocesserDict


# Cấu hình mặc định dùng để tạo options.json khi file chưa tồn tại.
default_options: OptionsDict = {
    "dataset": {
        "dataset_dir": "dataset",
        "image_dir": "dataset/images",
        "label_path": "dataset/labels.csv",
        "test_size": 0.2,
        "random_state": 42,
        "shuffle": True,
    },
    "output": {
        "output_dir": "output",
        "processed_dir": "output/processed",
        "errorlog_dir": "errorlog",
    },
    "preprocesser": {
        "save_processed": True,
        "auto_rotate": True,
        "resize_scale": 2,
        "contrast_factor": 2.0,
        "median_filter_size": 3,
        "background_blur_radius": 25,
        "modes": {
            "raw": {"enable": True, "threshold": 0},
            "gray": {"enable": True, "threshold": 0},
            "contrast": {"enable": True, "threshold": 0},
            "binary": {"enable": True, "thresholds": [150]},
            "background": {"enable": True, "thresholds": [150]},
        },
    },
}


# Loại bỏ dấu tiếng Việt, bao gồm cả trường hợp đặc biệt của chữ đ và Đ.
def remove_vietnamese_accents(text: str) -> str:
    text = normalize("NFD", text)
    text = "".join(char for char in text if category(char) != "Mn")
    text = text.replace("đ", "d").replace("Đ", "D")

    return text


# Chuẩn hóa văn bản OCR để so sánh công bằng giữa nhãn thật và kết quả dự đoán.
def normalize_text(text: str) -> str:
    text = str(text)
    text = remove_vietnamese_accents(text)
    text = text.lower()
    text = sub(r"[^a-z0-9\s]", " ", text)
    text = " ".join(text.split())

    return text


# Làm sạch văn bản trước khi ghi CSV để tránh lỗi xuống dòng hoặc lệch cột.
def clean_text_for_csv(text: str) -> str:
    text = str(text)
    text = text.replace("\r", " ")
    text = text.replace("\n", " ")
    text = text.replace("\t", " ")
    text = " ".join(text.split())

    return text


# Đọc options.json, nếu chưa có thì tự tạo file bằng cấu hình mặc định.
def read_options() -> OptionsDict:
    try:
        with open("options.json", "r", encoding="utf-8") as content:
            options: OptionsDict = load(content)

        return options

    except Exception:
        print('"options.json" not found, automatically generate the "options.json"...')

        with open("options.json", "w", encoding="utf-8") as content:
            dump(default_options, content, ensure_ascii=False, indent=4)

        return default_options


# Chuyển chuỗi về dạng an toàn để dùng trong tên file hoặc đường dẫn.
def safe_text(text: str) -> str:
    text = text.strip()
    text = sub(r"[^\w\-_.]+", "_", text)
    text = sub(r"_+", "_", text)

    return text.strip("_")


# Chuyển số thực thành chuỗi an toàn để đưa vào tên file.
def safe_float(text: float) -> str:
    return str(text).replace(".", "p")
