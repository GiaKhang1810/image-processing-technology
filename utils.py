from re import sub
from unicodedata import category, normalize
from json import load, dumps
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

default_options: OptionsDict = {
    "dataset": {
        "dataset_dir": "dataset",
        "image_dir": "dataset/images",
        "label_path": "dataset/labels.csv",
        "test_size": 0.2
    },
    "output": {
        "output_dir": "output",
        "processed_dir": "output/processed",
        "errorlog_dir": "errorlog"
    },
    "preprocesser": {
        "save_processed": True,
        "auto_rotate": True,
        "resize_scale": 2,
        "contrast_factor": 2.0,
        "median_filter_size": 3,
        "background_blur_radius": 25,
        "modes": {
            "raw": True,
            "gray": True,
            "contrast": True,
            "binary": {
                "enable": True,
                "threshold": [150]
            },
            "background": {
                "enable": True,
                "threshold": [150]
            }
        }
    } 
}

def remove_vietnamese_accents(text: str) -> str:
    text = normalize("NFD", text)
    text = "".join(char for char in text if category(char) != "Mn")
    text = text.replace("đ", "d").replace("Đ", "D")

    return text


def normalize_text(text: str) -> str:
    text = str(text)
    text = remove_vietnamese_accents(text)
    text = text.lower()
    text = sub(r"[^a-z0-9\s]", " ", text)
    text = " ".join(text.split())

    return text


def clean_text_for_csv(text: str) -> str:
    text = str(text)
    text = text.replace("\r", " ")
    text = text.replace("\n", " ")
    text = text.replace("\t", " ")
    text = " ".join(text.split())

    return text


def read_options() -> OptionsDict:
    try:
        with open("options.json", "r", encoding="utf-8") as content:
            options: OptionsDict = load(content)

        return options
    except Exception:
        print("\"options.json\" not found, automatically generate the \"options.json\"...")
        
        with open("options.json", "w", encoding="utf-8") as content:
            dumps(default_options, content, ensure_ascii=False, indent=4)

        return default_options