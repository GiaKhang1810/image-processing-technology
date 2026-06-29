from os import cpu_count
from pathlib import Path
from json import load, dump
from re import sub
from unicodedata import category, normalize

from .types.options import Options

OPTIONS_P = Path("options.json")

DEFAULT_OPTIONS: Options = {
    "preprocessor": {
        "storage": {
            "dataset": {"tune": "dataset/tune", "evaluate": "dataset/evaluate"},
            "output": {"tune": "output/tune", "evaluate": "output/evaluate"},
            "processed": False,
        },
        "rotation": {
            "enable": True,
            "step": 0.25,
            "minConfidence": 0.06,
            "maxangle": 8.0,
            "minangle": 0.35,
        },
        "sharpness": {
            "enable": True,
            "method": {"unsharp": True, "kernel": False},
            "sigma": 1.0,
            "amount": 1.5,
            "matrix": [[-1.0, -1.0, -1.0], [-1.0, 9.0, -1.0], [-1.0, -1.0, -1.0]],
        },
        "enhancer": {
            "enable": True,
            "resizeScale": 2,
            "contrastfactor": 2.0,
            "medianfilterSize": 3,
        },
        "remover": {
            "enable": True,
            "method": {
                "division": True,
                "gaussian": False,
                "median": False,
                "morphology": False,
            },
            "blurRadius": 25,
            "kernelsize": 3,
            "iterations": 1,
        },
        "binarization": {
            "enable": True,
            "method": {"fixed": True, "adaptive": False, "otsu": False},
            "threshold": 127,
            "blockSize": 11,
            "constant": 2.0,
        },
    },
    "main": {
        "errorlog": "output/errorlog",
        "bestoptions": "output/tune",
        "model": {"easyocr": True, "tesseract": False},
        "multithread": {"enable": True, "usethreads": 6},
        "run": {"tune": True, "evaluate": True},
    },
}


def get_options() -> Options:
    try:
        with open(file=OPTIONS_P, mode="r", encoding="utf8") as reader:
            content = load(reader)

            return content
    except Exception:
        with open(file=OPTIONS_P, mode="w", encoding="utf8") as writer:
            dump(obj=DEFAULT_OPTIONS, fp=writer)

            return DEFAULT_OPTIONS


def get_thread_count(use_threads: int) -> int:
    total_threads = cpu_count() or 1
    max_use_threads = max(1, int(total_threads * 0.4))
    requested_threads = max(1, use_threads)

    return min(requested_threads, max_use_threads)


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


def clamp(value: int | float, min_value: int | float, max_value: int | float):
    return max(min_value, min(value, max_value))


def ensure_odd(value: int, min_value: int = 1, max_value: int = 9) -> int:
    value = int(clamp(value, min_value, max_value))

    if value % 2 == 0:
        value += 1

    return int(clamp(value, min_value, max_value))


def count_True(obj: dict[str, bool]) -> int:
    return sum(1 for value in obj.values() if value is True)


def validate_one_True(obj: dict[str, bool]) -> None:
    total_True = count_True(obj)

    if total_True == 0:
        raise ValueError("At least one method must be enabled")

    if total_True > 1:
        raise ValueError("Only one method can be enabled")


def get_name_enable(obj: dict[str, bool]) -> str:
    enabled_methods = [name for name, enabled in obj.items() if enabled is True]

    return enabled_methods[0]
