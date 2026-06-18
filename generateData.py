from argparse import ArgumentParser
from csv import writer
from pathlib import Path
from random import choice, randint, sample, uniform

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ITEMS = [
    "Ca phe sua",
    "Tra da",
    "Banh mi",
    "Pho bo",
    "Com tam",
    "Sinh to bo",
    "Sinh sau rieng",
    "Ca phe da",
    "Pho ga",
    "Banh bao",
    "Xuc xich",
    "Sinh to dau",
]


def generate_base_receipt(text: str) -> np.ndarray:
    img = Image.new("RGB", (400, 300), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    font = ImageFont.load_default()

    draw.text((20, 20), "BIEN LAI THANH TOAN", fill=(0, 0, 0), font=font)
    draw.text((20, 60), text, fill=(0, 0, 0), font=font)
    draw.text((20, 250), "Cam on quy khach!", fill=(0, 0, 0), font=font)

    return np.array(img)


def apply_blur(img: np.ndarray) -> np.ndarray:
    k = choice([3, 5, 7])

    return cv2.GaussianBlur(img, (k, k), 0)


def apply_low_light(img: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hsv = np.array(hsv, dtype=np.float64)

    hsv[:, :, 2] = hsv[:, :, 2] * uniform(0.3, 0.6)
    hsv[:, :, 2][hsv[:, :, 2] > 255] = 255

    hsv = np.array(hsv, dtype=np.uint8)

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def apply_glare(img: np.ndarray) -> np.ndarray:
    overlay = img.copy()
    output = img.copy()

    h, w = img.shape[:2]

    cx = randint(0, w)
    cy = randint(0, h)
    radius = randint(50, 150)

    cv2.circle(overlay, (cx, cy), radius, (255, 255, 255), -1)
    cv2.addWeighted(overlay, 0.5, output, 0.5, 0, output)

    return output


def apply_perspective_distortion(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]

    pts1 = np.float32(
        [
            [0, 0],
            [w, 0],
            [0, h],
            [w, h],
        ]
    )

    pad = randint(10, 40)

    pts2 = np.float32(
        [
            [randint(0, pad), randint(0, pad)],
            [w - randint(0, pad), randint(0, pad)],
            [randint(0, pad), h - randint(0, pad)],
            [w - randint(0, pad), h - randint(0, pad)],
        ]
    )

    matrix = cv2.getPerspectiveTransform(pts1, pts2)

    return cv2.warpPerspective(
        img,
        matrix,
        (w, h),
        borderValue=(200, 200, 200),
    )


def apply_random_effects(img: np.ndarray) -> np.ndarray:
    effects = sample(
        ["blur", "dark", "glare", "distort"],
        k=randint(1, 3),
    )

    if "blur" in effects:
        img = apply_blur(img)

    if "dark" in effects:
        img = apply_low_light(img)

    if "glare" in effects:
        img = apply_glare(img)

    if "distort" in effects:
        img = apply_perspective_distortion(img)

    return img


def generate_receipt_data(index: int) -> tuple[str, str, np.ndarray]:
    item_name = choice(ITEMS)
    price = randint(15, 100) * 1000
    day = randint(1, 28)
    month = 10
    year = 2026

    receipt_text = f"Mon: {item_name}\nGia: {price} VND\nNgay: {day}/{month}/{year}"

    full_text = (
        f"BIEN LAI THANH TOAN | "
        f"Mon: {item_name} | "
        f"Gia: {price} VND | "
        f"Ngay: {day}/{month}/{year} | "
        f"Cam on quy khach!"
    )

    filename = f"receipt_{index:03d}.jpg"

    img = generate_base_receipt(receipt_text)
    img = apply_random_effects(img)

    return filename, full_text, img


def generate_dataset(length: int, output_dir: str = "dataset") -> None:
    if length <= 0:
        raise ValueError("--len must be greater than 0")

    output_path = Path(output_dir)
    image_dir = output_path / "images"
    label_path = output_path / "labels.csv"

    image_dir.mkdir(parents=True, exist_ok=True)

    csv_data = [["filename", "text"]]

    for i in range(1, length + 1):
        filename, text, img = generate_receipt_data(i)

        filepath = image_dir / filename

        cv2.imwrite(
            str(filepath),
            cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
        )

        csv_data.append([filename, text])

    with label_path.open("w", newline="", encoding="utf-8") as file:
        csv_writer = writer(file)
        csv_writer.writerows(csv_data)

    print(f"{length} images have been successfully created.")
    print(f"Images: {image_dir}")
    print(f"Labels: {label_path}")


def main() -> None:
    parser = ArgumentParser()

    parser.add_argument(
        "--len",
        dest="length",
        type=int,
        default=200,
        help="Number of receipt images to generate",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="dataset",
        help="Output dataset directory",
    )

    args = parser.parse_args()

    generate_dataset(
        length=args.length,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
