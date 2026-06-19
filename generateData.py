from argparse import ArgumentParser
from csv import writer
from pathlib import Path
from random import choice, randint, sample, uniform

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ITEMS = [
    "pho bo",
    "pho ga",
    "pho tai",
    "pho nam",
    "pho gan",
    "pho gau",
    "pho sot vang",
    "pho xao",
    "pho cuon",
    "pho ap chao",
    "bun bo hue",
    "bun rieu",
    "bun cha",
    "bun thit nuong",
    "bun dau mam tom",
    "bun mam",
    "bun mang vit",
    "bun oc",
    "bun ca",
    "bun ca ro",
    "bun ca cham",
    "bun suon",
    "bun moc",
    "bun gio heo",
    "bun bo nam bo",
    "bun tom",
    "bun nghe",
    "bun quay",
    "bun ken",
    "bun nuoc leo",
    "bun cha ca",
    "bun ga nuong",
    "bun nem nuong",
    "bun xao",
    "bun tron",
    "bun cha gio",
    "bun bi",
    "bun mam nem",
    "bun long",
    "bun ngan",
    "mi quang",
    "mi vit tiem",
    "mi ga tan",
    "mi xao bo",
    "mi xao hai san",
    "mi kho",
    "mi hoanh thanh",
    "mi bo kho",
    "mi cay",
    "mi ga quay",
    "hu tieu",
    "hu tieu nam vang",
    "hu tieu my tho",
    "hu tieu kho",
    "hu tieu xao",
    "hu tieu bo vien",
    "hu tieu hai san",
    "hu tieu ga",
    "hu tieu muc",
    "hu tieu sa te",
    "banh mi",
    "banh mi thit",
    "banh mi cha lua",
    "banh mi op la",
    "banh mi bo kho",
    "banh mi xiu mai",
    "banh mi ga",
    "banh mi pate",
    "banh mi heo quay",
    "banh mi cha ca",
    "banh mi trung",
    "banh mi que",
    "banh mi nem nuong",
    "banh xeo",
    "banh khot",
    "banh cuon",
    "banh uot",
    "banh beo",
    "banh nam",
    "banh bot loc",
    "banh canh",
    "banh canh cua",
    "banh canh ca loc",
    "banh canh gio heo",
    "banh canh tom",
    "banh canh cha ca",
    "banh gio",
    "banh duc",
    "banh duc nong",
    "banh duc man",
    "banh duc ngot",
    "banh da cua",
    "banh da tron",
    "banh da lon",
    "banh it",
    "banh it tran",
    "banh it la gai",
    "banh tet",
    "banh chung",
    "banh day",
    "banh phu the",
    "banh cam",
    "banh ran",
    "banh tieu",
    "banh bo",
    "banh bo nuong",
    "banh bo thot not",
    "banh chuoi",
    "banh chuoi hap",
    "banh chuoi nuong",
    "banh khoai",
    "banh khoai mi",
    "banh khoai mo",
    "banh khoai tay",
    "banh trang tron",
    "banh trang nuong",
    "banh trang cuon",
    "banh trang cham",
    "banh trang me",
    "banh trang sa te",
    "banh trang phoi suong",
    "banh trang cuon thit heo",
    "banh hoi",
    "banh hoi heo quay",
    "banh hoi long heo",
    "banh can",
    "banh can trung",
    "banh can hai san",
    "banh canh nam pho",
    "banh ep",
    "banh loc",
    "banh pia",
    "banh in",
    "banh gai",
    "banh tro",
    "banh te",
    "banh gio cha",
    "banh mi hap",
    "com tam",
    "com tam suon",
    "com tam bi cha",
    "com tam suon bi cha",
    "com tam suon trung",
    "com ga",
    "com ga xoi mo",
    "com ga hoi an",
    "com ga tam ky",
    "com chien",
    "com chien duong chau",
    "com chien hai san",
    "com chien ca man",
    "com chien bo luc lac",
    "com chien trung",
    "com suon",
    "com suon ram",
    "com suon nuong",
    "com thit kho",
    "com ca kho",
    "com ca chien",
    "com ga kho",
    "com bo xao",
    "com bo luc lac",
    "com rang",
    "com rang dua bo",
    "com rang thap cam",
    "com nieu",
    "com chay",
    "com lam",
    "com hen",
    "com am phu",
    "com do",
    "com trang",
    "com tron",
    "com chien toi",
    "com chien ca moi",
    "com hap la sen",
    "xoi man",
    "xoi ngot",
    "xoi ga",
    "xoi bap",
    "xoi dau xanh",
    "xoi dau den",
    "xoi dau phong",
    "xoi lac",
    "xoi xeo",
    "xoi khuc",
    "xoi gac",
    "xoi vo",
    "xoi mit",
    "xoi sau rieng",
    "xoi chien",
    "xoi pate",
    "xoi thit kho",
    "xoi cha lua",
    "xoi trung",
    "xoi lap xuong",
    "xoi thap cam",
    "chao ga",
    "chao vit",
    "chao long",
    "chao suon",
    "chao ca",
    "chao trai",
    "chao muc",
    "chao hau",
    "chao thit bam",
    "chao dau xanh",
    "chao trang",
    "chao bo",
    "chao ech",
    "chao tom",
    "chao ga ac",
    "goi cuon",
    "goi ga",
    "goi vit",
    "goi bo",
    "goi ngo sen",
    "goi du du",
    "goi xoai",
    "goi sua",
    "goi ca trich",
    "goi cu hu dua",
    "goi tom thit",
    "goi tai heo",
    "goi rau muong",
    "goi mieu",
    "goi bap chuoi",
    "cha gio",
    "nem ran",
    "nem nuong",
    "nem lui",
    "nem chua",
    "nem chua ran",
    "nem chua nuong",
    "nem thinh",
    "nem tai",
    "cha lua",
    "cha bo",
    "cha ca",
    "cha muc",
    "cha tom",
    "cha com",
    "cha que",
    "cha chien",
    "gio lua",
    "gio bo",
    "gio thu",
    "gio bi",
    "thit kho tau",
    "thit kho trung",
    "thit kho tieu",
    "thit kho dua",
    "thit kho mam ruoc",
    "thit heo quay",
    "thit heo luoc",
    "thit heo nuong",
    "thit heo xao sa ot",
    "thit heo rang chay canh",
    "thit bo kho",
    "bo kho",
    "bo luc lac",
    "bo xao hanh tay",
    "bo xao rau muong",
    "bo xao la lot",
    "bo nuong la lot",
    "bo nhung dam",
    "bo nhung me",
    "bo cuon la cai",
    "bo tai chanh",
    "bo la lot",
    "bo kho gung",
    "ga kho gung",
    "ga kho sa ot",
    "ga nuong",
    "ga luoc",
    "ga hap muoi",
    "ga hap hanh",
    "ga rang muoi",
    "ga chien mam",
    "ga chien bo",
    "ga xao sa ot",
    "ga xao nam",
    "ga ac tiem thuoc bac",
    "ga tan",
    "ga bop rau ram",
    "vit quay",
    "vit luoc",
    "vit kho gung",
    "vit nau chao",
    "vit nau mang",
    "vit om sau",
    "vit tiem",
    "vit xao sa ot",
    "ca kho to",
    "ca kho tieu",
    "ca kho rieng",
    "ca kho dua",
    "ca loc kho to",
    "ca loc nuong trui",
    "ca loc hap bau",
    "ca chien xu",
    "ca chien mam",
    "ca hap xi dau",
    "ca hap hanh gung",
    "ca nuong muoi ot",
    "ca nuong giấy bac",
    "ca bong lau kho",
    "ca keo kho rau ram",
    "ca ro kho to",
    "ca thu sot ca",
    "ca thu kho",
    "ca basa kho",
    "ca dieu hong chien",
    "ca dieu hong hap",
    "muc xao sa te",
    "muc hap gung",
    "muc chien gion",
    "muc nhoi thit",
    "muc nuong sa te",
    "tom ram",
    "tom rim",
    "tom kho tau",
    "tom chien xu",
    "tom hap bia",
    "tom nuong muoi ot",
    "tom sot me",
    "tom rang muoi",
    "cua rang me",
    "cua hap",
    "cua sot ot",
    "cua rang muoi",
    "ghe hap",
    "ghe rang me",
    "oc luoc",
    "oc xao sa te",
    "oc huong rang muoi",
    "oc len xao dua",
    "oc mong tay xao rau muong",
    "oc buou nhoi thit",
    "hen xuc banh da",
    "ngheu hap sa",
    "ngheu xao bo toi",
    "so long nuong mo hanh",
    "so diep nuong mo hanh",
    "hau nuong pho mai",
    "hau nuong mo hanh",
    "lau thai",
    "lau mam",
    "lau ca",
    "lau ca keo",
    "lau ca linh",
    "lau ca duoi",
    "lau hai san",
    "lau bo",
    "lau ga la e",
    "lau ga ot hiem",
    "lau ga nam",
    "lau vit nau chao",
    "lau de",
    "lau chay",
    "lau cua dong",
    "lau rieu cua",
    "lau ech",
    "lau nam",
    "canh chua ca",
    "canh chua tom",
    "canh chua ga",
    "canh chua rau muong",
    "canh chua bac ha",
    "canh bi do",
    "canh kho qua",
    "canh kho qua nhoi thit",
    "canh rau ngot",
    "canh rau den",
    "canh cua rau day",
    "canh bau tom",
    "canh cai thit bam",
    "canh cai ca ro",
    "canh mang",
    "canh mang chua",
    "canh rong bien",
    "canh nam",
    "canh trung ca chua",
    "canh suon non",
    "rau muong xao toi",
    "rau lang xao toi",
    "rau cai xao bo",
    "bong cai xao",
    "dua leo mam nem",
    "dua gia",
    "ca phao",
    "ca phao mam tom",
    "dua chua",
    "dua mon",
    "mam tom",
    "mam nem",
    "mam ca linh",
    "mam ca loc",
    "mam ruoc",
    "mam chung",
    "mam kho quet",
    "kho quet",
    "trung chien",
    "trung chien thit bam",
    "trung chien hanh",
    "trung kho thit",
    "trung vit lon",
    "trung cut lon",
    "trung hap",
    "trung cuon",
    "dau hu chien",
    "dau hu sot ca",
    "dau hu nhoi thit",
    "dau hu kho nam",
    "dau hu xao sa ot",
    "dau hu chien sa ot",
    "dau hu ky cuon",
    "cao lau",
    "bot chien",
    "bot chien trung",
    "sup cua",
    "sup mang cua",
    "sup ga",
    "sup bap",
    "sup hai san",
    "sup thap cam",
    "bo bia",
    "bo bia ngot",
    "pha lau",
    "pha lau bo",
    "pha lau heo",
    "pha lau nuoc dua",
    "tiet canh",
    "long heo luoc",
    "long heo xao dua",
    "long ga xao mien",
    "mien ga",
    "mien vit",
    "mien xao cua",
    "mien xao long ga",
    "mien xao thap cam",
    "mien ngan",
    "mien luon",
    "mien tron",
    "luon xao lan",
    "luon om chuoi dau",
    "ech xao sa ot",
    "ech chien bo",
    "ech kho tieu",
    "de hap",
    "de nuong",
    "de tai chanh",
    "de xao lan",
    "de nau chao",
    "chao de",
    "che ba mau",
    "che dau xanh",
    "che dau den",
    "che dau do",
    "che dau trang",
    "che bap",
    "che buoi",
    "che chuoi",
    "che khoai mon",
    "che khoai lang",
    "che troi nuoc",
    "che thap cam",
    "che hat sen",
    "che long nhan",
    "che sam bo luong",
    "che suong sa hat luu",
    "che khuc bach",
    "che thai",
    "che dau van",
    "che con ong",
    "che lam",
    "tau hu nuoc duong",
    "sua dau nanh",
    "sua bap",
    "sua dau xanh",
    "sinh to bo",
    "sinh to xoai",
    "sinh to mang cau",
    "sinh to dau",
    "sinh to sau rieng",
    "nuoc mia",
    "nuoc sam",
    "nuoc rau ma",
    "tra tac",
    "tra dao",
    "ca phe sua da",
    "ca phe den da",
    "bac xiu",
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
