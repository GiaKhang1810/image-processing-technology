from argparse import ArgumentParser
from csv import QUOTE_ALL, writer
from pathlib import Path
from random import choices, randint, random, uniform
from string import ascii_uppercase, digits

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont


def get_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def random_content() -> str:
    chars = ascii_uppercase + digits

    parts = ["".join(choices(chars, k=randint(3, 6))) for _ in range(randint(2, 4))]

    return " ".join(parts)


def measure_text(content: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    dummy_image = Image.new("RGB", (1, 1), (255, 255, 255))
    dummy_draw = ImageDraw.Draw(dummy_image)

    box = dummy_draw.textbbox((0, 0), content, font=font)

    text_width = box[2] - box[0]
    text_height = box[3] - box[1]

    return text_width, text_height


def create_text_image(content: str) -> Image.Image:
    font_size = randint(24, 40)
    font = get_font(font_size)

    text_width, text_height = measure_text(content, font)

    padding_x = randint(40, 70)
    padding_y = randint(30, 50)

    width = max(420, text_width + padding_x * 2)
    height = max(120, text_height + padding_y * 2)

    background_color = randint(235, 255)

    image = Image.new(
        "RGB",
        (width, height),
        (background_color, background_color, background_color),
    )

    draw = ImageDraw.Draw(image)

    x = (width - text_width) // 2
    y = (height - text_height) // 2

    text_color = randint(0, 60)

    draw.text(
        (x, y),
        content,
        fill=(text_color, text_color, text_color),
        font=font,
    )

    return image


def apply_rotation(image: Image.Image) -> Image.Image:
    if random() < 0.7:
        angle = uniform(-7.0, 7.0)

        return image.rotate(
            angle,
            expand=True,
            fillcolor=(255, 255, 255),
            resample=Image.Resampling.BICUBIC,
        )

    return image


def apply_blur(image: Image.Image) -> Image.Image:
    if random() < 0.7:
        radius = uniform(0.8, 2.4)
        image = image.filter(ImageFilter.GaussianBlur(radius))

    if random() < 0.25:
        radius = uniform(2.5, 4.0)
        image = image.filter(ImageFilter.GaussianBlur(radius))

    return image


def apply_brightness(image: Image.Image) -> Image.Image:
    if random() < 0.65:
        factor = uniform(0.75, 1.45)

        return ImageEnhance.Brightness(image).enhance(factor)

    return image


def apply_contrast(image: Image.Image) -> Image.Image:
    if random() < 0.6:
        factor = uniform(0.7, 1.5)

        return ImageEnhance.Contrast(image).enhance(factor)

    return image


def apply_glare(image: Image.Image) -> Image.Image:
    if random() >= 0.5:
        return image

    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    width, height = image.size

    x = randint(0, width)
    y = randint(0, height)

    glare_width = randint(width // 4, width // 2)
    glare_height = randint(height // 4, height // 2)

    draw.ellipse(
        (
            x - glare_width,
            y - glare_height,
            x + glare_width,
            y + glare_height,
        ),
        fill=(255, 255, 255, randint(70, 140)),
    )

    if random() < 0.5:
        line_y = randint(0, height)
        draw.rectangle(
            (
                0,
                max(0, line_y - randint(4, 12)),
                width,
                min(height, line_y + randint(4, 12)),
            ),
            fill=(255, 255, 255, randint(35, 80)),
        )

    overlay = overlay.filter(ImageFilter.GaussianBlur(randint(18, 36)))

    image = Image.alpha_composite(image.convert("RGBA"), overlay)

    return image.convert("RGB")


def apply_light_bloom(image: Image.Image) -> Image.Image:
    if random() >= 0.35:
        return image

    width, height = image.size

    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for _ in range(randint(1, 3)):
        x1 = randint(0, width)
        y1 = randint(0, height)

        radius_x = randint(width // 6, width // 3)
        radius_y = randint(height // 6, height // 3)

        draw.ellipse(
            (
                x1 - radius_x,
                y1 - radius_y,
                x1 + radius_x,
                y1 + radius_y,
            ),
            fill=(255, 255, 255, randint(40, 90)),
        )

    overlay = overlay.filter(ImageFilter.GaussianBlur(randint(20, 45)))

    image = Image.alpha_composite(image.convert("RGBA"), overlay)
    image = image.convert("RGB")

    if random() < 0.5:
        image = ImageEnhance.Brightness(image).enhance(uniform(1.05, 1.25))

    return image


def apply_noise(image: Image.Image) -> Image.Image:
    if random() >= 0.45:
        return image

    noise = Image.effect_noise(
        image.size,
        uniform(8, 24),
    ).convert("RGB")

    return Image.blend(
        image,
        noise,
        uniform(0.05, 0.14),
    )


def apply_random_effects(image: Image.Image) -> Image.Image:
    image = apply_rotation(image)
    image = apply_blur(image)
    image = apply_brightness(image)
    image = apply_contrast(image)
    image = apply_glare(image)
    image = apply_light_bloom(image)
    image = apply_noise(image)

    return image.convert("RGB")


def generate_split(dataset_dir: Path, split: str, count: int) -> None:
    images_dir = dataset_dir / split / "images"
    labels_path = dataset_dir / split / "labels.csv"

    images_dir.mkdir(parents=True, exist_ok=True)

    with labels_path.open("w", newline="", encoding="utf-8") as file:
        csv_writer = writer(file, quoting=QUOTE_ALL)
        csv_writer.writerow(["filename", "content"])

        for index in range(1, count + 1):
            content = random_content()

            image = create_text_image(content)
            image = apply_random_effects(image)

            filename = f"image_{index}.png"

            image.save(images_dir / filename)

            csv_writer.writerow([filename, content])


def main() -> None:
    parser = ArgumentParser()

    parser.add_argument("--dataset", type=str, default="dataset")
    parser.add_argument("--tune", type=int, default=200)
    parser.add_argument("--evaluate", type=int, default=200)

    args = parser.parse_args()

    dataset_dir = Path(args.dataset)

    if args.tune > 0:
        generate_split(dataset_dir, "tune", args.tune)

    if args.evaluate > 0:
        generate_split(dataset_dir, "evaluate", args.evaluate)


if __name__ == "__main__":
    main()
