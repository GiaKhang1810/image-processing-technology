from os import makedirs
from os.path import join
from PIL import Image, ImageOps, ImageEnhance, ImageFilter

class Preprocess:
    def __init__(self, threshold: int = 150, processed_dir: str | None = None) -> None:
        if processed_dir is not None:
            makedirs(processed_dir, exist_ok=True)

        self.threshold = threshold
        self.processed_dir = processed_dir

    def modify(self, path: str, mode: str = "contrast", name: str | None = None) -> Image:
        try:
            image = Image.open(path)
            image = image.convert("RGB")

            if mode == "raw":
                if name is not None:
                    endpoint = join(self.processed_dir, name)
                    image.save(endpoint)

                return image

            width, height = image.size
            image = image.resize(
                (width * 2, height * 2),
                Image.Resampling.LANCZOS
            )

            image = ImageOps.grayscale(image)

            if mode == "gray":
                if name is not None:
                    endpoint = join(self.processed_dir, name)
                    image.save(endpoint)

                return image

            image = ImageEnhance.Contrast(image).enhance(2.0)
            image = image.filter(ImageFilter.SHARPEN)

            if mode == "contrast":
                if name is not None:
                    endpoint = join(self.processed_dir, name)
                    image.save(endpoint)

                return image

            image = image.filter(
                ImageFilter.MedianFilter(size=3)
            )

            if mode == "binary":
                image = image.point(lambda pixel: 255 if pixel > self.threshold else 0)

            if name is not None:
                endpoint = join(self.processed_dir, name)
                image.save(endpoint)

            return image
        except Exception as error:
            raise error