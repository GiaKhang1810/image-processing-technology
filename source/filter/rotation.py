from numpy import asarray, float32, arange, sum, uint8, where
from PIL.ImageFilter import FIND_EDGES
from PIL.ImageOps import autocontrast
from PIL.Image import Image, Resampling, new, fromarray

from ..types.options import RotationOptions
from ..util import clamp


class Rotation:
    def __init__(self, options: RotationOptions) -> None:
        self.__enable = bool(options["enable"])

        self.__step = float(clamp(value=options["step"], min_value=0.1, max_value=1.0))

        self.__minConfidence = float(
            clamp(value=options["minConfidence"], min_value=0.0, max_value=1.0)
        )

        self.__maxangle = float(
            clamp(value=options["maxangle"], min_value=1.0, max_value=15.0)
        )

        self.__minangle = float(
            clamp(value=options["minangle"], min_value=0.1, max_value=self.__maxangle)
        )

    def rotate(self, image: Image) -> Image:
        if not self.__enable:
            return image

        angle, confidence = self.__detect_angle(image=image)

        if abs(angle) < self.__minangle:
            return image

        if confidence < self.__minConfidence:
            return image

        return image.rotate(
            angle=angle,
            resample=Resampling.BICUBIC,
            expand=True,
            fillcolor=self.__get_fill_color(image=image),
        )

    def __detect_angle(self, image: Image) -> tuple[float, float]:
        mask = self.__create_mask(image=image)

        base_score = self.__score_angle(mask=mask, angle=0.0)

        best_angle = 0.0
        best_score = base_score

        angles = arange(
            -self.__maxangle,
            self.__maxangle + self.__step,
            self.__step,
            dtype=float32,
        )

        for angle in angles:
            angle = float(angle)

            score = self.__score_angle(mask=mask, angle=angle)

            if score > best_score:
                best_score = score
                best_angle = angle

        if base_score <= 0:
            return 0.0, 0.0

        confidence = (best_score - base_score) / base_score

        return best_angle, float(confidence)

    def __create_mask(self, image: Image) -> Image:
        gray = self.__to_gray(image=image)

        max_side = 900
        width, height = gray.size

        if max(width, height) > max_side:
            scale = max_side / max(width, height)

            gray = gray.resize(
                size=(int(width * scale), int(height * scale)),
                resample=Resampling.LANCZOS,
            )

        gray = autocontrast(gray)

        edges = gray.filter(FIND_EDGES)
        edges = autocontrast(edges)

        array = asarray(edges, dtype=uint8)

        threshold = max(20, int(array.mean() + array.std()))

        mask = where(array > threshold, 255, 0).astype(uint8)

        return fromarray(mask, mode="L")

    def __score_angle(self, mask: Image, angle: float) -> float:
        rotated = mask.rotate(
            angle=angle,
            resample=Resampling.NEAREST,
            expand=False,
            fillcolor=0,
        )

        array = asarray(rotated, dtype=uint8)

        rows = sum(array > 0, axis=1).astype(float32)

        if rows.size == 0:
            return 0.0

        mean = float(rows.mean())

        if mean <= 0:
            return 0.0

        return float(rows.var() / mean)

    def __to_gray(self, image: Image) -> Image:
        if image.mode in ("RGBA", "LA"):
            rgba = image.convert("RGBA")

            background = new("RGBA", rgba.size, (255, 255, 255, 255))
            background.alpha_composite(rgba)

            return background.convert("L")

        return image.convert("L")

    def __get_fill_color(self, image: Image):
        if image.mode == "RGBA":
            return (0, 0, 0, 0)

        if image.mode == "LA":
            return (255, 0)

        if image.mode == "L":
            return 255

        return (255, 255, 255)
