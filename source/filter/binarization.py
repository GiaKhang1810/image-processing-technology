from PIL.Image import Image, fromarray
from PIL.ImageFilter import BoxBlur
from numpy import (
    arange,
    asarray,
    bincount,
    dot,
    float32,
    float64,
    ndarray,
    uint8,
    where,
)

from ..types.options import BinarizationOptions
from ..util import clamp, ensure_odd, get_name_enable, validate_one_True


class Binarization:
    def __init__(self, options: BinarizationOptions) -> None:
        self.__enable = bool(options["enable"])

        if self.__enable:
            validate_one_True(obj=options["method"])

        self.__method = get_name_enable(obj=options["method"])

        self.__threshold = int(
            clamp(value=options["threshold"], min_value=0, max_value=255)
        )

        self.__blockSize = int(
            ensure_odd(
                value=int(clamp(value=options["blockSize"], min_value=3, max_value=101))
            )
        )

        self.__constant = float(
            clamp(value=options["constant"], min_value=-50.0, max_value=50.0)
        )

    def binarize(self, image: Image) -> Image:
        if not self.__enable:
            return image

        gray = image.convert("L")

        if self.__method == "fixed":
            return self.__fixed(image=gray)

        if self.__method == "adaptive":
            return self.__adaptive(image=gray)

        if self.__method == "otsu":
            return self.__otsu(image=gray)

        return gray

    def __fixed(self, image: Image) -> Image:
        return image.point(
            lambda pixel: 255 if pixel > self.__threshold else 0,
            mode="L",
        )

    def __adaptive(self, image: Image) -> Image:
        radius = self.__blockSize // 2

        mean = image.filter(BoxBlur(radius=radius))

        source = asarray(image, dtype=float32)
        local_mean = asarray(mean, dtype=float32)

        result = where(
            source > local_mean - self.__constant,
            255,
            0,
        ).astype(uint8)

        return fromarray(result, mode="L")

    def __otsu(self, image: Image) -> Image:
        array = asarray(image, dtype=uint8)

        threshold = self.__get_otsu_threshold(array=array)

        result = where(array > threshold, 255, 0).astype(uint8)

        return fromarray(result, mode="L")

    def __get_otsu_threshold(self, array: ndarray) -> int:
        hist = bincount(array.ravel(), minlength=256).astype(float64)

        total = array.size

        if total == 0:
            return self.__threshold

        sum_total = dot(arange(256), hist)

        weight_background = 0.0
        sum_background = 0.0

        max_variance = 0.0
        threshold = 0

        for value in range(256):
            weight_background += hist[value]

            if weight_background == 0:
                continue

            weight_foreground = total - weight_background

            if weight_foreground == 0:
                break

            sum_background += value * hist[value]

            mean_background = sum_background / weight_background
            mean_foreground = (sum_total - sum_background) / weight_foreground

            variance = (
                weight_background
                * weight_foreground
                * (mean_background - mean_foreground) ** 2
            )

            if variance > max_variance:
                max_variance = variance
                threshold = value

        return int(threshold)
