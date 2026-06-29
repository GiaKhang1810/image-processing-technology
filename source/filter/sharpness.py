from PIL.ImageFilter import UnsharpMask, Kernel, Filter
from PIL.Image import Image, merge

from ..types.options import SharpnessOptions
from ..util import clamp, validate_one_True, get_name_enable


class Sharpness:
    def __init__(self, options: SharpnessOptions) -> None:
        self.__enable = bool(options["enable"])

        if self.__enable:
            validate_one_True(obj=options["method"])

        self.__methodname = get_name_enable(obj=options["method"])

        self.__sigma = float(
            clamp(value=options["sigma"], min_value=0.1, max_value=5.0)
        )

        self.__amount = float(
            clamp(value=options["amount"], min_value=0.0, max_value=5.0)
        )

        self.__matrix = self.__normalize_matrix(matrix=options["matrix"])

    def sharpen(self, image: Image) -> Image:
        if not self.__enable:
            return image

        if self.__methodname == "unsharp":
            return self.__apply_unsharp(image=image)

        if self.__methodname == "kernel":
            return self.__apply_kernel(image=image)

        return image

    def __apply_unsharp(self, image: Image) -> Image:
        return self.__apply_filter(
            image=image,
            image_filter=UnsharpMask(
                radius=self.__sigma,
                percent=int(self.__amount * 100),
                threshold=0,
            ),
        )

    def __apply_kernel(self, image: Image) -> Image:
        scale = sum(self.__matrix)

        if scale == 0:
            scale = 1.0

        return self.__apply_filter(
            image=image,
            image_filter=Kernel(
                size=(3, 3),
                kernel=self.__matrix,
                scale=scale,
                offset=0,
            ),
        )

    def __apply_filter(self, image: Image, image_filter: Filter) -> Image:
        if image.mode == "RGBA":
            rgb = image.convert("RGB")
            alpha = image.getchannel("A")

            filtered = rgb.filter(image_filter)
            filtered.putalpha(alpha)

            return filtered

        if image.mode == "LA":
            gray = image.getchannel("L")
            alpha = image.getchannel("A")

            filtered = gray.filter(image_filter)

            return merge("LA", (filtered, alpha))

        if image.mode not in ("L", "RGB"):
            image = image.convert("RGB")

        return image.filter(image_filter)

    def __normalize_matrix(self, matrix: list[list[float]]) -> list[float]:
        if len(matrix) != 3:
            raise ValueError("Sharpness matrix must have 3 rows")

        result: list[float] = []

        for row in matrix:
            if len(row) != 3:
                raise ValueError("Sharpness matrix must have 3 columns")

            for value in row:
                result.append(float(value))

        return result
