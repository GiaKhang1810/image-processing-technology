from numpy import array, clip, float32, uint8
from PIL.Image import Image, fromarray
from PIL.ImageOps import grayscale
from PIL.ImageFilter import GaussianBlur, MedianFilter
from cv2 import MORPH_CLOSE, MORPH_RECT, getStructuringElement, morphologyEx

from ..types.options import RemoverOptions

from ..util import validate_one_True, get_name_enable, clamp, ensure_odd


class Remover:
    def __init__(self, options: RemoverOptions):
        self.__enable = options["enable"]

        if options["enable"]:
            validate_one_True(obj=options["method"])

        self.__methodname = get_name_enable(obj=options["method"])

        self.__blurRadius = float(
            clamp(value=options["blurRadius"], min_value=1.0, max_value=45.0)
        )

        self.__kernelsize = ensure_odd(
            value=options["kernelsize"], min_value=1, max_value=9
        )

        self.__iterations = int(
            clamp(value=options["iterations"], min_value=1, max_value=5)
        )

    def __division(self, image: Image) -> Image:
        gaussian = GaussianBlur(radius=self.__blurRadius)
        bg = image.filter(filter=gaussian)

        image_a = array(object=image).astype(dtype=float32)
        bg_a = array(object=bg).astype(dtype=float32)

        removed = image_a / (bg_a + 1.0) * 255.0
        removed = clip(removed, 0, 255).astype(dtype=uint8)

        return fromarray(obj=removed)

    def __gaussian(self, image: Image) -> Image:
        gaussian = GaussianBlur(radius=self.__blurRadius)
        bg = image.filter(filter=gaussian)

        image_a = array(object=image).astype(dtype=float32)
        bg_a = array(object=bg).astype(dtype=float32)

        removed = image_a - bg_a + 255.0
        removed = clip(removed, 0, 255).astype(dtype=uint8)

        return fromarray(obj=removed)

    def __median(self, image: Image) -> Image:
        median = MedianFilter(size=self.__kernelsize)
        bg = image.filter(filter=median)

        image_a = array(object=image).astype(dtype=float32)
        bg_a = array(object=bg).astype(dtype=float32)

        removed = image_a - bg_a + 255.0
        removed = clip(removed, 0, 255).astype(dtype=uint8)

        return fromarray(obj=removed)

    def __morphology(self, image: Image) -> Image:
        image_a = array(object=image).astype(dtype=float32)

        kernel = getStructuringElement(
            shape=MORPH_RECT,
            ksize=(
                self.__kernelsize,
                self.__kernelsize,
            ),
        )

        bg_a = morphologyEx(
            src=image_a,
            op=MORPH_CLOSE,
            kernel=kernel,
            iterations=self.__iterations,
        )

        removed = image_a.astype(dtype=float32) - bg_a.astype(dtype=float32) + 255.0
        removed = clip(removed, 0, 255).astype(dtype=uint8)

        return fromarray(obj=removed)

    def remove(self, image: Image) -> Image:
        if not self.__enable:
            return image

        if image.mode != "L":
            image = grayscale(image)

        match self.__methodname:
            case "division":
                return self.__division(image=image)

            case "gaussian":
                return self.__gaussian(image=image)

            case "median":
                return self.__median(image=image)

            case "morphology":
                return self.__morphology(image=image)
