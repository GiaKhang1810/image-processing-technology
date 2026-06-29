from PIL.Image import Image, Resampling
from PIL.ImageOps import grayscale
from PIL.ImageEnhance import Contrast
from PIL.ImageFilter import MedianFilter

from ..types.options import EnhancerOptions

from ..util import clamp, ensure_odd


class Enhancer:
    def __init__(self, options: EnhancerOptions) -> None:
        self.__enable = options["enable"]

        self.__resizeScale = int(
            clamp(value=options["resizeScale"], min_value=1, max_value=4)
        )

        self.__contrastfactor = float(
            clamp(value=options["contrastfactor"], min_value=0.5, max_value=2.0)
        )
        self.__medianfilterSize = ensure_odd(
            value=options["medianfilterSize"], min_value=1, max_value=9
        )

    def enchance(self, image: Image) -> Image:
        if not self.__enable:
            return image

        resize = self.__resizeScale
        width, height = image.size
        image = image.resize(
            (width * resize, height * resize),
            Resampling.LANCZOS,
        )

        image = grayscale(image)

        if self.__medianfilterSize > 1:
            medianfilter = MedianFilter(size=self.__medianfilterSize)
            image = image.filter(filter=medianfilter)

        image = Contrast(image=image).enhance(factor=self.__contrastfactor)

        return image
