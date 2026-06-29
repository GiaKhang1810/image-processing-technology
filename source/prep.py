from pathlib import Path

from PIL.Image import Image, open

from .filter.binarization import Binarization
from .filter.enhancer import Enhancer
from .filter.remover import Remover
from .filter.rotation import Rotation
from .filter.sharpness import Sharpness

from .types.options import PreprocessorOptions


class Prep(Binarization, Enhancer, Remover, Rotation, Sharpness):
    def __init__(self, options: PreprocessorOptions) -> None:
        Binarization.__init__(self=self, options=options["binarization"])
        Enhancer.__init__(self=self, options=options["enhancer"])
        Remover.__init__(self=self, options=options["remover"])
        Rotation.__init__(self=self, options=options["rotation"])
        Sharpness.__init__(self=self, options=options["sharpness"])

        self.__enableSave = bool(options["storage"]["processed"])

    def modify(self, IMAGE_P: Path, PROCESSED_P: Path | str, FILE_NAME: str) -> Image:
        with open(fp=IMAGE_P) as source:
            image = source.convert("RGB")

        image = self.enchance(image=image)
        image = self.remove(image=image)
        image = self.rotate(image=image)
        image = self.sharpen(image=image)
        image = self.binarize(image=image)

        if self.__enableSave and PROCESSED_P and FILE_NAME:
            PROCESSED_P = Path(PROCESSED_P)

            PROCESSED_P.mkdir(parents=True, exist_ok=True)
            ENDPOINT_P = PROCESSED_P / f"{Path(FILE_NAME).stem}.png"
            image.save(fp=ENDPOINT_P)

        return image
