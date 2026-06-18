from PIL.Image import Image
from numpy import array, ndarray


class Model:
    def __init__(self, modelname: str | None = "easyocr") -> None:
        modelname = modelname or "easyocr"

        match modelname:
            case "easyocr":
                from easyocr import Reader

                easyReader = Reader(["vi", "en"], gpu=True)

                def reader(image: ndarray) -> str:
                    content = easyReader.readtext(image, detail=0)

                    return " ".join(content)

                self.__reader = reader

            case "tesseract":
                from pytesseract import image_to_string

                def reader(image: Image) -> str:
                    return image_to_string(
                        image=image, lang="vie+eng", config="--psm 6"
                    ).strip()

                self.__reader = reader

            case _:
                raise Exception("Model not supported")

        self.__modelname = modelname

    @property
    def modelname(self) -> str:
        return self.__modelname

    def read(self, image: Image) -> str:
        if self.__modelname == "tesseract":
            return self.__reader(image)

        image_np = array(image)

        return self.__reader(image_np)
