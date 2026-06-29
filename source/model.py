from numpy import array
from PIL.Image import Image

from .types.options import ModelOptions

from .util import validate_one_True


class Model:
    def __init__(self, modelOptions: ModelOptions):
        modelname = "easyocr"

        validate_one_True(obj=modelOptions)

        if modelOptions["tesseract"]:
            modelname = "tesseract"

        self.__modelname = modelname

        match modelname:
            case "tesseract":
                from pytesseract import image_to_string

                self.__devicename = None

                def readtext(image: Image) -> str:
                    content = str(
                        image_to_string(
                            image=image, config=r"--oem 3 --psm 6", lang="eng"
                        )
                    )

                    return content.strip()

                self.__readtext = readtext

            case "easyocr":
                from torch.cuda import is_available, get_device_name
                from easyocr import Reader

                useGraphics = is_available()
                if useGraphics:
                    self.__devicename = get_device_name(0)
                else:
                    self.__devicename = None

                reader = Reader(lang_list=["vi", "en"], gpu=useGraphics)

                def readtext(image: Image) -> str:
                    image_np = array(image)
                    content = reader.readtext(image=image_np, detail=0)

                    return " ".join(content)

                self.__readtext = readtext

    @property
    def modelname(self) -> str:
        return self.__modelname

    @property
    def devicename(self) -> str:
        return self.__devicename

    def readtext(self, image: Image) -> str:
        content = self.__readtext(image=image)

        return content
