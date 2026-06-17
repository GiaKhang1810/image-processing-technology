from PIL.Image import Image
from numpy import array, ndarray

class Model:
    def __init__(self, model: str | None = "easyocr"):
        match model:
            case "easyocr":
                from easyocr import Reader

                easyReader = Reader(["vi", "en"], gpu=True)
                def reader(image: ndarray) -> str:
                    content = easyReader.readtext(image, detail=0)
                    return " ".join(content)

                self.reader = reader
            case "tesseract":
                from pytesseract import image_to_string

                def reader(image: Image) -> str:
                    return image_to_string(
                        image,
                        lang="vie+eng",
                        config="--psm 6"
                    ).strip()

                self.reader = reader
            case "paddle":
                from paddleocr import PaddleOCR

                padReader = PaddleOCR(
                    lang="vi",
                    use_angle_cls=True,
                    show_log=False
                )

                def reader(image: ndarray) -> str:
                    content = []

                    for page in padReader.predict(image):
                        if page is None:
                            continue

                        for line in page:
                            content.append(line[1][0])

                    return " ".join(content)

                self.reader = reader
            case _:
                raise Exception("Model not supported")

        self.model = model

    def read(self, image: Image) -> str:
        try:
            image_np: Image | ndarray

            if self.model == "tesseract":
                image_np = Image
            else:
                image_np = array(image)

            return self.reader(image_np)
        except Exception as error:
            raise error