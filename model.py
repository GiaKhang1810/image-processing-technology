from PIL.Image import Image
from numpy import array, ndarray
from platform import processor


class Model:
    # Khởi tạo model OCR theo tên được chọn và tạo hàm đọc ảnh tương ứng.
    def __init__(self, modelname: str | None = "easyocr") -> None:
        modelname = modelname or "easyocr"

        match modelname:
            case "easyocr":
                from torch import cuda
                from easyocr import Reader

                using_graphics_card = cuda.is_available()
                if using_graphics_card:
                    print(f"Using graphics card: {cuda.get_device_name(0)}")
                else:
                    print("Graphics card not avaible")

                print(f"Using processor: {processor()}")

                # EasyOCR nhận ảnh dạng numpy array và trả về danh sách các đoạn text.
                easyReader = Reader(["vi", "en"], gpu=using_graphics_card)

                def reader(image: ndarray) -> str:
                    content = easyReader.readtext(image=image, detail=0)

                    return " ".join(content)

                self.__reader = reader

            case "tesseract":
                from pytesseract import image_to_string

                # Tesseract nhận ảnh PIL Image và đọc text theo ngôn ngữ Việt + Anh.
                def reader(image: Image) -> str:
                    return image_to_string(
                        image=image, lang="vie+eng", config="--oem 1 --psm 6"
                    ).strip()

                self.__reader = reader

            case _:
                raise Exception("Model not supported")

        self.__modelname = modelname

    @property
    def modelname(self) -> str:
        return self.__modelname

    # Đọc text từ ảnh, tự chuyển kiểu dữ liệu nếu model đang dùng là EasyOCR.
    def read(self, image: Image) -> str:
        if self.__modelname == "tesseract":
            return self.__reader(image)

        image_np = array(image)

        return self.__reader(image_np)
