from PIL.Image import Image
from numpy import array, ndarray


class Model:
    def __init__(self, model: str | None = "easyocr") -> None:
        # Chuẩn hóa tên model để tránh lỗi khi truyền None.
        model = model or "easyocr"

        # Chọn model OCR theo tên được truyền vào.
        match model:
            case "easyocr":
                from easyocr import Reader

                # Khởi tạo EasyOCR với tiếng Việt và tiếng Anh.
                easyReader = Reader(["vi", "en"], gpu=True)

                # Hàm đọc ảnh dành riêng cho EasyOCR.
                def reader(image: ndarray) -> str:
                    # EasyOCR đọc ảnh dạng numpy array và chỉ lấy nội dung text.
                    content = easyReader.readtext(image, detail=0)

                    return " ".join(content)

                # Lưu hàm reader vào object để dùng lại trong hàm read().
                self.reader = reader

            case "tesseract":
                from pytesseract import image_to_string

                # Hàm đọc ảnh dành riêng cho Tesseract OCR.
                def reader(image: Image) -> str:
                    # Tesseract đọc trực tiếp ảnh dạng PIL Image.
                    return image_to_string(
                        image=image, lang="vie+eng", config="--psm 6"
                    ).strip()

                # Lưu hàm reader vào object để dùng lại trong hàm read().
                self.reader = reader

            case _:
                # Báo lỗi nếu model không nằm trong danh sách hỗ trợ.
                raise Exception("Model not supported")

        # Lưu tên model hiện tại vào object.
        self.model = model

    def read(self, image: Image) -> str:
        # Nếu dùng Tesseract thì truyền trực tiếp PIL Image.
        if self.model == "tesseract":
            return self.reader(image)

        # Nếu dùng EasyOCR thì chuyển PIL Image sang numpy array.
        image_np = array(image)

        # Gọi hàm reader tương ứng với model hiện tại.
        return self.reader(image_np)


"""
File này dùng để quản lý model OCR.
Class Model hỗ trợ hai model là EasyOCR và Tesseract OCR.
EasyOCR nhận ảnh dạng numpy array nên cần chuyển ảnh bằng array(image).
Tesseract OCR nhận ảnh dạng PIL Image nên có thể truyền ảnh trực tiếp.
Hàm read() giúp chương trình chính không cần quan tâm mỗi model cần kiểu dữ liệu ảnh khác nhau.
"""
