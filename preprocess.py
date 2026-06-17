from os import makedirs
from os.path import join
from PIL import Image, ImageOps, ImageEnhance, ImageFilter


class Preprocess:
    def __init__(self, processed_dir: str | None = None) -> None:
        # Nếu có truyền thư mục lưu ảnh đã xử lý thì tạo thư mục đó.
        if processed_dir is not None:
            makedirs(processed_dir, exist_ok=True)

        # Lưu lại đường dẫn thư mục chứa ảnh đã xử lý.
        self.processed_dir = processed_dir

    def save_image(self, image: Image.Image, name: str | None) -> None:
        # Nếu không truyền tên file thì không lưu ảnh.
        if name is None:
            return

        # Nếu không có thư mục lưu ảnh thì không lưu ảnh.
        if self.processed_dir is None:
            return

        # Tạo đường dẫn đầy đủ đến file ảnh cần lưu.
        endpoint = join(self.processed_dir, name)

        # Lưu ảnh đã xử lý ra file.
        image.save(endpoint)

    def modify(
        self,
        path: str,
        mode: str = "contrast",
        threshold: int = 150,
        name: str | None = None,
    ) -> Image.Image:
        # Mở ảnh từ đường dẫn đầu vào.
        image = Image.open(path)

        # Chuyển ảnh sang RGB để tránh lỗi ảnh có alpha channel.
        image = image.convert("RGB")

        # Nếu mode là raw thì giữ nguyên ảnh gốc.
        if mode == "raw":
            # Lưu ảnh raw nếu có truyền name.
            self.save_image(image, name)

            # Trả về ảnh gốc.
            return image

        # Lấy kích thước ảnh ban đầu.
        width, height = image.size

        # Phóng to ảnh lên 2 lần để chữ rõ hơn khi OCR.
        image = image.resize((width * 2, height * 2), Image.Resampling.LANCZOS)

        # Chuyển ảnh sang grayscale.
        image = ImageOps.grayscale(image)

        # Nếu mode là gray thì chỉ xử lý đến grayscale.
        if mode == "gray":
            # Lưu ảnh grayscale nếu có truyền name.
            self.save_image(image, name)

            # Trả về ảnh grayscale.
            return image

        # Tăng độ tương phản của ảnh.
        image = ImageEnhance.Contrast(image).enhance(2.0)

        # Làm nét ảnh để chữ rõ hơn.
        image = image.filter(ImageFilter.SHARPEN)

        # Nếu mode là contrast thì dừng sau bước tăng tương phản và làm nét.
        if mode == "contrast":
            # Lưu ảnh contrast nếu có truyền name.
            self.save_image(image, name)

            # Trả về ảnh đã tăng tương phản.
            return image

        # Khử nhiễu nhẹ bằng bộ lọc trung vị.
        image = image.filter(ImageFilter.MedianFilter(size=3))

        # Nếu mode là binary thì chuyển ảnh sang đen trắng bằng threshold.
        if mode == "binary":
            # Pixel lớn hơn threshold thành trắng, ngược lại thành đen.
            image = image.point(lambda pixel: 255 if pixel > threshold else 0)

        # Lưu ảnh cuối cùng nếu có truyền name.
        self.save_image(image, name)

        # Trả về ảnh đã xử lý.
        return image


"""
File này dùng Pillow để tiền xử lý ảnh trước khi đưa vào OCR.
Class Preprocess hỗ trợ các mode xử lý gồm raw, gray, contrast và binary.
Mode raw giữ nguyên ảnh gốc.
Mode gray phóng to ảnh và chuyển sang ảnh xám.
Mode contrast phóng to ảnh, chuyển xám, tăng tương phản và làm nét.
Mode binary xử lý thêm bước khử nhiễu và chuyển ảnh về đen trắng bằng threshold.
Ảnh sau xử lý sẽ được lưu vào "processed_dir" nếu có truyền name.
"""
