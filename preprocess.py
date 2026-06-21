from pathlib import Path
from numpy import asarray, arange, clip, uint8
from PIL import Image, ImageOps, ImageEnhance, ImageFilter

from utils import safe_float, safe_text


class Preprocesser:
    # Khởi tạo cấu hình tiền xử lý và chuẩn bị thư mục lưu ảnh nếu cần.
    def __init__(
        self,
        processed_dir: Path | None,
        resize_scale: int = 2,
        median_filter_size: int = 3,
        contrast_factor: float = 2.0,
        background_blur_radius: int = 25,
        auto_rotate: bool = True,
        save_processed: bool = True,
    ) -> None:
        if processed_dir is not None and not processed_dir.is_dir():
            processed_dir.mkdir(parents=True, exist_ok=True)

        self.__processed_dir = processed_dir
        self.__resize_scale = resize_scale
        self.__median_filter_size = median_filter_size
        self.__contrast_factor = contrast_factor
        self.__background_blur_radius = background_blur_radius
        self.__auto_rotate = auto_rotate
        self.__save_processed = save_processed

    # Lưu ảnh đã xử lý với tên file chứa đầy đủ thông tin cấu hình.
    def __save(
        self, image: Image.Image, name: str | None, mode: str, threshold: int
    ) -> None:
        if not self.__save_processed:
            return

        if name is None:
            return

        if self.__processed_dir is None:
            return

        base_name = Path(name).stem
        base_name = safe_text(base_name)

        filename = (
            f"{base_name}"
            f"__mode-{mode}"
            f"__threshold-{threshold}"
            f"__resize-{self.__resize_scale}"
            f"__median-{self.__median_filter_size}"
            f"__contrast-{safe_float(self.__contrast_factor)}"
            f"__bgblur-{self.__background_blur_radius}"
            f"__rotate-{int(self.__auto_rotate)}"
            f".png"
        )

        endpoint = self.__processed_dir / filename

        image.save(endpoint)

    # Xóa nền bằng cách làm mờ nền, chuẩn hóa độ sáng rồi tăng tương phản ảnh.
    def __remove(self, image: Image.Image) -> Image.Image:
        gray = ImageOps.grayscale(image)

        background = gray.filter(
            ImageFilter.GaussianBlur(radius=self.__background_blur_radius)
        )

        gray_np = asarray(gray).astype("float32")
        bg_np = asarray(background).astype("float32")

        # Tránh lỗi chia cho 0 khi chuẩn hóa ảnh theo nền.
        bg_np[bg_np == 0] = 1

        normalized = gray_np / bg_np * 255
        normalized = clip(normalized, 0, 255).astype(uint8)

        removed = Image.fromarray(normalized)
        removed = ImageOps.autocontrast(removed)
        removed = ImageEnhance.Contrast(removed).enhance(
            min(self.__contrast_factor, 1.5)
        )
        removed = removed.filter(ImageFilter.SHARPEN)

        return removed

    # Tự động tìm góc xoay tốt nhất để làm thẳng dòng chữ trong ảnh.
    def __rotate(self, image: Image.Image) -> Image.Image:
        max_angle = 15.0
        coarse_step = 1.0
        fine_step = 0.2

        # Dùng ảnh nhỏ để tìm góc xoay, giúp giảm thời gian xử lý.
        score_source = image.copy()
        score_source.thumbnail((900, 900), Image.Resampling.BILINEAR)

        # Chuẩn bị ảnh nhị phân để tính độ thẳng của các dòng chữ.
        gray = ImageOps.grayscale(score_source)
        gray = ImageOps.autocontrast(gray)
        gray_np = asarray(gray)

        binary_np = ((gray_np < 200) * 255).astype(uint8)
        binary_image = Image.fromarray(binary_np)

        # Góc có phương sai histogram ngang cao hơn thường làm dòng chữ thẳng hơn.
        def score_angle(angle: float) -> float:
            rotated = binary_image.rotate(
                angle,
                resample=Image.Resampling.BICUBIC,
                expand=True,
                fillcolor=0,
            )

            rotated_np = asarray(rotated)
            horizontal_hist = rotated_np.sum(axis=1)

            return float(horizontal_hist.var())

        best_angle = 0.0
        best_score = -1.0

        # Quét thô để tìm vùng góc tốt nhất.
        for angle in arange(-max_angle, max_angle + coarse_step, coarse_step):
            current_score = score_angle(float(angle))

            if current_score > best_score:
                best_score = current_score
                best_angle = float(angle)

        fine_start = best_angle - coarse_step
        fine_end = best_angle + coarse_step + fine_step

        # Quét tinh quanh góc tốt nhất để tăng độ chính xác.
        for angle in arange(fine_start, fine_end, fine_step):
            current_score = score_angle(float(angle))

            if current_score > best_score:
                best_score = current_score
                best_angle = float(angle)

        image = image.rotate(
            angle=best_angle,
            resample=Image.Resampling.BICUBIC,
            expand=True,
            fillcolor=(255, 255, 255),
        )

        return image

    # Tiền xử lý ảnh theo mode: raw, background, gray, contrast hoặc binary.
    def modify(
        self, image_path: str, mode: str, threshold: int, name: str | None
    ) -> Image.Image:
        image = Image.open(image_path)
        image = image.convert("RGB")

        if mode == "raw":
            self.__save(image=image, name=name, mode=mode, threshold=threshold)

            return image

        resize = self.__resize_scale
        width, height = image.size

        image = image.resize(
            (width * resize, height * resize),
            Image.Resampling.LANCZOS,
        )

        if self.__auto_rotate:
            image = self.__rotate(image)

        if mode == "background":
            image = self.__remove(image)
            self.__save(image=image, name=name, mode=mode, threshold=threshold)

            return image

        # Các mode còn lại dùng ảnh xám làm nền xử lý.
        image = ImageOps.grayscale(image)

        if mode == "gray":
            self.__save(image=image, name=name, mode=mode, threshold=threshold)

            return image

        image = ImageEnhance.Contrast(image).enhance(min(self.__contrast_factor, 1.5))
        image = image.filter(ImageFilter.SHARPEN)

        if mode == "contrast":
            self.__save(image=image, name=name, mode=mode, threshold=threshold)

            return image

        # Binary dùng thêm lọc nhiễu trung vị trước khi nhị phân hóa ảnh.
        image = image.filter(ImageFilter.MedianFilter(size=self.__median_filter_size))

        if mode == "binary":
            image = image.point(lambda pixel: 255 if pixel > threshold else 0)

        self.__save(image=image, name=name, mode=mode, threshold=threshold)

        return image
