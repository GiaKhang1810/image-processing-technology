from pathlib import Path
from numpy import array, asarray, clip, float32, uint8
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
from cv2 import (
    MORPH_CLOSE,
    MORPH_RECT,
    THRESH_BINARY_INV,
    THRESH_OTSU,
    filter2D,
    findNonZero,
    getStructuringElement,
    minAreaRect,
    morphologyEx,
    threshold as thresholder,
)

from utils import safe_float, safe_text

valid_modes = {"raw", "background", "gray", "contrast", "binary"}


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
        auto_sharpen: bool = True,
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
        self.__auto_sharpen = auto_sharpen
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

    # Cải thiện độ nét
    def __sharpen(self, image: Image.Image) -> Image.Image:
        gray = ImageOps.grayscale(image)

        kernel = array(
            [
                [0, -1, 0],
                [-1, 5, -1],
                [0, -1, 0],
            ],
            dtype=float32,
        )

        image_np = array(gray)
        image_np = filter2D(image_np, -1, kernel)

        return Image.fromarray(image_np)

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

        return removed

    # Tự động tìm góc xoay tốt nhất để làm thẳng dòng chữ trong ảnh.
    def __rotate(self, image: Image.Image) -> Image.Image:
        max_angle = 15.0

        # Dùng ảnh nhỏ để xác định góc nghiêng, tránh xử lý trực tiếp trên ảnh lớn.
        score_source = image.copy()
        score_source.thumbnail((900, 900), Image.Resampling.BILINEAR)

        gray = ImageOps.grayscale(score_source)
        gray = ImageOps.autocontrast(gray)
        gray_np = asarray(gray)

        # Tạo ma trận nhị phân 2D: chữ màu trắng, nền màu đen.
        _, binary_np = thresholder(
            gray_np,
            0,
            255,
            THRESH_BINARY_INV + THRESH_OTSU,
        )

        # Nối các pixel chữ gần nhau lại để OpenCV dễ xác định vùng chữ.
        kernel = getStructuringElement(MORPH_RECT, (30, 5))
        binary_np = morphologyEx(binary_np, MORPH_CLOSE, kernel)

        # Lấy tọa độ tất cả pixel chữ trong ma trận 2D.
        points = findNonZero(binary_np)

        if points is None:
            return image

        # Tìm hình chữ nhật nhỏ nhất bao quanh vùng chữ và lấy góc nghiêng.
        rect = minAreaRect(points)
        angle = float(rect[-1])

        # Chuẩn hóa góc trả về của OpenCV về góc cần xoay ảnh.

        if angle < -45:
            rotate_angle = -(90 + angle)
        elif angle > 45:
            rotate_angle = 90 - angle
        else:
            rotate_angle = -angle

        # Nếu góc quá lớn thì bỏ qua để tránh xoay sai ảnh.
        if abs(rotate_angle) > max_angle:
            return image

        image = image.rotate(
            angle=rotate_angle,
            resample=Image.Resampling.BICUBIC,
            expand=True,
            fillcolor=(255, 255, 255),
        )

        return image

    # Tiền xử lý ảnh theo mode: raw, background, gray, contrast hoặc binary.
    def modify(
        self, image_path: str, mode: str, threshold: int, name: str | None
    ) -> Image.Image:
        if mode not in valid_modes:
            raise ValueError(f"Invalid preprocess mode: {mode}")

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

        if self.__auto_sharpen:
            image = self.__sharpen(image)

            self.__save(image=image, name=name, mode=mode, threshold=threshold)
            return image

        image = ImageOps.grayscale(image)

        image = ImageEnhance.Contrast(image).enhance(min(self.__contrast_factor, 1.5))

        if self.__auto_sharpen:
            image = self.__sharpen(image)

        if mode == "gray":
            self.__save(image=image, name=name, mode=mode, threshold=threshold)
            return image

        if mode == "contrast":
            self.__save(image=image, name=name, mode=mode, threshold=threshold)
            return image

        image = image.filter(ImageFilter.MedianFilter(size=self.__median_filter_size))

        if mode == "binary":
            image = image.point(lambda pixel: 255 if pixel > threshold else 0)

        self.__save(image=image, name=name, mode=mode, threshold=threshold)
        return image
