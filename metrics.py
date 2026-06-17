from re import sub
from unicodedata import category, normalize


def remove_vietnamese_accents(text: str) -> str:
    # Tách chữ và dấu tiếng Việt thành các ký tự riêng.
    text = normalize("NFD", text)

    # Giữ lại các ký tự không phải dấu.
    text = "".join(char for char in text if category(char) != "Mn")

    # Chuyển ký tự đ/Đ thành d/D để dễ so sánh OCR.
    text = text.replace("đ", "d").replace("Đ", "D")

    return text


def normalize_text(text: str) -> str:
    # Ép dữ liệu về string để tránh lỗi khi gặp None hoặc NaN.
    text = str(text)

    # Xóa dấu tiếng Việt.
    text = remove_vietnamese_accents(text)

    # Chuyển toàn bộ chữ về chữ thường.
    text = text.lower()

    # Xóa ký tự đặc biệt, chỉ giữ chữ cái, số và khoảng trắng.
    text = sub(r"[^a-z0-9\s]", " ", text)

    # Xóa khoảng trắng dư giữa các từ.
    text = " ".join(text.split())

    return text


def distance(bef: str, aft: str) -> int:
    # Tạo bảng quy hoạch động để tính số thao tác chỉnh sửa giữa hai chuỗi.
    dp = [[0] * (len(aft) + 1) for _ in range(len(bef) + 1)]

    # Khởi tạo cột đầu tiên, tương ứng với thao tác xóa ký tự.
    for i in range(len(bef) + 1):
        dp[i][0] = i

    # Khởi tạo hàng đầu tiên, tương ứng với thao tác thêm ký tự.
    for j in range(len(aft) + 1):
        dp[0][j] = j

    # Duyệt từng ký tự của chuỗi ban đầu.
    for i in range(1, len(bef) + 1):
        # Duyệt từng ký tự của chuỗi sau OCR.
        for j in range(1, len(aft) + 1):
            # Nếu hai ký tự giống nhau thì không tính lỗi, ngược lại tính một lỗi thay thế.
            cost = 0 if bef[i - 1] == aft[j - 1] else 1

            # Chọn số thao tác ít nhất giữa xóa, thêm hoặc thay thế ký tự.
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)

    return dp[len(bef)][len(aft)]


def cer(expected: str, predicted: str) -> float:
    # Chuẩn hóa text đúng.
    expected = normalize_text(expected)

    # Chuẩn hóa text OCR dự đoán.
    predicted = normalize_text(predicted)

    # Nếu text đúng rỗng thì xử lý riêng để tránh chia cho 0.
    if len(expected) == 0:
        return 0.0 if len(predicted) == 0 else 1.0

    return distance(expected, predicted) / len(expected)


def accuracy(cer_score: float) -> float:
    # Quy đổi CER thành accuracy gần đúng.
    return max(0.0, 1.0 - cer_score)


"""
File này dùng để đánh giá độ chính xác của OCR.
Text được chuẩn hóa bằng cách xóa dấu tiếng Việt, chuyển về chữ thường, xóa ký tự đặc biệt và gom khoảng trắng dư.
Hàm distance() tính Levenshtein Distance, tức số thao tác ít nhất để biến chuỗi OCR thành chuỗi đúng.
Hàm cer() tính Character Error Rate, CER càng thấp thì model OCR càng tốt.
Hàm accuracy() quy đổi CER thành độ chính xác gần đúng bằng công thức 1 - CER.
"""
