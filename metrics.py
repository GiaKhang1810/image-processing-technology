from utils import normalize_text


# Tính khoảng cách Levenshtein giữa chuỗi đúng và chuỗi OCR dự đoán.
def distance(expected: str, predicted: str) -> int:
    # Tạo bảng quy hoạch động để lưu số thao tác chỉnh sửa tối thiểu.
    dp = [[0] * (len(predicted) + 1) for _ in range(len(expected) + 1)]

    # Khởi tạo cột đầu tiên, tương ứng với số thao tác xóa ký tự.
    for i in range(len(expected) + 1):
        dp[i][0] = i

    # Khởi tạo hàng đầu tiên, tương ứng với số thao tác thêm ký tự.
    for j in range(len(predicted) + 1):
        dp[0][j] = j

    # Duyệt từng ký tự trong chuỗi đúng.
    for i in range(1, len(expected) + 1):
        # Duyệt từng ký tự trong chuỗi OCR dự đoán.
        for j in range(1, len(predicted) + 1):
            # Nếu hai ký tự giống nhau thì không có lỗi.
            # Nếu khác nhau thì tính một lỗi thay thế.
            cost = 0 if expected[i - 1] == predicted[j - 1] else 1

            # Chọn số thao tác ít nhất giữa xóa, thêm hoặc thay thế ký tự.
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )

    # Trả về khoảng cách chỉnh sửa nhỏ nhất giữa hai chuỗi.
    return dp[len(expected)][len(predicted)]


# Tính độ chính xác và chỉ số CER giữa text đúng và text OCR dự đoán.
def metricser(expected: str, predicted: str) -> tuple[float, float]:
    # Chuẩn hóa text đúng trước khi so sánh.
    expected = normalize_text(expected)

    # Chuẩn hóa text OCR dự đoán trước khi so sánh.
    predicted = normalize_text(predicted)

    # Khai báo biến lưu chỉ số CER.
    cer_score: float

    # Nếu text đúng rỗng thì xử lý riêng để tránh chia cho 0.
    if len(expected) == 0:
        # Nếu cả hai chuỗi đều rỗng thì không có lỗi.
        # Nếu chuỗi dự đoán có nội dung thì xem như sai hoàn toàn.
        cer_score = 0.0 if len(predicted) == 0 else 1.0
    else:
        # Tính CER bằng khoảng cách chỉnh sửa chia cho độ dài chuỗi đúng.
        cer_score = distance(expected, predicted) / len(expected)

    # Trả về accuracy và CER.
    return max(0.0, 1.0 - cer_score), cer_score
