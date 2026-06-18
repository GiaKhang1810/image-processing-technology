from utils import normalize_text


def distance(expected: str, predicted: str) -> int:
    # Tạo bảng quy hoạch động để tính số thao tác chỉnh sửa giữa hai chuỗi.
    dp = [[0] * (len(predicted) + 1) for _ in range(len(expected) + 1)]

    # Khởi tạo cột đầu tiên, tương ứng với thao tác xóa ký tự.
    for i in range(len(expected) + 1):
        dp[i][0] = i

    # Khởi tạo hàng đầu tiên, tương ứng với thao tác thêm ký tự.
    for j in range(len(predicted) + 1):
        dp[0][j] = j

    # Duyệt từng ký tự của chuỗi ban đầu.
    for i in range(1, len(expected) + 1):
        # Duyệt từng ký tự của chuỗi sau OCR.
        for j in range(1, len(predicted) + 1):
            # Nếu hai ký tự giống nhau thì không tính lỗi, ngược lại tính một lỗi thay thế.
            cost = 0 if expected[i - 1] == predicted[j - 1] else 1

            # Chọn số thao tác ít nhất giữa xóa, thêm hoặc thay thế ký tự.
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)

    return dp[len(expected)][len(predicted)]


def metricser(expected: str, predicted: str) -> tuple[float, float]:
    expected = normalize_text(expected)

    # Chuẩn hóa text OCR dự đoán.
    predicted = normalize_text(predicted)

    # Nếu text đúng rỗng thì xử lý riêng để tránh chia cho 0.
    cer_score: float
    
    if len(expected) == 0:
        cer_score = 0.0 if len(predicted) == 0 else 1.0
    else:
        cer_score = distance(expected, predicted) / len(expected)

    return max(0.0, 1.0 - cer_score), cer_score
