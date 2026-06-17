from re import sub
from unicodedata import category, normalize

def remove_vietnamese_accents(text: str) -> str:
    # Tách dấu tiếng Việt khỏi chữ.
    text = normalize("NFD", text)

    # Xóa các ký tự dấu.
    text = "".join(
        char for char in text
        if category(char) != "Mn"
    )

    text = text.replace("đ", "d").replace("Đ", "D")

    return text


def normalize_text(text: str) -> str:
    text = str(text)

    text = remove_vietnamese_accents(text)

    text = text.lower()

    text = sub(r"[^a-z0-9\s]", " ", text)

    text = " ".join(text.split())

    return text

def distance(bef: str, aft: str) -> int:
    dp = [
        [0] * (len(aft) + 1) for _ in range(len(bef) + 1)
    ]

    for i in range(len(bef) + 1):
        dp[i][0] = i

    for j in range(len(aft) + 1):
        dp[0][j] = j

    for i in range(1, len(bef) + 1):
        for j in range(1, len(aft) + 1):
            cost = 0 if bef[i - 1] == aft[j - 1] else 1

            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost
            )

    return dp[len(bef)][len(aft)]

def cer(expected: str, predicted: str) -> float:
    expected = normalize_text(expected)
    predicted = normalize_text(predicted)

    if len(expected) == 0:
        return 0.0 if len(predicted) == 0 else 1.0

    return distance(expected, predicted) / len(expected)

def accuracy(cer_score: float) -> float:
    return max(0.0, 1.0 - cer_score)