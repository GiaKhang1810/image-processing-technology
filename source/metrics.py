from .util import normalize_text


def distance(expected: str, predicted: str) -> int:
    dp = [[0] * (len(predicted) + 1) for _ in range(len(expected) + 1)]

    for i in range(len(expected) + 1):
        dp[i][0] = i

    for j in range(len(predicted) + 1):
        dp[0][j] = j

    for i in range(1, len(expected) + 1):
        for j in range(1, len(predicted) + 1):
            cost = 0 if expected[i - 1] == predicted[j - 1] else 1

            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )

    return dp[len(expected)][len(predicted)]


def compute(expected: str, predicted: str) -> tuple[float, float]:
    expected = normalize_text(expected)
    predicted = normalize_text(predicted)

    cer_score: float
    if len(expected) == 0:
        cer_score = 0.0 if len(predicted) == 0 else 1.0
    else:
        cer_score = distance(expected, predicted) / len(expected)

    return max(0.0, 1.0 - cer_score), cer_score
