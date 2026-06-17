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
    expected = " ".join(expected.lower().split())
    predicted = " ".join(predicted.lower().split())

    if len(expected) == 0:
        return 0.0 if len(predicted) == 0 else 1.0

    return distance(expected, predicted) / len(expected)

def accuracy(cer_score: float) -> float:
    return max(0.0, 1.0 - cer_score)