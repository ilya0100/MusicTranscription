import numpy as np

from sklearn.metrics import f1_score


def corrected_f1_score(
    y_pred: np.ndarray, y_true: np.ndarray, pred_size: np.ndarray
) -> float:
    scores = []
    for predict, true, size in zip(y_pred, y_true, pred_size):
        predict = np.argsort(predict)[::-1][:size]
        if size == 0:
            predict = np.array([128])
        predict.sort()

        size = max(size, true[true > -1].size)
        if size > len(predict):
            predict = np.pad(predict, (0, size - len(predict)), constant_values=-1)
        true = np.sort(true[:size])

        score = f1_score(true, predict, average="macro")
        scores.append(score)
    return sum(scores) / len(scores)


def accuracy(y_pred: np.ndarray, y_true: np.ndarray, pred_size: np.ndarray) -> float:
    scores = []
    for predict, true, size in zip(y_pred, y_true, pred_size):
        predict = np.argsort(predict)[::-1][:size]
        if size < 0:
            predict = np.array([128])

        size_max = max(size, true[true > -1].size)
        if size_max > len(predict):
            predict = np.pad(predict, (0, size_max - len(predict)), constant_values=-1)
        true = true[:size_max]
        if predict.size != true.size:
            raise RuntimeError(
                f"Predict size ({predict.size}) != true size ({true.size})"
            )

        score = np.intersect1d(predict, true).size / true.size
        scores.append(score)
    return sum(scores) / len(scores)
