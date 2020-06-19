import numpy as np
from sklearn import metrics


def evaluate(predicted, pred_labels):
    """
    Evaluiere einen classifier auf Evaluierungsdaten.
    """
    print(f"Confusion matrix:\n{metrics.confusion_matrix(pred_labels, predicted)}")
    print(f"{metrics.classification_report(pred_labels, predicted)}")
    return np.mean(predicted == pred_labels)
