import numpy as np

from src.evaluate import evaluate


def test_phishing_zero_is_the_positive_class():
    class FixedModel:
        classes_ = [0, 1]

        def predict_proba(self, inputs):
            return np.array([[0.9, 0.1], [0.4, 0.6], [0.3, 0.7], [0.2, 0.8]])

        def predict(self, inputs):
            return self.predict_proba(inputs).argmax(axis=1)

    metrics, _ = evaluate(FixedModel(), [1, 2, 3, 4], [0, 0, 1, 1])
    assert metrics["recall"] == 0.5
    assert metrics["precision"] == 1
    assert metrics["roc_auc"] == 1
    assert (metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"]) == (2, 0, 1, 1)
