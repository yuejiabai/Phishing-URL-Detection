"""Scores use the official UCI phishing label as the positive class."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)

from src.attacks import METHODS, perturb
from src.data_loader import LEGITIMATE, PHISHING
from src.train import NGRAM, lexical_matrix


def evaluate(model, inputs, labels):
    predictions = model.predict(inputs)
    index = list(model.classes_).index(PHISHING)
    probabilities = model.predict_proba(inputs)[:, index]
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[LEGITIMATE, PHISHING]).ravel()
    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, pos_label=PHISHING, zero_division=0),
        "recall": recall_score(labels, predictions, pos_label=PHISHING, zero_division=0),
        "f1": f1_score(labels, predictions, pos_label=PHISHING, zero_division=0),
        "roc_auc": roc_auc_score(np.asarray(labels) == PHISHING, probabilities),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }, predictions


def robustness(test, models, clean_predictions):
    phishing = test.loc[test["label"].eq(PHISHING), "url"]
    records = []
    for method in METHODS:
        print(f"Evaluating {method}", flush=True)
        edits = phishing.map(lambda url: perturb(url, method))
        pairs = pd.DataFrame(edits.to_list(), index=edits.index, columns=["perturbed_url", "status"])
        pairs.insert(0, "original_url", phishing)
        pairs.index.name = "clean_row_id"
        slug = method.lower().replace(" ", "_")
        pairs.to_csv(f"data/pairs_{slug}.csv")
        valid = pairs.loc[pairs["status"].eq("valid"), "perturbed_url"]
        lexical = lexical_matrix(valid) if len(valid) else None
        for name, model in models.items():
            before = after = np.nan
            if len(valid):
                before = clean_predictions[name].loc[valid.index].eq(PHISHING).mean()
                after = (model.predict(valid if name == NGRAM else lexical) == PHISHING).mean()
            records.append({
                "model": name,
                "perturbation": method,
                "phishing_test_rows": len(phishing),
                "valid_rows": len(valid),
                "failed_rows": len(phishing) - len(valid),
                "no_eligible_edit": pairs["status"].eq("no_eligible_edit").sum(),
                "invalid_original": pairs["status"].eq("invalid_original").sum(),
                "invalid_result": pairs["status"].eq("invalid_result").sum(),
                "original_recall": before,
                "perturbed_recall": after,
                "recall_change_pp": 100 * (after - before),
            })
    return pd.DataFrame(records)
