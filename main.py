"""Run the complete PhiUSIIL experiment from the project root."""

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from src.data_loader import fetch_data, prepare_data, split_data, split_summary
from src.evaluate import evaluate, robustness
from src.train import NGRAM, lexical_matrix, make_models


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-only", action="store_true", help="Cache the UCI dataset and exit")
    parser.add_argument("--uci-archive", type=Path, help="Use an official UCI ZIP already on disk")
    args = parser.parse_args()
    print("Loading UCI dataset 967", flush=True)
    raw, summary = fetch_data(archive=args.uci_archive)
    print(f"Loaded {len(raw):,} rows", flush=True)
    if args.download_only:
        return

    for folder in ["results", "figures"]:
        Path(folder).mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(Path(".mplconfig").resolve()))
    from src.visualize import plots

    frame, cleaning = prepare_data(raw)
    train, test = split_data(frame)
    summary.update(cleaning, split=split_summary(train, test))
    Path("results/dataset_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Domain holdout: {len(train):,} train and {len(test):,} test URLs", flush=True)
    print("Computing lexical features", flush=True)
    train_lexical, test_lexical = lexical_matrix(train["url"]), lexical_matrix(test["url"])
    models = make_models()
    rows, predictions = [], {}
    for name, model in models.items():
        print(f"Training {name}", flush=True)
        x_train, x_test = (train["url"], test["url"]) if name == NGRAM else (train_lexical, test_lexical)
        model.fit(x_train, train["label"])
        scores, predicted = evaluate(model, x_test, test["label"])
        rows.append({"model": name, **scores})
        predictions[name] = pd.Series(predicted, index=test.index)
    metrics = pd.DataFrame(rows)
    metrics.to_csv("results/metrics.csv", index=False)
    pairs = robustness(test, models, predictions)
    pairs.to_csv("results/attack_results.csv", index=False)
    plots(summary, metrics, pairs)
    print(metrics.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
