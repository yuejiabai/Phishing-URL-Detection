"""Save figures directly from the experiment outputs."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from src.train import NGRAM


def save(fig, name):
    fig.savefig(f"figures/{name}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plots(summary, metrics, robustness):
    sns.set_theme(style="whitegrid", palette="colorblind")
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = summary["clean_class_counts"]
    bars = ax.bar([label.title() for label in labels], list(labels.values()))
    ax.bar_label(bars, fmt="%d", padding=4)
    ax.margins(y=0.13)
    ax.set(title="Dataset Class Distribution", ylabel="URLs")
    save(fig, "class_distribution")

    fig, ax = plt.subplots(figsize=(9, 4))
    scores = metrics.set_index("model").drop(index=NGRAM)
    scores[["accuracy", "precision", "recall", "f1", "roc_auc"]].T.plot.bar(ax=ax, rot=0)
    ax.set(title="Baseline Model Performance Comparison", ylabel="Score", ylim=(0, 1.08))
    ax.set_xticks(range(5), ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"])
    ax.legend(loc="lower center", ncol=2)
    save(fig, "baseline_comparison")

    base = robustness.loc[robustness["model"].ne(NGRAM)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True, layout="constrained")
    for ax, (name, rows) in zip(axes, base.groupby("model", sort=False)):
        rows.set_index("perturbation")[["original_recall", "perturbed_recall"]].rename(
            columns={"original_recall": "Original", "perturbed_recall": "Edited"}
        ).plot.bar(ax=ax, rot=12)
        ax.set(title=name, xlabel="", ylabel="Phishing recall", ylim=(0, 1.05))
        ax.legend(loc="lower center")
    fig.suptitle("Recall Before and After URL Perturbations")
    save(fig, "recall_before_after")

    fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
    robustness.pivot(index="perturbation", columns="model", values="recall_change_pp").plot.bar(
        ax=ax, rot=10
    )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set(title="Character N-gram Robustness Comparison", xlabel="", ylabel="Recall change (pp)")
    ax.legend(title="", loc="best")
    save(fig, "ngram_robustness")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
    for ax, row in zip(axes, metrics.itertuples()):
        sns.heatmap([[row.tn, row.fp], [row.fn, row.tp]], ax=ax, annot=True, fmt="d",
                    cbar=False, cmap="Blues", xticklabels=["Legitimate", "Phishing"],
                    yticklabels=["Legitimate", "Phishing"])
        ax.set(title=row.model, xlabel="Predicted class", ylabel="True class")
    fig.suptitle("Confusion Matrices on the Clean Holdout")
    save(fig, "confusion_matrix")
