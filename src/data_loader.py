"""Fetch UCI data and construct a domain-disjoint holdout."""

import hashlib
import json
from pathlib import Path
from urllib.parse import urlsplit
from zipfile import ZipFile

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from ucimlrepo import fetch_ucirepo

from src.features import parse_url, registered_group


PHISHING = 0
LEGITIMATE = 1
SEED = 42
ARCHIVE_URL = "https://archive.ics.uci.edu/static/public/967/phiusiil%2Bphishing%2Burl%2Bdataset.zip"


def counts(labels):
    return {
        "phishing": int(labels.eq(PHISHING).sum()),
        "legitimate": int(labels.eq(LEGITIMATE).sum()),
    }


def fetch_data(directory=Path("data"), archive=None):
    directory.mkdir(parents=True, exist_ok=True)
    csv_path = directory / "phiusiil_url_labels.csv"
    metadata_path = directory / "source_summary.json"
    if archive is None and csv_path.exists() and metadata_path.exists():
        return pd.read_csv(csv_path), json.loads(metadata_path.read_text(encoding="utf-8"))

    if archive is None:
        dataset = fetch_ucirepo(id=967)
        original = dataset.data.original
        name = dataset.metadata.name
        source_url = dataset.metadata.data_url
        loader = "ucimlrepo"
    else:
        with ZipFile(archive) as package:
            csv_files = [name for name in package.namelist() if name.lower().endswith(".csv")]
            if len(csv_files) != 1:
                raise ValueError("Expected one CSV inside the official UCI archive")
            with package.open(csv_files[0]) as csv_file:
                original = pd.read_csv(csv_file, low_memory=False)
        name = "PhiUSIIL Phishing URL (Website)"
        source_url = ARCHIVE_URL
        loader = "official UCI archive"
    frame = original[["URL", "label"]].rename(columns={"URL": "url"})
    observed = set(frame["label"].dropna().unique())
    if observed != {PHISHING, LEGITIMATE}:
        raise ValueError(f"Unexpected UCI labels: {observed}")
    summary = {
        "dataset_id": 967,
        "dataset_name": name,
        "source_url": source_url,
        "loader": loader,
        "raw_rows": len(original),
        "raw_columns": original.columns.tolist(),
        "raw_missing_by_column": original.isna().sum().to_dict(),
        "raw_class_counts": counts(frame["label"]),
    }
    frame.to_csv(csv_path, index=False)
    metadata_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return frame, summary


def prepare_data(frame):
    available = frame.dropna(subset=["url", "label"])
    urls = available["url"].str.strip()
    valid = available.assign(url=urls).loc[urls.map(lambda url: parse_url(url) is not None)]
    # Remove all rows for URLs with contradictory labels before deduplication.
    conflicts = valid.groupby("url")["label"].transform("nunique").gt(1)
    consistent = valid.loc[~conflicts]
    clean = consistent.drop_duplicates("url").reset_index(drop=True)
    clean["label"] = clean["label"].astype(int)
    clean["group"] = clean["url"].map(registered_group)
    return clean, {
        "missing_url_or_label_removed": len(frame) - len(available),
        "invalid_url_removed": len(available) - len(valid),
        "conflicting_label_rows_removed": int(conflicts.sum()),
        "duplicate_url_rows_removed": len(consistent) - len(clean),
        "clean_rows": len(clean),
        "clean_class_counts": counts(clean["label"]),
        "unresolved_group_rows": int(clean["group"].eq("unresolved").sum()),
        "clean_url_label_sha256": hashlib.sha256(
            clean[["url", "label"]].to_csv(index=False, lineterminator="\n").encode()
        ).hexdigest(),
    }


def split_data(frame):
    # Rank seeded candidate holdouts by row fraction and class prevalence only.
    splitter = GroupShuffleSplit(n_splits=32, test_size=0.2, random_state=SEED)
    positive = frame["label"].to_numpy() == PHISHING
    prevalence = positive.mean()
    candidates = []
    for train, test in splitter.split(frame, groups=frame["group"]):
        if not 0 < positive[train].mean() < 1 or not 0 < positive[test].mean() < 1:
            continue
        score = abs(len(test) / len(frame) - 0.2) + abs(positive[test].mean() - prevalence)
        candidates.append((score, train, test))
    if not candidates:
        raise ValueError("Unable to split by domain with both classes in each partition")
    _, train, test = min(candidates, key=lambda candidate: candidate[0])
    return frame.iloc[train].copy(), frame.iloc[test].copy()


def split_summary(train, test):
    overlap = set(train["group"]) & set(test["group"])
    if overlap:
        raise RuntimeError("Training and test groups overlap")
    result = {"seed": SEED, "group_overlap": len(overlap)}
    for name, frame in [("train", train), ("test", test)]:
        non_root = frame["url"].map(lambda url: urlsplit(url).path not in {"", "/"})
        result[name] = {
            "rows": len(frame),
            "groups": int(frame["group"].nunique()),
            "class_counts": counts(frame["label"]),
            "phishing_fraction": float(frame["label"].eq(PHISHING).mean()),
            "non_root_path_counts": counts(frame.loc[non_root, "label"]),
        }
    return result
