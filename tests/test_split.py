import pandas as pd

from src.data_loader import prepare_data, split_data, split_summary
from src.features import registered_group


def test_domain_holdout_is_deterministic_and_disjoint():
    rows = [
        (f"https://{sub}.example{i}.org/", i % 2)
        for i in range(20)
        for sub in ("a", "b")
    ]
    rows.extend([("https://a.unknown-test-suffix/", 0), ("https://another.local/", 1)])
    frame = pd.DataFrame(rows, columns=["url", "label"])
    frame["group"] = frame["url"].map(registered_group)
    assert frame["group"].iloc[-2:].tolist() == ["unresolved", "unresolved"]
    train, test = split_data(frame)
    assert split_summary(train, test)["group_overlap"] == 0
    assert len(train) + len(test) == len(frame)
    second_train, second_test = split_data(frame)
    assert train.index.equals(second_train.index)
    assert test.index.equals(second_test.index)


def test_cleaning_counts_conflicts_and_duplicates_separately():
    frame = pd.DataFrame({
        "url": ["https://example.org", "https://example.org", "broken", None,
                "https://example.com", "https://example.com"],
        "label": [0, 0, 1, 1, 0, 1],
    })
    clean, summary = prepare_data(frame)
    assert len(clean) == 1
    assert summary["missing_url_or_label_removed"] == 1
    assert summary["invalid_url_removed"] == 1
    assert summary["duplicate_url_rows_removed"] == 1
    assert summary["conflicting_label_rows_removed"] == 2
