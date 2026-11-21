"""Model definitions and a shared feature matrix for lexical baselines."""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data_loader import SEED
from src.features import lexical_features


NGRAM = "Character n-gram"


def lexical_matrix(urls):
    return pd.DataFrame.from_records(map(lexical_features, urls))


def make_models():
    return {
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, random_state=SEED)
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, min_samples_leaf=3, n_jobs=4, random_state=SEED
        ),
        NGRAM: make_pipeline(
            TfidfVectorizer(analyzer="char", ngram_range=(3, 5), max_features=40000, min_df=3),
            LogisticRegression(solver="liblinear", max_iter=1000, random_state=SEED),
        ),
    }
