import re
import math
from urllib.parse import urlparse
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


SPECIAL_CHARS = set("!@#$%^&*()[]{};:,<>?\\|`~=_+")

def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    probs = [float(s.count(c)) / len(s) for c in set(s)]
    return -sum(p * math.log2(p) for p in probs)


def count_special_chars(s: str) -> int:
    return sum(1 for c in s if c in SPECIAL_CHARS)


def tokenize_url(url: str):
    return re.split(r"[./\-_\?=&:]+", url)

#--------------------------------------------------------------------------------------------------------

def extract_features_from_url(url: str) -> dict:
    parsed = urlparse(url)
    hostname = parsed.hostname or ""
    path = parsed.path or ""
    query = parsed.query or ""

    tokens = [t for t in tokenize_url(url) if t]

    # Lengths
    url_len = len(url)
    host_len = len(hostname)
    path_len = len(path)
    query_len = len(query)

    # Structure
    num_dots = url.count(".")
    num_subdomains = hostname.count(".") - 1 if hostname.count(".") > 0 else 0

    # Character counts
    num_digits = sum(c.isdigit() for c in url)
    num_letters = sum(c.isalpha() for c in url)
    num_special = count_special_chars(url)

    digit_ratio = num_digits / url_len if url_len else 0.0
    letter_ratio = num_letters / url_len if url_len else 0.0
    special_ratio = num_special / url_len if url_len else 0.0

    # Tokens
    num_tokens = len(tokens)
    avg_token_len = np.mean([len(t) for t in tokens]) if tokens else 0.0
    max_token_len = max([len(t) for t in tokens]) if tokens else 0

    # Statistical
    entropy = shannon_entropy(url)

    features = {
        # Lengths
        "url_len": url_len,
        "host_len": host_len,
        "path_len": path_len,
        "query_len": query_len,

        # Structure
        "num_dots": num_dots,
        "num_subdomains": num_subdomains,

        # Characters
        "num_digits": num_digits,
        "num_letters": num_letters,
        "num_special": num_special,
        "digit_ratio": digit_ratio,
        "letter_ratio": letter_ratio,
        "special_ratio": special_ratio,

        # Tokens
        "num_tokens": num_tokens,
        "avg_token_len": avg_token_len,
        "max_token_len": max_token_len,

        # Statistical
        "entropy": entropy,
    }

    return features

def extract_features(urls, return_df=True):
    feature_dicts = [extract_features_from_url(u) for u in urls]

    if return_df:
        return pd.DataFrame(feature_dicts)
    else:
        return np.array([list(d.values()) for d in feature_dicts])

#--------------------------------------------------------------------------------------------------------

class LexicalFeatures(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.columns_ = None

    def fit(self, X, y=None):
        df = extract_features(X, return_df=True)
        self.columns_ = df.columns
        return self

    def transform(self, X):
        df = extract_features(X, return_df=True)
        df = df[self.columns_]
        return df.values

#--------------------------------------------------------------------------------------------------------
def get_lexical_features(minmax=False):
    return Pipeline([
        ("features", LexicalFeatures()),
        ("scaler", MinMaxScaler() if minmax else StandardScaler())
    ])

def get_tfidf_features():
    return TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=2,
        max_features=50000
    )

def get_full_features(minmax=False):
    lexical_pipeline = get_lexical_features(minmax)
    tfidf = get_tfidf_features()
    features = FeatureUnion([
        ("tfidf", tfidf),
        ("lexical", lexical_pipeline)
    ])
    return features