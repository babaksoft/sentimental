import re

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from .config import config

USER_HANDLE_RX = r"@[\S]+"
HASH_TAG_RX = r"#([\S]+)"
HTML_TAG_RX = r"<[a-zA-Z0-9\s_/='\"]+>"
HTML_OBJECT_RX = r"&?[a-z]+;"
UNICODE_OBJ_RX = r"#x[0-9a-fA-F]+;"
EXPAND_TEXT_RX = r"([a-zA-Z]\s){3,}"
WEB_URL_RX = r"((www\.[\S]+)|(https?://[\S]+))"
PUNCT_CHARS_RX = r"([\w]+)[\.?!,:;'\"]+"


def get_values(x):
    if isinstance(x, (pd.DataFrame, pd.Series)):
        values = x.values.flatten()
    else:
        values = x.flatten()
    return values


def replace_expanded(match: re.Match) -> str:
    expanded = match.string[match.span()[0]:match.span()[1]]
    return expanded.replace(" ", "")


def clean_text(text: str) -> str:
    cleaned = text

    # Convert to lowercase
    cleaned = cleaned.lower()

    # Remove Unicode strings
    cleaned = re.sub(UNICODE_OBJ_RX, "", cleaned)

    # Remove URLs, HTML tags and HTML objects
    cleaned = re.sub(WEB_URL_RX, "", cleaned)
    cleaned = re.sub(HTML_TAG_RX, "", cleaned)
    cleaned = re.sub(HTML_OBJECT_RX, "", cleaned)

    # Remove user handles (@user)
    cleaned = re.sub(USER_HANDLE_RX, "", cleaned)

    # Remove hashtag prefix from hashtags
    cleaned = re.sub(HASH_TAG_RX, r"\1", cleaned)

    # Replace expanded text (e.g. t w i t t e r) with normal form
    cleaned = re.sub(EXPAND_TEXT_RX, replace_expanded, cleaned)

    # Remove trailing punctuation character(s) (except emoticons)
    cleaned = re.sub(PUNCT_CHARS_RX, r"\1", cleaned)

    # Replace extra whitespace and new-line with a single space
    cleaned = re.sub(r"\s+", " ", cleaned)

    # Remove tokens with non-ASCII characters (including emojis)
    cleaned = " ".join([word for word in cleaned.split(" ") if word.isascii()])

    return cleaned


def statement_transform(x):
    values = get_values(x)
    return np.array([
        clean_text(text) for text in values
    ]).reshape(-1, 1)


def build_pipeline() -> ColumnTransformer:
    text_pipeline = Pipeline([
        ("clean", FunctionTransformer(
            func=statement_transform,
            feature_names_out="one-to-one"
        )),
        ("vectorize", TfidfVectorizer(
            strip_accents=None,
            lowercase=False,
            preprocessor=None
        ))
    ])

    return ColumnTransformer([
        ("text", text_pipeline, config.FEATURE)
    ])
