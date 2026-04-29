import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from explore import load_data, explore_data
from train import preprocess, train_model, evaluate_model, run_training


# ── Fixtures ──────────────────────────────────────────────
@pytest.fixture
def df():
    return load_data("IRIS.csv")


@pytest.fixture
def model_and_le(df):
    from sklearn.model_selection import train_test_split
    X, y, le = preprocess(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = train_model(X_train, y_train)
    return model, le, X_test, y_test


# ── Data Tests ────────────────────────────────────────────
def test_data_loads(df):
    assert df is not None
    assert len(df) == 150


def test_correct_columns(df):
    expected = {"sepal_length", "sepal_width", "petal_length", "petal_width", "species"}
    assert set(df.columns) == expected


def test_no_missing_values(df):
    assert df.isnull().sum().sum() == 0


def test_three_species(df):
    assert df["species"].nunique() == 3


def test_balanced_classes(df):
    counts = df["species"].value_counts()
    assert all(counts == 50)


# ── Preprocessing Tests ───────────────────────────────────
def test_preprocess_shapes(df):
    X, y, le = preprocess(df)
    assert X.shape == (150, 4)
    assert len(y) == 150


def test_label_encoding(df):
    X, y, le = preprocess(df)
    assert set(y) == {0, 1, 2}


# ── Model Tests ───────────────────────────────────────────
def test_model_type(model_and_le):
    model, le, X_test, y_test = model_and_le
    assert isinstance(model, RandomForestClassifier)


def test_model_accuracy(model_and_le):
    from sklearn.metrics import accuracy_score
    model, le, X_test, y_test = model_and_le
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    assert acc >= 0.90  # we expect at least 90% accuracy


def test_predictions_valid_classes(model_and_le):
    model, le, X_test, y_test = model_and_le
    y_pred = model.predict(X_test)
    assert set(y_pred).issubset({0, 1, 2})