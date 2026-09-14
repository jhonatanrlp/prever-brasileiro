import numpy as np
import pandas as pd
import pytest

from src.evaluation import evaluate_probabilistic, expanding_window_splits


def _toy_features(seasons: list[int]) -> pd.DataFrame:
    return pd.DataFrame({"season": seasons, "value": range(len(seasons))})


def test_expanding_window_splits_grows_train_and_advances_one_season():
    features = _toy_features([2020, 2020, 2021, 2022, 2023, 2024, 2025, 2026])
    splits = list(expanding_window_splits(features, min_train_seasons=2))

    assert [test_season for _, _, test_season in splits] == [2022, 2023, 2024, 2025, 2026]
    train0, test0, _ = splits[0]
    assert set(train0["season"]) == {2020, 2021}
    assert set(test0["season"]) == {2022}


def test_expanding_window_train_never_contains_test_season_or_later():
    features = _toy_features([2020, 2021, 2022, 2023, 2024])
    for train, test, test_season in expanding_window_splits(features, min_train_seasons=1):
        assert train["season"].max() < test_season
        assert (test["season"] == test_season).all()


def test_expanding_window_raises_when_not_enough_seasons():
    features = _toy_features([2020, 2021])
    with pytest.raises(ValueError):
        list(expanding_window_splits(features, min_train_seasons=5))


def test_evaluate_probabilistic_perfect_predictions_have_zero_loss():
    y_true = np.array(["H", "D", "A"])
    y_proba = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    metrics = evaluate_probabilistic(y_true, y_proba)
    assert metrics["accuracy"] == 1.0
    assert metrics["brier_score"] == pytest.approx(0.0, abs=1e-9)
    assert metrics["log_loss"] < 1e-6


def test_evaluate_probabilistic_uniform_guess_has_positive_loss():
    y_true = np.array(["H", "D", "A"])
    y_proba = np.full((3, 3), 1 / 3)
    metrics = evaluate_probabilistic(y_true, y_proba)
    assert metrics["log_loss"] > 1.0
    assert metrics["brier_score"] > 0.0
