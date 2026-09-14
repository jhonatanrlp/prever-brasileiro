"""Compara baselines simples contra modelos de ML para prever V/E/D, com validação
temporal (walk-forward por temporada, nunca split aleatório — ver README.md).

Modelos:
    - home_advantage: baseline que ignora força de time, usa só a frequência
      histórica de H/D/A observada no treino.
    - elo: baseline que converte a probabilidade binária de vitória do mandante
      (Elo) em 3 classes, com uma taxa de empate constante estimada no treino.
    - logistic_regression / random_forest / gradient_boosting: treinados sobre as
      features de `build_pre_match_features` (Elo, forma, força do adversário).

Uso:
    python scripts/compare_models.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation import RESULT_CLASSES, evaluate_probabilistic, expanding_window_splits  # noqa: E402
from src.temporal import build_pre_match_features  # noqa: E402

PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUTS_DIR = ROOT / "outputs"
MIN_TRAIN_SEASONS = 15

FEATURE_COLS = [
    "elo_diff",
    "home_win_prob_elo",
    "home_strength_of_schedule",
    "away_strength_of_schedule",
    "home_points_per_game",
    "away_points_per_game",
    "home_goal_diff_per_game",
    "away_goal_diff_per_game",
    "attack_vs_defense_home",
    "attack_vs_defense_away",
    "form_diff_last_5",
    "home_form_points_last_5",
    "away_form_points_last_5",
]


def load_features() -> pd.DataFrame:
    historical = pd.read_csv(PROCESSED_DIR / "matches.csv", parse_dates=["date"])
    recent = pd.read_csv(PROCESSED_DIR / "matches_2025_2026.csv", parse_dates=["date"])
    recent = recent[recent["played"]].rename(columns={"cbf_id": "match_id"})

    cols = ["match_id", "season", "date", "home_team_id", "away_team_id", "home_goals", "away_goals"]
    matches = pd.concat([historical[cols], recent[cols]], ignore_index=True)
    return build_pre_match_features(matches)


def home_advantage_probs(train: pd.DataFrame, n_test: int) -> np.ndarray:
    freq = train["target_result"].value_counts(normalize=True)
    row = [freq.get(cls, 0.0) for cls in RESULT_CLASSES]
    return np.tile(row, (n_test, 1))


def elo_baseline_probs(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    draw_rate = float((train["target_result"] == "D").mean())
    home_win_elo = test["home_win_prob_elo"].to_numpy()
    remaining = 1.0 - draw_rate
    home = remaining * home_win_elo
    away = remaining * (1.0 - home_win_elo)
    draw = np.full_like(home, draw_rate)
    return np.column_stack([home, draw, away])


def sklearn_probs(model, train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    medians = train[FEATURE_COLS].median()
    x_train = train[FEATURE_COLS].fillna(medians)
    x_test = test[FEATURE_COLS].fillna(medians)
    model.fit(x_train, train["target_result"])
    proba = model.predict_proba(x_test)
    # Reordena as colunas para a ordem canônica H/D/A (sklearn ordena por rótulo alfabético).
    return proba[:, [list(model.classes_).index(c) for c in RESULT_CLASSES]]


def main() -> None:
    features = load_features()

    models = {
        "logistic_regression": lambda tr, te: sklearn_probs(
            make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)), tr, te
        ),
        "random_forest": lambda tr, te: sklearn_probs(
            RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42), tr, te
        ),
        "gradient_boosting": lambda tr, te: sklearn_probs(
            GradientBoostingClassifier(random_state=42), tr, te
        ),
    }

    rows = []
    for train, test, season in expanding_window_splits(features, min_train_seasons=MIN_TRAIN_SEASONS):
        y_true = test["target_result"].to_numpy()

        predictions = {
            "home_advantage": home_advantage_probs(train, len(test)),
            "elo": elo_baseline_probs(train, test),
        }
        for name, fit_predict in models.items():
            predictions[name] = fit_predict(train, test)

        for model_name, y_proba in predictions.items():
            metrics = evaluate_probabilistic(y_true, y_proba)
            rows.append({"season": season, "model": model_name, "n_matches": len(test), **metrics})

        print(f"Temporada {season}: {len(test)} partidas avaliadas")

    results = pd.DataFrame(rows)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUTS_DIR / "model_comparison.csv", index=False)

    summary = (
        results.groupby("model")[["log_loss", "brier_score", "accuracy"]]
        .mean()
        .sort_values("log_loss")
    )
    print("\nMédia por modelo (todas as temporadas de teste, menor log_loss é melhor):")
    print(summary.to_string())


if __name__ == "__main__":
    main()
