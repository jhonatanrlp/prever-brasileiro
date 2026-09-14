"""Compara baselines simples contra modelos de ML para prever V/E/D, com validação
temporal (walk-forward por temporada, nunca split aleatório — ver README.md).

Modelos:
    - home_advantage: baseline que ignora força de time, usa só a frequência
      histórica de H/D/A observada no treino.
    - elo: baseline que converte a probabilidade binária de vitória do mandante
      (Elo) em 3 classes, com uma taxa de empate constante estimada no treino.
    - poisson_dixon_coles: o modelo de produção original (`fit_poisson_goals_model`)
      — ataque/defesa por média ponderada, rho por perfil de máxima verossimilhança.
    - poisson_dixon_coles_mle: mesma família de modelo, mas ataque, defesa, médias-
      base e rho ajustados TODOS de uma vez por máxima verossimilhança conjunta
      (`fit_poisson_goals_model_mle`) — ver comparação empírica dos dois no README.
    - logistic_regression / random_forest / gradient_boosting: treinados sobre as
      features de `build_pre_match_features` (Elo, forma, força do adversário,
      chutes/chutes a gol quando disponíveis).

Ambos os modelos de gols são refeitos a cada temporada de teste só com dados
anteriores a ela (walk-forward de verdade, sem vazamento).

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
from src.poisson_goals import fit_poisson_goals_model, fit_poisson_goals_model_mle  # noqa: E402
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
    "shots_diff",
    "shots_on_target_diff",
]


def load_features() -> pd.DataFrame:
    historical = pd.read_csv(PROCESSED_DIR / "matches.csv", parse_dates=["date"])
    recent = pd.read_csv(PROCESSED_DIR / "matches_2025_2026.csv", parse_dates=["date"])
    recent = recent[recent["played"]].rename(columns={"cbf_id": "match_id"})
    team_stats = pd.read_csv(PROCESSED_DIR / "team_match_stats.csv")

    cols = ["match_id", "season", "date", "home_team_id", "away_team_id", "home_goals", "away_goals"]
    matches = pd.concat([historical[cols], recent[cols]], ignore_index=True)
    return build_pre_match_features(matches, team_stats=team_stats)


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


def poisson_probs(
    train: pd.DataFrame, test: pd.DataFrame, current_season: int, use_mle: bool, diagnostics: list[dict]
) -> np.ndarray:
    train_matches = train.rename(
        columns={"target_home_goals": "home_goals", "target_away_goals": "away_goals"}
    )
    fit_fn = fit_poisson_goals_model_mle if use_mle else fit_poisson_goals_model
    model = fit_fn(train_matches, current_season=current_season)

    if use_mle:
        if not model.converged:
            print(f"  [aviso] MLE conjunta não convergiu na temporada {current_season} ({model.n_iterations} iterações)")
        diagnostics.append(
            {
                "season": current_season,
                "rho": model.rho,
                "log_likelihood": model.log_likelihood,
                "aic": model.aic,
                "bic": model.bic,
                "converged": model.converged,
                "n_iterations": model.n_iterations,
            }
        )

    return np.array(
        [
            model.outcome_probabilities(h, a)
            for h, a in zip(test["home_team_id"], test["away_team_id"])
        ]
    )


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
    mle_diagnostics: list[dict] = []
    for train, test, season in expanding_window_splits(features, min_train_seasons=MIN_TRAIN_SEASONS):
        y_true = test["target_result"].to_numpy()

        predictions = {
            "home_advantage": home_advantage_probs(train, len(test)),
            "elo": elo_baseline_probs(train, test),
            "poisson_dixon_coles": poisson_probs(train, test, current_season=season, use_mle=False, diagnostics=[]),
            "poisson_dixon_coles_mle": poisson_probs(
                train, test, current_season=season, use_mle=True, diagnostics=mle_diagnostics
            ),
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

    diagnostics_df = pd.DataFrame(mle_diagnostics)
    diagnostics_df.to_csv(OUTPUTS_DIR / "dixon_coles_mle_diagnostics.csv", index=False)
    print("\nDiagnóstico do ajuste MLE conjunta por temporada de treino:")
    print(diagnostics_df.to_string(index=False))

    metric_cols = ["log_loss", "brier_score", "rps", "accuracy"]
    mean_summary = results.groupby("model")[metric_cols].mean().sort_values("log_loss")
    std_summary = results.groupby("model")[metric_cols].std().loc[mean_summary.index]

    print("\nMédia por modelo (todas as temporadas de teste, menor é melhor exceto acurácia):")
    print(mean_summary.to_string())
    print("\nDesvio-padrão entre temporadas (estabilidade — menor é mais consistente):")
    print(std_summary.to_string())

    print("\nComparação direta: modelo de produção vs. MLE conjunta (mesmo backtest, mesmas partidas):")
    pivot = results[results["model"].isin(["poisson_dixon_coles", "poisson_dixon_coles_mle"])]
    comparison = pivot.pivot(index="season", columns="model", values=metric_cols)
    print(comparison.to_string())


if __name__ == "__main__":
    main()
