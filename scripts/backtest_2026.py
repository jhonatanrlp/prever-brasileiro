"""Backtest rodada-a-rodada do Brasileirão 2026: para cada rodada já disputada,
reajusta o modelo de gols de Poisson só com o que era conhecido ANTES daquela
rodada (histórico 2003-2024 + 2025 + rodadas 1..r-1 de 2026) e mede o quão boas
eram as probabilidades geradas para a rodada r.

Isso responde "o que o modelo diria na rodada X, sabendo só o que sabíamos até
então" — só é possível porque `data/processed/matches_2025_2026.csv` tem data e
rodada reais (API da CBF), diferente da tentativa anterior via Wikipédia.

Uso:
    python scripts/backtest_2026.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation import evaluate_probabilistic  # noqa: E402
from src.poisson_goals import fit_poisson_goals_model  # noqa: E402

PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUTS_DIR = ROOT / "outputs"
COLS = ["season", "home_team_id", "away_team_id", "home_goals", "away_goals"]


def load_matches() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retorna (base_2003_2025, jogos_2026_disputados_com_rodada)."""
    historical = pd.read_csv(PROCESSED_DIR / "matches.csv")[COLS]

    recent = pd.read_csv(PROCESSED_DIR / "matches_2025_2026.csv")
    recent_played = recent[recent["played"]]

    base = pd.concat([historical, recent_played[recent_played["season"] == 2025][COLS]], ignore_index=True)
    season_2026 = recent_played[recent_played["season"] == 2026][["round", *COLS]].sort_values("round")
    return base, season_2026


def main() -> None:
    base, season_2026 = load_matches()
    rounds = sorted(season_2026["round"].unique())

    rows = []
    training_matches = base.copy()
    for round_number in rounds:
        round_matches = season_2026[season_2026["round"] == round_number]

        model = fit_poisson_goals_model(training_matches, current_season=2026)

        y_true = []
        y_proba = []
        for row in round_matches.itertuples(index=False):
            p_home, p_draw, p_away = model.outcome_probabilities(row.home_team_id, row.away_team_id)
            y_proba.append([p_home, p_draw, p_away])
            if row.home_goals > row.away_goals:
                y_true.append("H")
            elif row.home_goals < row.away_goals:
                y_true.append("A")
            else:
                y_true.append("D")

        metrics = evaluate_probabilistic(pd.Series(y_true).to_numpy(), pd.DataFrame(y_proba).to_numpy())
        rows.append({"round": round_number, "n_matches": len(round_matches), **metrics})
        print(f"Rodada {round_number}: {len(round_matches)} jogos — log_loss={metrics['log_loss']:.3f}")

        # Revela os resultados da rodada para o modelo da PRÓXIMA rodada.
        training_matches = pd.concat([training_matches, round_matches[COLS]], ignore_index=True)

    results = pd.DataFrame(rows)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUTS_DIR / "backtest_2026.csv", index=False)
    print(f"\nGravado outputs/backtest_2026.csv ({len(results)} rodadas)")
    print(f"Log loss médio: {results['log_loss'].mean():.3f} | acurácia média: {results['accuracy'].mean():.3f}")


if __name__ == "__main__":
    main()
