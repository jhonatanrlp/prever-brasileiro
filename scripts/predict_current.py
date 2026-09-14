"""Previsão do restante do Brasileirão 2026 a partir do estado atual.

Pipeline:
    1. Carrega o histórico 2003-2024 (data/processed/matches.csv) + 2025/2026 já
       disputados (data/processed/matches_2025_2026.csv, calendário oficial da
       CBF com data e rodada reais — ver src/cbf_calendario.py)
       para ajustar o modelo de gols de Poisson, com peso maior para temporadas
       recentes.
    2. Carrega a classificação atual de 2026, CALCULADA a partir das partidas de
       2026 já disputadas (data/processed/current_standings_2026.csv, gerado por
       scripts/build_recent_seasons.py) — não copiada de uma fonte externa.
    3. Gera P(vitória mandante)/P(empate)/P(vitória visitante) para cada um dos
       jogos restantes.
    4. Roda Monte Carlo (configs/config.yaml: simulation.n_seasons) a partir da
       tabela atual real + os jogos restantes.
    5. Grava outputs/current_prediction.csv e outputs/match_predictions_2026.csv.

Uso:
    python scripts/download_data.py
    python scripts/build_dataset.py
    python scripts/fetch_cbf_calendar.py
    python scripts/build_recent_seasons.py
    python scripts/predict_current.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.poisson_goals import fit_poisson_goals_model  # noqa: E402
from src.monte_carlo import simulate_remaining_season  # noqa: E402
from src.standings import CompetitionRules  # noqa: E402

PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUTS_DIR = ROOT / "outputs"
CONFIG_PATH = ROOT / "configs" / "config.yaml"
CURRENT_SEASON = 2026


def load_training_matches() -> pd.DataFrame:
    cols = ["season", "home_team_id", "away_team_id", "home_goals", "away_goals"]
    historical = pd.read_csv(PROCESSED_DIR / "matches.csv")[cols]

    recent = pd.read_csv(PROCESSED_DIR / "matches_2025_2026.csv")
    recent = recent[recent["played"]][cols]

    return pd.concat([historical, recent], ignore_index=True)


def main() -> None:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    rules = CompetitionRules.from_config(CONFIG_PATH)
    n_simulations = config["simulation"]["n_seasons"]
    random_seed = config["random_seed"]

    training_matches = load_training_matches()
    goals_model = fit_poisson_goals_model(training_matches, current_season=CURRENT_SEASON)

    current_table = pd.read_csv(PROCESSED_DIR / "current_standings_2026.csv")
    remaining_fixtures = pd.read_csv(PROCESSED_DIR / "remaining_fixtures_2026.csv")

    print(f"Times na tabela atual: {len(current_table)}; jogos restantes: {len(remaining_fixtures)}")
    print(f"Rodando {n_simulations} simulações de Monte Carlo...")

    prediction = simulate_remaining_season(
        current_table=current_table,
        remaining_fixtures=remaining_fixtures,
        goals_model=goals_model,
        rules=rules,
        n_simulations=n_simulations,
        random_seed=random_seed,
    )

    team_names = pd.read_csv(ROOT / "data" / "external" / "team_mapping.csv").set_index("team_id")[
        "canonical_name"
    ]
    prediction.insert(1, "team", prediction["team_id"].map(team_names))

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    prediction.to_csv(OUTPUTS_DIR / "current_prediction.csv", index=False)
    print(f"Gravado outputs/current_prediction.csv ({len(prediction)} times)")

    match_predictions = []
    for row in remaining_fixtures.itertuples(index=False):
        p_home, p_draw, p_away = goals_model.outcome_probabilities(row.home_team_id, row.away_team_id)
        match_predictions.append(
            {
                "round": row.round,
                "date": row.date if pd.notna(row.date) else "A definir",
                "home_team": team_names.get(row.home_team_id, row.home_team_id),
                "away_team": team_names.get(row.away_team_id, row.away_team_id),
                "home_win_probability": p_home,
                "draw_probability": p_draw,
                "away_win_probability": p_away,
            }
        )
    match_predictions_df = pd.DataFrame(match_predictions).sort_values("round")
    match_predictions_df.to_csv(OUTPUTS_DIR / "match_predictions_2026.csv", index=False)
    print(f"Gravado outputs/match_predictions_2026.csv ({len(match_predictions_df)} jogos)")


if __name__ == "__main__":
    main()
