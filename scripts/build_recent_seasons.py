"""Constrói o modelo dimensional de 2025/2026 a partir do calendário oficial da CBF
(data/raw/cbf/season_2025.json, season_2026.json — ver src/cbf_calendario.py), com
data e rodada reais por partida. Isso permite concatenar essas partidas ao histórico
2003-2024 e alimentar `build_pre_match_features` normalmente (ver docs/METHODOLOGY.md).

Gera em data/processed/:
    matches_2025_2026.csv        todas as partidas de 2025/2026, mesmo schema de matches.csv
    remaining_fixtures_2026.csv  partidas de 2026 ainda não disputadas
    current_standings_2026.csv   classificação atual, calculada a partir das partidas
                                  de 2026 já disputadas (não copiada de fonte externa)

Uso:
    python scripts/fetch_cbf_calendar.py
    python scripts/build_recent_seasons.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cbf_calendario import parse_matches  # noqa: E402
from src.normalize_teams import TeamNormalizer  # noqa: E402
from src.standings import CompetitionRules, build_standings  # noqa: E402

RAW_DIR = ROOT / "data" / "raw" / "cbf"
PROCESSED_DIR = ROOT / "data" / "processed"


def _result(home_goals: float, away_goals: float) -> str | None:
    if pd.isna(home_goals) or pd.isna(away_goals):
        return None
    if home_goals > away_goals:
        return "H"
    if home_goals < away_goals:
        return "A"
    return "D"


def load_season(year: int, normalizer: TeamNormalizer) -> pd.DataFrame:
    raw = json.loads((RAW_DIR / f"season_{year}.json").read_text(encoding="utf-8"))
    df = parse_matches(raw)
    df["season"] = year
    df["home_team_id"] = normalizer.normalize_series(df["home"])
    df["away_team_id"] = normalizer.normalize_series(df["away"])
    df["result"] = [
        _result(h, a) for h, a in zip(df["home_goals"], df["away_goals"])
    ]
    df = df.drop(columns=["date"]).rename(columns={"date_parsed": "date"})
    return df[
        [
            "cbf_id",
            "season",
            "round",
            "date",
            "home_team_id",
            "away_team_id",
            "home_goals",
            "away_goals",
            "result",
            "played",
            "venue",
        ]
    ]


def main() -> None:
    normalizer = TeamNormalizer()
    matches = pd.concat(
        [load_season(2025, normalizer), load_season(2026, normalizer)], ignore_index=True
    )
    matches = matches.sort_values(["season", "round", "date"]).reset_index(drop=True)
    matches.to_csv(PROCESSED_DIR / "matches_2025_2026.csv", index=False)
    print(f"matches_2025_2026: {matches.shape}, jogos disputados: {int(matches['played'].sum())}")

    remaining_2026 = matches[(matches["season"] == 2026) & (~matches["played"])][
        ["round", "date", "home_team_id", "away_team_id"]
    ]
    remaining_2026.to_csv(PROCESSED_DIR / "remaining_fixtures_2026.csv", index=False)
    print(f"remaining_fixtures_2026: {remaining_2026.shape}")

    played_2026 = matches[(matches["season"] == 2026) & (matches["played"])]
    rules = CompetitionRules.from_config()
    current_table = build_standings(
        played_2026.rename(columns={})[["home_team_id", "away_team_id", "home_goals", "away_goals"]],
        rules,
    )
    current_table = current_table.rename(
        columns={"goals_for": "goals_for", "goals_against": "goals_against"}
    )
    current_table.to_csv(PROCESSED_DIR / "current_standings_2026.csv", index=False)
    print(f"current_standings_2026 (calculada, não copiada): {current_table.shape}")
    print(current_table.head(5)[["team_id", "position", "points", "played"]].to_string(index=False))


if __name__ == "__main__":
    main()
