"""Constrói o modelo dimensional em data/processed/ a partir dos CSVs brutos.

Tabelas geradas:
    matches.csv           1 linha por partida (season, rodada, mandante/visitante por team_id, placar)
    team_match_stats.csv  1 linha por (match_id, team_id): chutes, posse, escanteios, cartões, etc.
    goals.csv             1 linha por gol
    cards.csv             1 linha por cartão
    teams.csv             cópia do dicionário mestre de times (data/external/team_mapping.csv)

Uso:
    python scripts/build_dataset.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.normalize_teams import TeamNormalizer  # noqa: E402
from src.seasons import assign_season_ids  # noqa: E402

RAW_DIR = ROOT / "data" / "raw" / "adaoduque"
PROCESSED_DIR = ROOT / "data" / "processed"
DATE_FORMAT = "%d/%m/%Y"


def _result(home_goals: int, away_goals: int) -> str:
    if home_goals > away_goals:
        return "H"
    if home_goals < away_goals:
        return "A"
    return "D"


def build_matches(normalizer: TeamNormalizer) -> pd.DataFrame:
    df = pd.read_csv(RAW_DIR / "campeonato-brasileiro-full.csv")
    df["match_date"] = pd.to_datetime(df["data"], format=DATE_FORMAT)
    df["season"] = assign_season_ids(df["match_date"])

    matches = pd.DataFrame(
        {
            "match_id": df["ID"],
            "season": df["season"],
            "competition": "Brasileirão Série A",
            "round": df["rodata"],
            "date": df["match_date"],
            "home_team_id": normalizer.normalize_series(df["mandante"]),
            "away_team_id": normalizer.normalize_series(df["visitante"]),
            "home_goals": df["mandante_Placar"],
            "away_goals": df["visitante_Placar"],
            "arena": df["arena"],
        }
    )
    matches["result"] = [
        _result(h, a) for h, a in zip(matches["home_goals"], matches["away_goals"])
    ]
    return matches.sort_values(["date", "match_id"]).reset_index(drop=True)


def build_team_match_stats(normalizer: TeamNormalizer) -> pd.DataFrame:
    path = RAW_DIR / "campeonato-brasileiro-estatisticas-full.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["team_id"] = normalizer.normalize_series(df["clube"])
    return df.rename(columns={"partida_id": "match_id", "rodata": "round"}).drop(columns=["clube"])


def build_goals(normalizer: TeamNormalizer) -> pd.DataFrame:
    path = RAW_DIR / "campeonato-brasileiro-gols.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["team_id"] = normalizer.normalize_series(df["clube"])
    return df.rename(columns={"partida_id": "match_id", "rodata": "round"}).drop(columns=["clube"])


def build_cards(normalizer: TeamNormalizer) -> pd.DataFrame:
    path = RAW_DIR / "campeonato-brasileiro-cartoes.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["team_id"] = normalizer.normalize_series(df["clube"])
    return df.rename(columns={"partida_id": "match_id", "rodata": "round"}).drop(columns=["clube"])


def main() -> None:
    normalizer = TeamNormalizer()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    matches = build_matches(normalizer)
    matches.to_csv(PROCESSED_DIR / "matches.csv", index=False)
    print(f"matches: {matches.shape}")

    stats = build_team_match_stats(normalizer)
    stats.to_csv(PROCESSED_DIR / "team_match_stats.csv", index=False)
    print(f"team_match_stats: {stats.shape}")

    goals = build_goals(normalizer)
    goals.to_csv(PROCESSED_DIR / "goals.csv", index=False)
    print(f"goals: {goals.shape}")

    cards = build_cards(normalizer)
    cards.to_csv(PROCESSED_DIR / "cards.csv", index=False)
    print(f"cards: {cards.shape}")

    normalizer.mapping.to_csv(PROCESSED_DIR / "teams.csv", index=False)
    print(f"teams: {normalizer.mapping.shape}")


if __name__ == "__main__":
    main()
