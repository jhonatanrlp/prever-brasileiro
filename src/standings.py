"""Regras do campeonato (config, não hardcoded) e construção da tabela de
classificação a partir de partidas finalizadas.

`CompetitionRules` vem de configs/config.yaml — pontuação, rebaixamento e vagas
continentais nunca são constantes espalhadas pelo código. `build_standings` é
desacoplado do modelo preditivo: recebe apenas placares já decididos (reais ou
sorteados pelo simulador de Monte Carlo) e devolve a classificação.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "config.yaml"


@dataclass
class CompetitionRules:
    points_win: int
    points_draw: int
    points_loss: int
    n_teams: int
    n_relegated: int
    libertadores_direct: int
    libertadores_qualifiers: int
    sulamericana_slots: int
    tiebreakers: list[str] = field(default_factory=list)

    @property
    def libertadores_total(self) -> int:
        return self.libertadores_direct + self.libertadores_qualifiers

    def points_for_result(self, goals_for: int, goals_against: int) -> int:
        if goals_for > goals_against:
            return self.points_win
        if goals_for < goals_against:
            return self.points_loss
        return self.points_draw

    @classmethod
    def from_config(cls, path: Path | str = DEFAULT_CONFIG_PATH) -> "CompetitionRules":
        with open(path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        comp = cfg["competition"]
        return cls(
            points_win=comp["points"]["win"],
            points_draw=comp["points"]["draw"],
            points_loss=comp["points"]["loss"],
            n_teams=comp["n_teams"],
            n_relegated=comp["n_relegated"],
            libertadores_direct=comp["continental_slots"]["libertadores"]["direct"],
            libertadores_qualifiers=comp["continental_slots"]["libertadores"]["qualifiers"],
            sulamericana_slots=comp["continental_slots"]["sulamericana"]["slots"],
            tiebreakers=comp["tiebreakers"],
        )


@dataclass
class StandingsRow:
    team_id: str
    points: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    goals_for: int = 0
    goals_against: int = 0

    @property
    def goal_difference(self) -> int:
        return self.goals_for - self.goals_against

    @property
    def played(self) -> int:
        return self.wins + self.draws + self.losses


def build_standings(matches: pd.DataFrame, rules: CompetitionRules) -> pd.DataFrame:
    """`matches` precisa ter: home_team_id, away_team_id, home_goals, away_goals.
    Só devem ser passadas partidas já finalizadas (reais ou simuladas).
    """
    rows: dict[str, StandingsRow] = {}

    def _get(team_id: str) -> StandingsRow:
        if team_id not in rows:
            rows[team_id] = StandingsRow(team_id=team_id)
        return rows[team_id]

    for row in matches.itertuples(index=False):
        home = _get(row.home_team_id)
        away = _get(row.away_team_id)
        home.goals_for += row.home_goals
        home.goals_against += row.away_goals
        away.goals_for += row.away_goals
        away.goals_against += row.home_goals

        home_points = rules.points_for_result(row.home_goals, row.away_goals)
        away_points = rules.points_for_result(row.away_goals, row.home_goals)
        home.points += home_points
        away.points += away_points

        if row.home_goals > row.away_goals:
            home.wins += 1
            away.losses += 1
        elif row.home_goals < row.away_goals:
            away.wins += 1
            home.losses += 1
        else:
            home.draws += 1
            away.draws += 1

    table = pd.DataFrame(
        [
            {
                "team_id": r.team_id,
                "points": r.points,
                "played": r.played,
                "wins": r.wins,
                "draws": r.draws,
                "losses": r.losses,
                "goals_for": r.goals_for,
                "goals_against": r.goals_against,
                "goal_difference": r.goal_difference,
            }
            for r in rows.values()
        ]
    )

    # Critérios implementados: pontos, vitórias, saldo de gols, gols pró (cobrem a
    # esmagadora maioria dos casos reais). Confronto direto e cartões (também
    # previstos em configs/config.yaml: competition.tiebreakers) exigem uma
    # mini-liga par a par e não foram implementados nesta fase.
    sort_columns = {
        "points": "points",
        "wins": "wins",
        "goal_difference": "goal_difference",
        "goals_for": "goals_for",
    }
    ascending_cols = [sort_columns[c] for c in rules.tiebreakers if c in sort_columns]
    table = table.sort_values(by=ascending_cols, ascending=False).reset_index(drop=True)
    table["position"] = table.index + 1
    return table
