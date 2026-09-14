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


def _head_to_head_winner(team_a: str, team_b: str, matches: pd.DataFrame, rules: CompetitionRules) -> str | None:
    """Só se aplica oficialmente "entre duas equipes" — nunca para grupos de 3+
    times empatados. Retorna o team_id com mais pontos nos jogos entre os dois, ou
    None se também empatarem no confronto direto.
    """
    between = matches[
        ((matches["home_team_id"] == team_a) & (matches["away_team_id"] == team_b))
        | ((matches["home_team_id"] == team_b) & (matches["away_team_id"] == team_a))
    ]
    points = {team_a: 0, team_b: 0}
    for row in between.itertuples(index=False):
        home_points = rules.points_for_result(row.home_goals, row.away_goals)
        away_points = rules.points_for_result(row.away_goals, row.home_goals)
        points[row.home_team_id] += home_points
        points[row.away_team_id] += away_points
    if points[team_a] == points[team_b]:
        return None
    return team_a if points[team_a] > points[team_b] else team_b


def _card_counts(cards: pd.DataFrame, team_id: str) -> tuple[int, int]:
    team_cards = cards[cards["team_id"] == team_id]["cartao"]
    red = int((team_cards == "Vermelho").sum())
    yellow = int((team_cards == "Amarelo").sum())
    return red, yellow


def _break_ties(
    table: pd.DataFrame, matches: pd.DataFrame, rules: CompetitionRules, cards: pd.DataFrame | None
) -> pd.DataFrame:
    """Resolve empates em (pontos, vitórias, saldo, gols pró) usando confronto
    direto (só entre exatamente 2 times) e cartões (se `cards` for informado).
    Grupos que continuarem empatados mantêm a ordem original (sorteio — não
    simulado).
    """
    order = table["team_id"].tolist()
    tie_key = list(zip(table["points"], table["wins"], table["goal_difference"], table["goals_for"]))

    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and tie_key[j] == tie_key[i]:
            j += 1
        group = order[i:j]
        resolved = False

        if len(group) == 2 and "head_to_head" in rules.tiebreakers:
            winner = _head_to_head_winner(group[0], group[1], matches, rules)
            if winner is not None:
                if winner != group[0]:
                    group = [group[1], group[0]]
                    order[i:j] = group
                resolved = True

        if not resolved and len(group) > 1 and cards is not None:
            index_by_card_type = {"fewer_red_cards": 0, "fewer_yellow_cards": 1}
            criteria = [c for c in rules.tiebreakers if c in index_by_card_type]
            if criteria:
                counts = {team_id: _card_counts(cards, team_id) for team_id in group}
                order[i:j] = sorted(
                    group, key=lambda t: tuple(counts[t][index_by_card_type[c]] for c in criteria)
                )

        i = j

    return table.set_index("team_id").loc[order].reset_index()


def build_standings(
    matches: pd.DataFrame, rules: CompetitionRules, cards: pd.DataFrame | None = None
) -> pd.DataFrame:
    """`matches` precisa ter: home_team_id, away_team_id, home_goals, away_goals.
    Só devem ser passadas partidas já finalizadas (reais ou simuladas).

    `cards` é opcional: DataFrame com `team_id` e `cartao` ("Amarelo"/"Vermelho"),
    já filtrado para as mesmas partidas de `matches` (ex.: `data/processed/cards.csv`
    filtrado pela temporada). Sem `cards`, os critérios de cartões são ignorados e
    empates que chegarem até ali mantêm a ordem por confronto direto/sorteio.
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

    sort_columns = {
        "points": "points",
        "wins": "wins",
        "goal_difference": "goal_difference",
        "goals_for": "goals_for",
    }
    ascending_cols = [sort_columns[c] for c in rules.tiebreakers if c in sort_columns]
    table = table.sort_values(by=ascending_cols, ascending=False).reset_index(drop=True)
    table = _break_ties(table, matches, rules, cards)
    table["position"] = table.index + 1
    return table
