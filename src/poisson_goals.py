"""Modelo de gols baseado em Poisson (Fase 12).

Abordagem clássica: cada time tem uma força de ataque e uma força de defesa,
relativas à média da liga. O número esperado de gols do mandante é a média de gols
do mandante na liga vezes o ataque do mandante vezes a defesa do visitante (e
simetricamente para o visitante). A partir de (lambda_home, lambda_away),
`outcome_probabilities` deriva P(vitória mandante)/P(empate)/P(vitória visitante)
somando a distribuição de Poisson bivariada independente sobre todos os placares
plausíveis.

Partidas mais recentes pesam mais no ajuste (`season_half_life`), porque a força de
um time muda ao longo do tempo — não faz sentido tratar 2003 e 2026 com o mesmo peso.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import poisson

DEFAULT_SEASON_HALF_LIFE = 3.0
MAX_GOALS = 10
MIN_LAMBDA = 0.1


@dataclass
class PoissonGoalsModel:
    league_avg_home_goals: float
    league_avg_away_goals: float
    attack: dict[str, float]
    defense: dict[str, float]

    def lambdas(self, home_team_id: str, away_team_id: str) -> tuple[float, float]:
        home_attack = self.attack.get(home_team_id, 1.0)
        away_defense = self.defense.get(away_team_id, 1.0)
        away_attack = self.attack.get(away_team_id, 1.0)
        home_defense = self.defense.get(home_team_id, 1.0)

        lambda_home = max(self.league_avg_home_goals * home_attack * away_defense, MIN_LAMBDA)
        lambda_away = max(self.league_avg_away_goals * away_attack * home_defense, MIN_LAMBDA)
        return lambda_home, lambda_away

    def outcome_probabilities(self, home_team_id: str, away_team_id: str) -> tuple[float, float, float]:
        lambda_home, lambda_away = self.lambdas(home_team_id, away_team_id)
        goals = np.arange(0, MAX_GOALS + 1)
        home_probs = poisson.pmf(goals, lambda_home)
        away_probs = poisson.pmf(goals, lambda_away)
        joint = np.outer(home_probs, away_probs)

        p_home = float(np.tril(joint, k=-1).sum())
        p_draw = float(np.trace(joint))
        p_away = float(np.triu(joint, k=1).sum())
        total = p_home + p_draw + p_away
        return p_home / total, p_draw / total, p_away / total


def fit_poisson_goals_model(
    matches: pd.DataFrame, current_season: int, season_half_life: float = DEFAULT_SEASON_HALF_LIFE
) -> PoissonGoalsModel:
    """`matches` precisa ter: season, home_team_id, away_team_id, home_goals, away_goals."""
    weights = 0.5 ** ((current_season - matches["season"]).clip(lower=0) / season_half_life)

    league_avg_home_goals = np.average(matches["home_goals"], weights=weights)
    league_avg_away_goals = np.average(matches["away_goals"], weights=weights)
    league_avg_goals = (league_avg_home_goals + league_avg_away_goals) / 2

    teams = pd.unique(matches[["home_team_id", "away_team_id"]].values.ravel("K"))
    attack: dict[str, float] = {}
    defense: dict[str, float] = {}

    for team_id in teams:
        home_mask = matches["home_team_id"] == team_id
        away_mask = matches["away_team_id"] == team_id

        goals_for = np.concatenate(
            [matches.loc[home_mask, "home_goals"], matches.loc[away_mask, "away_goals"]]
        )
        goals_against = np.concatenate(
            [matches.loc[home_mask, "away_goals"], matches.loc[away_mask, "home_goals"]]
        )
        team_weights = np.concatenate([weights[home_mask], weights[away_mask]])

        if len(goals_for) == 0 or team_weights.sum() == 0:
            attack[team_id] = 1.0
            defense[team_id] = 1.0
            continue

        avg_for = np.average(goals_for, weights=team_weights)
        avg_against = np.average(goals_against, weights=team_weights)
        attack[team_id] = avg_for / league_avg_goals
        defense[team_id] = avg_against / league_avg_goals

    return PoissonGoalsModel(
        league_avg_home_goals=league_avg_home_goals,
        league_avg_away_goals=league_avg_away_goals,
        attack=attack,
        defense=defense,
    )
