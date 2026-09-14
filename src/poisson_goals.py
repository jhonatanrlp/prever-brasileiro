"""Modelo de gols baseado em Poisson, com ajuste Dixon-Coles (Fase 12).

Abordagem clássica: cada time tem uma força de ataque e uma força de defesa,
relativas à média da liga. O número esperado de gols do mandante é a média de gols
do mandante na liga vezes o ataque do mandante vezes a defesa do visitante (e
simetricamente para o visitante).

Poisson independente sozinha subestima sistematicamente placares baixos e
correlacionados (0x0, 1x0, 0x1, 1x1) — times "seguram o resultado" de um jeito que a
independência entre os dois gols não captura. Dixon & Coles (1997) corrigem isso
multiplicando a probabilidade conjunta por um fator `tau` só nesses 4 placares,
controlado por um parâmetro `rho` (tipicamente pequeno e negativo) ajustado por
máxima verossimilhança nos dados de treino — não escolhido a dedo.

Partidas mais recentes pesam mais no ajuste das forças de ataque/defesa
(`season_half_life`), porque a força de um time muda ao longo do tempo — não faz
sentido tratar 2003 e 2026 com o mesmo peso.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import poisson

DEFAULT_SEASON_HALF_LIFE = 3.0
MAX_GOALS = 10
MIN_LAMBDA = 0.1


def _dixon_coles_tau(home_goals: int, away_goals: int, lambda_home: float, lambda_away: float, rho: float) -> float:
    if home_goals == 0 and away_goals == 0:
        return 1.0 - lambda_home * lambda_away * rho
    if home_goals == 0 and away_goals == 1:
        return 1.0 + lambda_home * rho
    if home_goals == 1 and away_goals == 0:
        return 1.0 + lambda_away * rho
    if home_goals == 1 and away_goals == 1:
        return 1.0 - rho
    return 1.0


@dataclass
class PoissonGoalsModel:
    league_avg_home_goals: float
    league_avg_away_goals: float
    attack: dict[str, float]
    defense: dict[str, float]
    rho: float = 0.0

    def lambdas(self, home_team_id: str, away_team_id: str) -> tuple[float, float]:
        home_attack = self.attack.get(home_team_id, 1.0)
        away_defense = self.defense.get(away_team_id, 1.0)
        away_attack = self.attack.get(away_team_id, 1.0)
        home_defense = self.defense.get(home_team_id, 1.0)

        lambda_home = max(self.league_avg_home_goals * home_attack * away_defense, MIN_LAMBDA)
        lambda_away = max(self.league_avg_away_goals * away_attack * home_defense, MIN_LAMBDA)
        return lambda_home, lambda_away

    def score_matrix(self, home_team_id: str, away_team_id: str) -> np.ndarray:
        """Matriz (MAX_GOALS+1) x (MAX_GOALS+1) com P(placar = i x j), já com o
        ajuste Dixon-Coles aplicado e normalizada para somar 1.
        """
        lambda_home, lambda_away = self.lambdas(home_team_id, away_team_id)
        goals = np.arange(0, MAX_GOALS + 1)
        joint = np.outer(poisson.pmf(goals, lambda_home), poisson.pmf(goals, lambda_away))

        for i in (0, 1):
            for j in (0, 1):
                joint[i, j] *= _dixon_coles_tau(i, j, lambda_home, lambda_away, self.rho)

        return joint / joint.sum()

    def outcome_probabilities(self, home_team_id: str, away_team_id: str) -> tuple[float, float, float]:
        joint = self.score_matrix(home_team_id, away_team_id)
        p_home = float(np.tril(joint, k=-1).sum())
        p_draw = float(np.trace(joint))
        p_away = float(np.triu(joint, k=1).sum())
        return p_home, p_draw, p_away


def _fit_rho(
    matches: pd.DataFrame, weights: np.ndarray, attack: dict[str, float], defense: dict[str, float],
    league_avg_home_goals: float, league_avg_away_goals: float,
) -> float:
    """Ajusta rho por máxima verossimilhança ponderada (mesmos pesos de recência
    usados no resto do modelo) sobre as partidas de treino.
    """
    home_attack = matches["home_team_id"].map(attack).fillna(1.0).to_numpy()
    away_defense = matches["away_team_id"].map(defense).fillna(1.0).to_numpy()
    away_attack = matches["away_team_id"].map(attack).fillna(1.0).to_numpy()
    home_defense = matches["home_team_id"].map(defense).fillna(1.0).to_numpy()

    lambda_home = np.maximum(league_avg_home_goals * home_attack * away_defense, MIN_LAMBDA)
    lambda_away = np.maximum(league_avg_away_goals * away_attack * home_defense, MIN_LAMBDA)
    home_goals = matches["home_goals"].to_numpy()
    away_goals = matches["away_goals"].to_numpy()

    def neg_log_likelihood(rho: float) -> float:
        tau = np.array(
            [
                _dixon_coles_tau(h, a, lh, la, rho)
                for h, a, lh, la in zip(home_goals, away_goals, lambda_home, lambda_away)
            ]
        )
        tau = np.clip(tau, 1e-6, None)  # tau pode ficar <=0 para rho extremo; log só de valores positivos
        log_lik = (
            np.log(tau)
            + poisson.logpmf(home_goals, lambda_home)
            + poisson.logpmf(away_goals, lambda_away)
        )
        return -float(np.sum(weights * log_lik))

    result = minimize_scalar(neg_log_likelihood, bounds=(-0.3, 0.3), method="bounded")
    return float(result.x)


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

    rho = _fit_rho(matches, weights.to_numpy(), attack, defense, league_avg_home_goals, league_avg_away_goals)

    return PoissonGoalsModel(
        league_avg_home_goals=league_avg_home_goals,
        league_avg_away_goals=league_avg_away_goals,
        attack=attack,
        defense=defense,
        rho=rho,
    )
