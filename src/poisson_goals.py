"""Modelo de gols baseado em Poisson, com ajuste Dixon-Coles (Fase 12).

Abordagem clássica: cada time tem uma força de ataque e uma força de defesa,
relativas à média da liga. O número esperado de gols do mandante é a média de gols
do mandante na liga vezes o ataque do mandante vezes a defesa do visitante (e
simetricamente para o visitante).

Poisson independente sozinha subestima sistematicamente placares baixos e
correlacionados (0x0, 1x0, 0x1, 1x1) — times "seguram o resultado" de um jeito que a
independência entre os dois gols não captura. Dixon & Coles (1997) corrigem isso
multiplicando a probabilidade conjunta por um fator `tau` só nesses 4 placares,
controlado por um parâmetro `rho` (tipicamente pequeno e negativo).

Duas formas de ajustar os parâmetros (ataque, defesa, rho) estão disponíveis:

- `fit_poisson_goals_model` (original): ataque/defesa por média ponderada
  (método dos momentos), depois `rho` sozinho por máxima verossimilhança perfilada.
  Rápido, mas estima os parâmetros em dois passos separados, não conjuntamente.
- `fit_poisson_goals_model_mle` (nova): ataque, defesa, home advantage e `rho`
  todos estimados de uma vez, maximizando a mesma log-verossimilhança do modelo
  completo. Metodologicamente mais correto — ver comparação empírica das duas no
  README ("MLE conjunta vs. método dos momentos").

As duas produzem o mesmo `PoissonGoalsModel` (mesma API: `lambdas`, `score_matrix`,
`outcome_probabilities`), então o restante do pipeline não precisa saber qual foi
usada.

Partidas mais recentes pesam mais no ajuste (`season_half_life`), porque a força de
um time muda ao longo do tempo — não faz sentido tratar 2003 e 2026 com o mesmo peso.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import poisson

DEFAULT_SEASON_HALF_LIFE = 3.0
MAX_GOALS = 10
MIN_LAMBDA = 0.1
RHO_BOUND = 0.3


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


def _dixon_coles_tau_vec(
    home_goals: np.ndarray, away_goals: np.ndarray, lambda_home: np.ndarray, lambda_away: np.ndarray, rho: float
) -> np.ndarray:
    """Mesma correção que `_dixon_coles_tau`, vetorizada sobre arrays de partidas
    (usada no ajuste por MLE, chamada centenas de vezes pelo otimizador — a versão
    escalar em loop Python seria proibitivamente lenta aqui).
    """
    tau = np.ones_like(lambda_home)
    m00 = (home_goals == 0) & (away_goals == 0)
    m01 = (home_goals == 0) & (away_goals == 1)
    m10 = (home_goals == 1) & (away_goals == 0)
    m11 = (home_goals == 1) & (away_goals == 1)
    tau[m00] = 1.0 - lambda_home[m00] * lambda_away[m00] * rho
    tau[m01] = 1.0 + lambda_home[m01] * rho
    tau[m10] = 1.0 + lambda_away[m10] * rho
    tau[m11] = 1.0 - rho
    return tau


@dataclass
class PoissonGoalsModel:
    league_avg_home_goals: float
    league_avg_away_goals: float
    attack: dict[str, float]
    defense: dict[str, float]
    rho: float = 0.0
    # Diagnósticos de ajuste — só populados por fit_poisson_goals_model_mle;
    # None para o método antigo ou quando o modelo é construído manualmente
    # (ex.: em testes). Nunca lidos por lambdas/score_matrix/outcome_probabilities.
    log_likelihood: float | None = None
    aic: float | None = None
    bic: float | None = None
    converged: bool | None = None
    n_iterations: int | None = None

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

    def top_scores(self, home_team_id: str, away_team_id: str, k: int = 5) -> list[dict]:
        """Os `k` placares mais prováveis, ordenados por probabilidade decrescente,
        cada um como {"home_goals", "away_goals", "probability"}. Use
        `top_scores_coverage` para saber que fração da distribuição completa esses
        `k` placares representam — normalmente bem menos que 100%: futebol tem
        muitos placares plausíveis, "o mais provável" não é "o esperado".
        """
        matrix = self.score_matrix(home_team_id, away_team_id)
        flat_indices = np.argsort(matrix.ravel())[::-1][:k]
        home_goals_idx, away_goals_idx = np.unravel_index(flat_indices, matrix.shape)
        return [
            {"home_goals": int(h), "away_goals": int(a), "probability": float(matrix[h, a])}
            for h, a in zip(home_goals_idx, away_goals_idx)
        ]

    @staticmethod
    def top_scores_coverage(top_scores: list[dict]) -> float:
        return float(sum(s["probability"] for s in top_scores))


def _method_of_moments_attack_defense(
    matches: pd.DataFrame, weights: np.ndarray, teams: np.ndarray, league_avg_goals: float
) -> tuple[dict[str, float], dict[str, float]]:
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

        attack[team_id] = np.average(goals_for, weights=team_weights) / league_avg_goals
        defense[team_id] = np.average(goals_against, weights=team_weights) / league_avg_goals
    return attack, defense


def _fit_rho(
    matches: pd.DataFrame, weights: np.ndarray, attack: dict[str, float], defense: dict[str, float],
    league_avg_home_goals: float, league_avg_away_goals: float,
) -> float:
    """Ajusta rho por máxima verossimilhança ponderada (mesmos pesos de recência
    usados no resto do modelo) sobre as partidas de treino, com ataque/defesa
    FIXOS (método dos momentos) — é um perfil de verossimilhança, não uma MLE
    conjunta. Ver `_fit_dixon_coles_mle` para a versão que ajusta tudo junto.
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

    result = minimize_scalar(neg_log_likelihood, bounds=(-RHO_BOUND, RHO_BOUND), method="bounded")
    return float(result.x)


def fit_poisson_goals_model(
    matches: pd.DataFrame, current_season: int, season_half_life: float = DEFAULT_SEASON_HALF_LIFE
) -> PoissonGoalsModel:
    """Método original: ataque/defesa por média ponderada, rho por perfil de MLE.
    `matches` precisa ter: season, home_team_id, away_team_id, home_goals, away_goals.
    """
    weights = 0.5 ** ((current_season - matches["season"]).clip(lower=0) / season_half_life)
    league_avg_home_goals = np.average(matches["home_goals"], weights=weights)
    league_avg_away_goals = np.average(matches["away_goals"], weights=weights)
    league_avg_goals = (league_avg_home_goals + league_avg_away_goals) / 2

    teams = pd.unique(matches[["home_team_id", "away_team_id"]].values.ravel("K"))
    attack, defense = _method_of_moments_attack_defense(matches, weights, teams, league_avg_goals)
    rho = _fit_rho(matches, weights.to_numpy(), attack, defense, league_avg_home_goals, league_avg_away_goals)

    return PoissonGoalsModel(
        league_avg_home_goals=league_avg_home_goals,
        league_avg_away_goals=league_avg_away_goals,
        attack=attack,
        defense=defense,
        rho=rho,
    )


def _fit_dixon_coles_mle(
    matches: pd.DataFrame,
    weights: np.ndarray,
    teams: np.ndarray,
    initial_attack: dict[str, float],
    initial_defense: dict[str, float],
    initial_home: float,
    initial_away: float,
) -> tuple[dict[str, float], dict[str, float], float, float, float, float, float, float, bool, int]:
    """Estima ataque, defesa, as duas médias-base (mandante/visitante) e `rho`
    de uma vez, maximizando uma única log-verossimilhança — ao contrário de
    `fit_poisson_goals_model`, que ajusta ataque/defesa por média e só depois
    perfila `rho` separadamente.

    Parametrização (para manter ataque/defesa sempre positivos e identificáveis):
    otimiza-se em log(ataque_i) e log(defesa_i); o time N-ésimo tem seu log-efeito
    fixado como menos a soma dos outros (restrição soma(log ataque) = 0, idem
    defesa) — sem essa restrição, multiplicar todo ataque por k e dividir toda
    defesa por k (ou vice-versa) dá exatamente a mesma verossimilhança, então os
    parâmetros não seriam identificáveis. `rho` é reparametrizado via tangente
    hiperbólica para ficar sempre dentro de (-RHO_BOUND, RHO_BOUND) sem precisar de
    restrição de caixa no otimizador (mais estável numericamente).

    Ponto de partida: o resultado do método dos momentos (`initial_attack`/
    `initial_defense`) — já é uma estimativa razoável, então a otimização converge
    rápido a partir dali em vez de começar do zero.
    """
    n_teams = len(teams)
    n_free = n_teams - 1
    team_index = {team_id: i for i, team_id in enumerate(teams)}
    # Penalização ridge leve sobre log(ataque)/log(defesa): sem isso, um time com
    # amostra pequena ou extrema (ex.: nunca marcou gol) pode levar a otimização a
    # divergir — a restrição de identificabilidade (soma dos log-efeitos = 0) faz
    # o efeito de UM time em 0 empurrar os outros para o infinito para compensar.
    # O peso é pequeno o bastante para não distorcer o ajuste em amostras normais
    # (a log-verossimilhança real domina), mas evita esse colapso numérico.
    ridge_weight = 2.0

    home_idx = matches["home_team_id"].map(team_index).to_numpy()
    away_idx = matches["away_team_id"].map(team_index).to_numpy()
    home_goals = matches["home_goals"].to_numpy(dtype=float)
    away_goals = matches["away_goals"].to_numpy(dtype=float)
    weights_arr = np.asarray(weights, dtype=float)

    def unpack(params: np.ndarray):
        log_attack_free = params[:n_free]
        log_defense_free = params[n_free : 2 * n_free]
        log_base_home = params[2 * n_free]
        log_base_away = params[2 * n_free + 1]
        rho_raw = params[2 * n_free + 2]
        log_attack = np.append(log_attack_free, -log_attack_free.sum())
        log_defense = np.append(log_defense_free, -log_defense_free.sum())
        rho = RHO_BOUND * np.tanh(rho_raw)
        return log_attack, log_defense, log_base_home, log_base_away, rho

    def neg_log_likelihood(params: np.ndarray) -> float:
        log_attack, log_defense, log_base_home, log_base_away, rho = unpack(params)
        lambda_home = np.exp(log_base_home + log_attack[home_idx] + log_defense[away_idx])
        lambda_away = np.exp(log_base_away + log_attack[away_idx] + log_defense[home_idx])
        tau = _dixon_coles_tau_vec(home_goals, away_goals, lambda_home, lambda_away, rho)
        tau = np.clip(tau, 1e-8, None)
        log_lik = (
            np.log(tau)
            + poisson.logpmf(home_goals, lambda_home)
            + poisson.logpmf(away_goals, lambda_away)
        )
        ridge_penalty = ridge_weight * float(np.sum(log_attack**2) + np.sum(log_defense**2))
        return -float(np.sum(weights_arr * log_lik)) + ridge_penalty

    # Chão pequeno antes do log: método dos momentos pode dar exatamente 0 (um
    # time que nunca marcou/nunca sofreu gol na amostra) e log(0) = -inf quebraria
    # o ponto de partida do otimizador antes mesmo de começar.
    floor = 0.05
    x0 = np.concatenate(
        [
            np.log([max(initial_attack[t], floor) for t in teams[:-1]]),
            np.log([max(initial_defense[t], floor) for t in teams[:-1]]),
            [np.log(initial_home), np.log(initial_away), 0.0],
        ]
    )

    result = minimize(neg_log_likelihood, x0, method="L-BFGS-B")
    log_attack, log_defense, log_base_home, log_base_away, rho = unpack(result.x)

    attack = {team_id: float(np.exp(log_attack[i])) for i, team_id in enumerate(teams)}
    defense = {team_id: float(np.exp(log_defense[i])) for i, team_id in enumerate(teams)}
    base_home = float(np.exp(log_base_home))
    base_away = float(np.exp(log_base_away))
    log_likelihood = -float(result.fun)
    n_params = len(x0)
    n_obs = len(matches)
    aic = 2 * n_params - 2 * log_likelihood
    bic = n_params * np.log(n_obs) - 2 * log_likelihood

    return (
        attack, defense, base_home, base_away, float(rho),
        log_likelihood, aic, bic, bool(result.success), int(result.nit),
    )


def fit_poisson_goals_model_mle(
    matches: pd.DataFrame, current_season: int, season_half_life: float = DEFAULT_SEASON_HALF_LIFE
) -> PoissonGoalsModel:
    """Ataque, defesa, médias-base e `rho` ajustados juntos por máxima
    verossimilhança (ver `_fit_dixon_coles_mle`). Mesma API de saída de
    `fit_poisson_goals_model` — troca a implementação interna do ajuste, não o
    contrato com o resto do pipeline (`PoissonGoalsModel.lambdas`/`score_matrix`/
    `outcome_probabilities` funcionam do mesmo jeito para os dois).

    `matches` precisa ter: season, home_team_id, away_team_id, home_goals, away_goals.
    """
    weights = 0.5 ** ((current_season - matches["season"]).clip(lower=0) / season_half_life)
    league_avg_home_goals = np.average(matches["home_goals"], weights=weights)
    league_avg_away_goals = np.average(matches["away_goals"], weights=weights)
    league_avg_goals = (league_avg_home_goals + league_avg_away_goals) / 2

    teams = pd.unique(matches[["home_team_id", "away_team_id"]].values.ravel("K"))
    initial_attack, initial_defense = _method_of_moments_attack_defense(
        matches, weights, teams, league_avg_goals
    )

    (
        attack, defense, base_home, base_away, rho,
        log_likelihood, aic, bic, converged, n_iterations,
    ) = _fit_dixon_coles_mle(
        matches, weights, teams, initial_attack, initial_defense,
        league_avg_home_goals, league_avg_away_goals,
    )

    return PoissonGoalsModel(
        league_avg_home_goals=base_home,
        league_avg_away_goals=base_away,
        attack=attack,
        defense=defense,
        rho=rho,
        log_likelihood=log_likelihood,
        aic=aic,
        bic=bic,
        converged=converged,
        n_iterations=n_iterations,
    )
