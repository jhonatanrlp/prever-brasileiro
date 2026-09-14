"""Simulação de Monte Carlo do restante da temporada (Fase 13).

Recebe a tabela atual (pontos/GP/GC reais até agora), os jogos restantes e um
modelo de gols (Poisson + Dixon-Coles). Para cada temporada simulada, sorteia um
placar por partida restante diretamente da matriz de probabilidades conjunta do
modelo (`PoissonGoalsModel.score_matrix`, já com o ajuste de correlação
Dixon-Coles) — não de duas Poisson independentes, o que jogaria fora exatamente a
correlação que o Dixon-Coles corrige. Atualiza pontos e saldo, e classifica ao
final com os mesmos critérios de desempate usados para tabelas reais
(`src/standings.py`).

Desacoplado do modelo de classificação (V/E/D): usa o modelo de gols para gerar o
placar completo, o que automaticamente resolve V/E/D e saldo de gols de forma
consistente — não são dois modelos independentes que podem discordar entre si.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.poisson_goals import PoissonGoalsModel
from src.standings import CompetitionRules


def _sample_scoreline(
    goals_model: PoissonGoalsModel, home_id: str, away_id: str, n_simulations: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Sorteia `n_simulations` placares (gols mandante, gols visitante) da matriz
    de probabilidade conjunta do modelo (transformada inversa sobre a distribuição
    achatada) — preserva a correlação Dixon-Coles, que amostrar Poisson(home) e
    Poisson(away) de forma independente destruiria.
    """
    joint = goals_model.score_matrix(home_id, away_id)
    n_goal_values = joint.shape[0]
    flat_cumulative = np.cumsum(joint.ravel())
    flat_cumulative[-1] = 1.0  # evita erro de arredondamento deixar a soma < 1

    draws = rng.random(n_simulations)
    flat_index = np.searchsorted(flat_cumulative, draws)
    return flat_index // n_goal_values, flat_index % n_goal_values


def simulate_remaining_season(
    current_table: pd.DataFrame,
    remaining_fixtures: pd.DataFrame,
    goals_model: PoissonGoalsModel,
    rules: CompetitionRules,
    n_simulations: int,
    random_seed: int = 42,
) -> pd.DataFrame:
    """`current_table` precisa ter: team_id, points, goals_for, goals_against
    (estado real, já disputado). `remaining_fixtures` precisa ter: home_team_id,
    away_team_id.

    Retorna um DataFrame com uma linha por time: pontos esperados/mediana/P10/P90,
    posição esperada e probabilidades de título/Libertadores/Sul-Americana/
    rebaixamento, estimadas pela frequência empírica nas `n_simulations` temporadas.
    """
    rng = np.random.default_rng(random_seed)

    team_ids = current_table["team_id"].tolist()
    n_teams = len(team_ids)
    team_index = {team_id: i for i, team_id in enumerate(team_ids)}

    # Estado acumulado por (time, simulação): shape (n_teams, n_simulations).
    points = np.tile(current_table.set_index("team_id").loc[team_ids, "points"].to_numpy()[:, None], n_simulations).astype(np.int64)
    goals_for = np.tile(current_table.set_index("team_id").loc[team_ids, "goals_for"].to_numpy()[:, None], n_simulations).astype(np.int64)
    goals_against = np.tile(current_table.set_index("team_id").loc[team_ids, "goals_against"].to_numpy()[:, None], n_simulations).astype(np.int64)

    home_ids = remaining_fixtures["home_team_id"].to_numpy()
    away_ids = remaining_fixtures["away_team_id"].to_numpy()

    for home_id, away_id in zip(home_ids, away_ids):
        home_idx, away_idx = team_index[home_id], team_index[away_id]
        hg, ag = _sample_scoreline(goals_model, home_id, away_id, n_simulations, rng)

        goals_for[home_idx] += hg
        goals_against[home_idx] += ag
        goals_for[away_idx] += ag
        goals_against[away_idx] += hg

        home_win = hg > ag
        away_win = ag > hg
        points[home_idx] += np.where(home_win, rules.points_win, np.where(away_win, rules.points_loss, rules.points_draw))
        points[away_idx] += np.where(away_win, rules.points_win, np.where(home_win, rules.points_loss, rules.points_draw))

    goal_difference = goals_for - goals_against

    # Ranking por simulação: ordena por (pontos, saldo, gols pró) decrescente.
    # np.lexsort ordena de forma ascendente pela ÚLTIMA chave como primária,
    # então: (a) transpomos para (n_simulations, n_teams), já que lexsort ordena
    # ao longo do último eixo; (b) negamos as métricas para obter ordem
    # decrescente com pontos como chave primária (última da tupla).
    points_t = points.T
    goal_difference_t = goal_difference.T
    goals_for_t = goals_for.T
    order = np.lexsort((-goals_for_t, -goal_difference_t, -points_t))  # shape (n_simulations, n_teams)

    positions = np.empty_like(order)
    sim_axis = np.arange(n_simulations)[:, None]
    rank = np.tile(np.arange(1, n_teams + 1), (n_simulations, 1))
    positions[sim_axis, order] = rank  # positions[sim, team_idx] = posição final
    positions = positions.T  # de volta para (n_teams, n_simulations)

    position_counts = {
        team_id: np.bincount(positions[team_index[team_id]] - 1, minlength=n_teams)
        for team_id in team_ids
    }
    final_points = {team_id: points[team_index[team_id]] for team_id in team_ids}
    base_points = {team_id: int(current_table.set_index("team_id").loc[team_id, "points"]) for team_id in team_ids}

    rows = []
    for team_id in team_ids:
        points_dist = final_points[team_id]
        team_position_counts = position_counts[team_id]
        expected_position = float(
            np.average(np.arange(1, rules.n_teams + 1), weights=team_position_counts)
        )
        probs = team_position_counts / n_simulations

        title_prob = probs[0]
        libertadores_prob = probs[: rules.libertadores_total].sum()
        sulamericana_prob = probs[rules.libertadores_total: rules.libertadores_total + rules.sulamericana_slots].sum()
        relegation_prob = probs[rules.n_teams - rules.n_relegated:].sum()

        rows.append(
            {
                "team_id": team_id,
                "current_points": base_points[team_id],
                "expected_points": float(points_dist.mean()),
                "median_points": float(np.median(points_dist)),
                "p10_points": float(np.percentile(points_dist, 10)),
                "p90_points": float(np.percentile(points_dist, 90)),
                "expected_position": expected_position,
                "title_probability": float(title_prob),
                "libertadores_probability": float(libertadores_prob),
                "sulamericana_probability": float(sulamericana_prob),
                "relegation_probability": float(relegation_prob),
            }
        )

    return pd.DataFrame(rows).sort_values("expected_points", ascending=False).reset_index(drop=True)
