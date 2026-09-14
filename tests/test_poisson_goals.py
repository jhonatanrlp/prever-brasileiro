import numpy as np
import pandas as pd

from src.poisson_goals import PoissonGoalsModel, _dixon_coles_tau, fit_poisson_goals_model


def _synthetic_matches() -> pd.DataFrame:
    # "A" ataca muito e defende bem; "B" é mediano; "C" ataca pouco e defende mal.
    rows = []
    for _ in range(10):
        rows.append({"season": 2024, "home_team_id": "A", "away_team_id": "B", "home_goals": 3, "away_goals": 1})
        rows.append({"season": 2024, "home_team_id": "B", "away_team_id": "A", "home_goals": 1, "away_goals": 2})
        rows.append({"season": 2024, "home_team_id": "B", "away_team_id": "C", "home_goals": 2, "away_goals": 0})
        rows.append({"season": 2024, "home_team_id": "C", "away_team_id": "B", "home_goals": 0, "away_goals": 2})
        rows.append({"season": 2024, "home_team_id": "A", "away_team_id": "C", "home_goals": 4, "away_goals": 0})
        rows.append({"season": 2024, "home_team_id": "C", "away_team_id": "A", "home_goals": 0, "away_goals": 3})
    return pd.DataFrame(rows)


def test_strong_attacker_has_attack_strength_above_one():
    model = fit_poisson_goals_model(_synthetic_matches(), current_season=2024)
    assert model.attack["A"] > 1.0
    assert model.attack["C"] < 1.0


def test_weak_defender_has_defense_strength_above_one():
    model = fit_poisson_goals_model(_synthetic_matches(), current_season=2024)
    assert model.defense["C"] > 1.0  # C sofre muitos gols
    assert model.defense["A"] < 1.0  # A sofre poucos gols


def test_outcome_probabilities_sum_to_one():
    model = fit_poisson_goals_model(_synthetic_matches(), current_season=2024)
    p_home, p_draw, p_away = model.outcome_probabilities("A", "C")
    assert abs((p_home + p_draw + p_away) - 1.0) < 1e-9


def test_strong_team_favored_at_home_against_weak_team():
    model = fit_poisson_goals_model(_synthetic_matches(), current_season=2024)
    p_home, _, p_away = model.outcome_probabilities("A", "C")
    assert p_home > p_away


def test_unknown_team_falls_back_to_league_average_strength():
    model = fit_poisson_goals_model(_synthetic_matches(), current_season=2024)
    lambda_home, lambda_away = model.lambdas("TIME_DESCONHECIDO", "OUTRO_DESCONHECIDO")
    assert lambda_home > 0
    assert lambda_away > 0


def test_dixon_coles_tau_is_one_outside_the_four_low_scores():
    assert _dixon_coles_tau(2, 0, 1.5, 1.0, rho=-0.1) == 1.0
    assert _dixon_coles_tau(3, 3, 1.5, 1.0, rho=-0.1) == 1.0


def test_dixon_coles_negative_rho_makes_draws_relatively_more_likely():
    """rho negativo (o caso empírico usual em futebol) empurra 0x0 e 1x1 para CIMA
    da independência (tau>1) e 1x0/0x1 para BAIXO (tau<1) — times "seguram" o
    resultado em placares baixos mais do que dois sorteios de Poisson independentes
    preveriam.
    """
    tau_0_0 = _dixon_coles_tau(0, 0, 1.2, 1.0, rho=-0.1)
    tau_1_1 = _dixon_coles_tau(1, 1, 1.2, 1.0, rho=-0.1)
    tau_1_0 = _dixon_coles_tau(1, 0, 1.2, 1.0, rho=-0.1)
    tau_0_1 = _dixon_coles_tau(0, 1, 1.2, 1.0, rho=-0.1)
    assert tau_0_0 > 1.0
    assert tau_1_1 > 1.0
    assert tau_1_0 < 1.0
    assert tau_0_1 < 1.0


def test_score_matrix_sums_to_one_and_matches_outcome_probabilities():
    model = fit_poisson_goals_model(_synthetic_matches(), current_season=2024)
    matrix = model.score_matrix("A", "C")
    assert abs(matrix.sum() - 1.0) < 1e-9

    p_home, p_draw, p_away = model.outcome_probabilities("A", "C")
    assert abs(np.tril(matrix, k=-1).sum() - p_home) < 1e-9
    assert abs(np.trace(matrix) - p_draw) < 1e-9
    assert abs(np.triu(matrix, k=1).sum() - p_away) < 1e-9


def test_fitted_rho_stays_within_optimization_bounds():
    model = fit_poisson_goals_model(_synthetic_matches(), current_season=2024)
    assert -0.3 <= model.rho <= 0.3


def test_zero_rho_reduces_to_independent_poisson():
    model = PoissonGoalsModel(
        league_avg_home_goals=1.5, league_avg_away_goals=1.0, attack={}, defense={}, rho=0.0
    )
    matrix = model.score_matrix("X", "Y")
    lambda_home, lambda_away = model.lambdas("X", "Y")
    from scipy.stats import poisson

    goals = np.arange(matrix.shape[0])
    expected = np.outer(poisson.pmf(goals, lambda_home), poisson.pmf(goals, lambda_away))
    expected /= expected.sum()
    assert np.allclose(matrix, expected)
