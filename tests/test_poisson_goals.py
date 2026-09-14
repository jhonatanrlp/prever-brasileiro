import numpy as np
import pandas as pd

from src.poisson_goals import (
    PoissonGoalsModel,
    _dixon_coles_tau,
    _dixon_coles_tau_vec,
    fit_poisson_goals_model,
    fit_poisson_goals_model_mle,
)


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


def test_top_scores_returns_k_entries_sorted_descending():
    model = fit_poisson_goals_model(_synthetic_matches(), current_season=2024)
    top5 = model.top_scores("A", "C", k=5)
    assert len(top5) == 5
    probs = [s["probability"] for s in top5]
    assert probs == sorted(probs, reverse=True)


def test_top_scores_coverage_is_between_zero_and_one_and_less_than_full_mass():
    model = fit_poisson_goals_model(_synthetic_matches(), current_season=2024)
    top5 = model.top_scores("A", "C", k=5)
    coverage = model.top_scores_coverage(top5)
    assert 0.0 < coverage <= 1.0
    # 5 de 121 células plausíveis não deveriam cobrir a massa inteira.
    assert coverage < 1.0


def test_top_scores_matches_score_matrix_values():
    model = fit_poisson_goals_model(_synthetic_matches(), current_season=2024)
    matrix = model.score_matrix("A", "C")
    top1 = model.top_scores("A", "C", k=1)[0]
    assert matrix[top1["home_goals"], top1["away_goals"]] == matrix.max()
    assert abs(top1["probability"] - matrix.max()) < 1e-12


def test_dixon_coles_tau_vec_matches_scalar_version():
    home_goals = np.array([0, 0, 1, 1, 2])
    away_goals = np.array([0, 1, 0, 1, 2])
    lambda_home = np.array([1.2, 1.2, 1.2, 1.2, 1.2])
    lambda_away = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    vec = _dixon_coles_tau_vec(home_goals, away_goals, lambda_home, lambda_away, rho=-0.1)
    scalar = [
        _dixon_coles_tau(h, a, lh, la, -0.1)
        for h, a, lh, la in zip(home_goals, away_goals, lambda_home, lambda_away)
    ]
    assert np.allclose(vec, scalar)


def test_mle_joint_fit_converges_and_produces_valid_probabilities():
    model = fit_poisson_goals_model_mle(_synthetic_matches(), current_season=2024)
    assert model.converged is True
    assert model.n_iterations is not None and model.n_iterations > 0
    assert model.log_likelihood is not None
    assert model.aic is not None and model.bic is not None

    p_home, p_draw, p_away = model.outcome_probabilities("A", "C")
    assert abs((p_home + p_draw + p_away) - 1.0) < 1e-9


def test_mle_joint_fit_preserves_relative_team_strength_ordering():
    """A ataca muito e defende bem, C é o oposto — a MLE conjunta deve concordar
    com o método dos momentos na ORDEM relativa das forças, mesmo ajustando os
    parâmetros de um jeito diferente.
    """
    model = fit_poisson_goals_model_mle(_synthetic_matches(), current_season=2024)
    assert model.attack["A"] > model.attack["B"] > model.attack["C"]
    assert model.defense["A"] < model.defense["B"] < model.defense["C"]


def test_mle_joint_fit_does_not_diverge_on_a_team_that_never_scores():
    """Caso degenerado: sem regularização, a restrição de identificabilidade
    (soma dos log-efeitos de ataque = 0) faz o ataque de um time que nunca marca
    empurrar os outros para o infinito. A penalização ridge em `_fit_dixon_coles_mle`
    existe exatamente para evitar isso.
    """
    matches = pd.DataFrame(
        [
            {"season": 2024, "home_team_id": "A", "away_team_id": "C", "home_goals": 3, "away_goals": 0},
            {"season": 2024, "home_team_id": "C", "away_team_id": "A", "home_goals": 0, "away_goals": 4},
            {"season": 2024, "home_team_id": "B", "away_team_id": "C", "home_goals": 2, "away_goals": 0},
            {"season": 2024, "home_team_id": "C", "away_team_id": "B", "home_goals": 0, "away_goals": 2},
        ]
    )
    model = fit_poisson_goals_model_mle(matches, current_season=2024)
    assert all(np.isfinite(v) and abs(v) < 50 for v in model.attack.values())
    assert all(np.isfinite(v) and abs(v) < 50 for v in model.defense.values())


def test_mle_and_method_of_moments_agree_on_rho_sign_and_magnitude_order():
    """Não exigimos que os dois métodos deem o mesmo rho exato (são ajustes
    diferentes), só que fiquem na mesma faixa geral — divergência grande indicaria
    um bug em algum dos dois.
    """
    old = fit_poisson_goals_model(_synthetic_matches(), current_season=2024)
    new = fit_poisson_goals_model_mle(_synthetic_matches(), current_season=2024)
    assert abs(old.rho - new.rho) < 0.3


def test_old_fit_function_still_has_no_diagnostic_fields():
    """Trava a compatibilidade: quem já usava fit_poisson_goals_model (predict_current.py,
    backtest_2026.py) não deve ver comportamento novo — os campos de diagnóstico só
    existem na MLE conjunta.
    """
    model = fit_poisson_goals_model(_synthetic_matches(), current_season=2024)
    assert model.log_likelihood is None
    assert model.aic is None
    assert model.bic is None
    assert model.converged is None
