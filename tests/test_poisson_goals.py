import pandas as pd

from src.poisson_goals import fit_poisson_goals_model


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
