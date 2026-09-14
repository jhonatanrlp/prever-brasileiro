import pandas as pd

from src.poisson_goals import PoissonGoalsModel
from src.monte_carlo import simulate_remaining_season
from src.standings import CompetitionRules


def _rules(n_teams: int = 4) -> CompetitionRules:
    return CompetitionRules(
        points_win=3,
        points_draw=1,
        points_loss=0,
        n_teams=n_teams,
        n_relegated=1,
        libertadores_direct=1,
        libertadores_qualifiers=0,
        sulamericana_slots=1,
        tiebreakers=["points", "goal_difference", "goals_for"],
    )


def test_team_with_huge_scoring_advantage_almost_certainly_wins_title():
    current_table = pd.DataFrame(
        [
            {"team_id": "STRONG", "points": 0, "goals_for": 0, "goals_against": 0},
            {"team_id": "WEAK1", "points": 0, "goals_for": 0, "goals_against": 0},
            {"team_id": "WEAK2", "points": 0, "goals_for": 0, "goals_against": 0},
            {"team_id": "WEAK3", "points": 0, "goals_for": 0, "goals_against": 0},
        ]
    )
    remaining = pd.DataFrame(
        [
            {"home_team_id": "STRONG", "away_team_id": "WEAK1"},
            {"home_team_id": "STRONG", "away_team_id": "WEAK2"},
            {"home_team_id": "STRONG", "away_team_id": "WEAK3"},
        ]
    )
    model = PoissonGoalsModel(
        league_avg_home_goals=1.5,
        league_avg_away_goals=1.0,
        attack={"STRONG": 5.0, "WEAK1": 0.05, "WEAK2": 0.05, "WEAK3": 0.05},
        defense={"STRONG": 0.05, "WEAK1": 1.0, "WEAK2": 1.0, "WEAK3": 1.0},
    )
    result = simulate_remaining_season(current_table, remaining, model, _rules(), n_simulations=2000, random_seed=1)
    strong_row = result.set_index("team_id").loc["STRONG"]
    assert strong_row["title_probability"] > 0.95


def test_current_points_are_preserved_as_the_simulation_baseline():
    current_table = pd.DataFrame(
        [
            {"team_id": "A", "points": 40, "goals_for": 30, "goals_against": 10},
            {"team_id": "B", "points": 10, "goals_for": 10, "goals_against": 30},
        ]
    )
    remaining = pd.DataFrame(columns=["home_team_id", "away_team_id"])
    model = PoissonGoalsModel(1.0, 1.0, attack={}, defense={})
    result = simulate_remaining_season(
        current_table, remaining, model, _rules(n_teams=2), n_simulations=100, random_seed=1
    )
    assert result.set_index("team_id").loc["A", "expected_points"] == 40
    assert result.set_index("team_id").loc["A", "title_probability"] == 1.0


def test_probabilities_across_positions_sum_to_one_per_team():
    current_table = pd.DataFrame(
        [
            {"team_id": "A", "points": 10, "goals_for": 10, "goals_against": 10},
            {"team_id": "B", "points": 10, "goals_for": 10, "goals_against": 10},
        ]
    )
    remaining = pd.DataFrame([{"home_team_id": "A", "away_team_id": "B"}])
    model = PoissonGoalsModel(1.3, 1.1, attack={"A": 1.0, "B": 1.0}, defense={"A": 1.0, "B": 1.0})
    result = simulate_remaining_season(
        current_table, remaining, model, _rules(n_teams=2), n_simulations=500, random_seed=1
    )
    total_prob = (
        result["title_probability"] + result["relegation_probability"]
    )
    # Com 2 times, título e rebaixamento são mutuamente exclusivos e cobrem as 2 posições.
    assert all(abs(p - 1.0) < 1e-9 for p in total_prob)
