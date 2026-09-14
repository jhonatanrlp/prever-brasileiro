import pandas as pd

from src.standings import CompetitionRules, build_standings


def _rules() -> CompetitionRules:
    return CompetitionRules(
        points_win=3,
        points_draw=1,
        points_loss=0,
        n_teams=4,
        n_relegated=1,
        libertadores_direct=1,
        libertadores_qualifiers=0,
        sulamericana_slots=1,
        tiebreakers=["points", "wins", "goal_difference", "goals_for"],
    )


def test_points_awarded_correctly():
    matches = pd.DataFrame(
        [
            {"home_team_id": "A", "away_team_id": "B", "home_goals": 2, "away_goals": 0},
            {"home_team_id": "B", "away_team_id": "A", "home_goals": 1, "away_goals": 1},
        ]
    )
    table = build_standings(matches, _rules())
    a = table.set_index("team_id").loc["A"]
    b = table.set_index("team_id").loc["B"]
    assert a["points"] == 4
    assert b["points"] == 1


def test_tiebreak_by_goal_difference():
    matches = pd.DataFrame(
        [
            {"home_team_id": "A", "away_team_id": "C", "home_goals": 5, "away_goals": 0},
            {"home_team_id": "B", "away_team_id": "D", "home_goals": 1, "away_goals": 0},
        ]
    )
    table = build_standings(matches, _rules())
    assert table.iloc[0]["team_id"] == "A"  # mesmo com 3 pontos como B, tem SG melhor


def test_position_is_assigned_1_indexed():
    matches = pd.DataFrame(
        [{"home_team_id": "A", "away_team_id": "B", "home_goals": 1, "away_goals": 0}]
    )
    table = build_standings(matches, _rules())
    assert list(table["position"]) == [1, 2]
