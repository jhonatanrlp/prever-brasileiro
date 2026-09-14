from src.temporal import EloConfig, EloState


def test_new_teams_start_at_initial_rating():
    state = EloState(config=EloConfig(initial_rating=1500))
    assert state.get("A") == 1500
    assert state.get("B") == 1500


def test_home_advantage_favors_home_team_at_equal_rating():
    state = EloState(config=EloConfig(home_advantage=60))
    prob = state.expected_home_win_prob("A", "B")
    assert prob > 0.5


def test_winner_rating_increases_and_loser_decreases():
    state = EloState()
    home_before = state.get("A")
    away_before = state.get("B")
    state.update("A", "B", home_goals=2, away_goals=0)
    assert state.get("A") > home_before
    assert state.get("B") < away_before


def test_draw_moves_ratings_toward_each_other_when_home_was_favorite():
    state = EloState(config=EloConfig(home_advantage=200))
    home_before = state.get("A")
    state.update("A", "B", home_goals=1, away_goals=1)
    # Mandante era favorito (vantagem de mando alta) e só empatou: rating cai.
    assert state.get("A") < home_before


def test_ratings_are_read_before_being_updated():
    """Garante que o rating usado numa previsão é sempre o estado PRÉ-jogo:
    chamar update() não deve alterar um valor já lido anteriormente.
    """
    state = EloState()
    pre_match_rating = state.get("A")
    state.update("A", "B", home_goals=3, away_goals=0)
    assert pre_match_rating == 1500.0
    assert state.get("A") != pre_match_rating
