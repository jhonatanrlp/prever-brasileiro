"""Testes que travam a garantia central do projeto: nenhuma feature de uma
partida pode depender de resultados dessa mesma partida ou de partidas futuras.
"""

import pandas as pd

from src.temporal import build_pre_match_features


def _toy_matches() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"match_id": 1, "date": "2024-01-01", "home_team_id": "A", "away_team_id": "B", "home_goals": 3, "away_goals": 0},
            {"match_id": 2, "date": "2024-01-08", "home_team_id": "B", "away_team_id": "A", "home_goals": 1, "away_goals": 1},
            {"match_id": 3, "date": "2024-01-15", "home_team_id": "A", "away_team_id": "B", "home_goals": 0, "away_goals": 2},
        ]
    ).assign(date=lambda d: pd.to_datetime(d["date"]))


def test_first_match_has_no_history():
    features = build_pre_match_features(_toy_matches())
    first = features.iloc[0]
    assert first["home_matches_played"] == 0
    assert first["away_matches_played"] == 0
    assert pd.isna(first["home_points_per_game"])
    assert first["home_elo_pre"] == 1500.0


def test_second_match_only_reflects_first_match_result():
    features = build_pre_match_features(_toy_matches())
    second = features.iloc[1]
    # Na 2a partida, B (agora mandante) já jogou 1 partida (perdeu por 3x0 fora).
    assert second["home_matches_played"] == 1  # home_team_id da partida 2 é "B"
    assert second["home_points_per_game"] == 0.0
    assert second["home_goals_against_per_game"] == 3.0


def test_feature_row_never_encodes_its_own_result_in_pre_match_fields():
    """A 3a partida termina 0x2 (A perde). As features pré-jogo de A não podem
    já refletir esse resultado — devem refletir só as partidas 1 e 2.
    """
    features = build_pre_match_features(_toy_matches())
    third = features.iloc[2]
    # A jogou as partidas 1 (venceu 3x0 fora... não, A é mandante na 1) e 2 (empatou fora).
    # Antes da partida 3, A tem 2 jogos: vitória (3x0) e empate (1x1) => 4 pontos em 2 jogos.
    assert third["home_matches_played"] == 2
    assert third["home_points_per_game"] == 2.0


def test_walk_forward_is_deterministic_and_chronological():
    shuffled = _toy_matches().sample(frac=1, random_state=0)
    features_from_shuffled = build_pre_match_features(shuffled)
    features_from_ordered = build_pre_match_features(_toy_matches())
    pd.testing.assert_frame_equal(
        features_from_shuffled.reset_index(drop=True),
        features_from_ordered.reset_index(drop=True),
    )
