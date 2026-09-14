import pandas as pd

from src.cbf_calendario import parse_matches


def _sample_round() -> list[dict]:
    return [
        {
            "cbf_id": "1",
            "num_jogo": "1",
            "round": 1,
            "home": "Flamengo",
            "away": "Botafogo",
            "home_goals": "2",
            "away_goals": "1",
            "date": "29/03/2026",
            "time": "16:00",
            "venue": "Maracanã",
        },
        {
            "cbf_id": "2",
            "num_jogo": "2",
            "round": 1,
            "home": "Palmeiras",
            "away": "Santos",
            "home_goals": None,
            "away_goals": None,
            "date": "A Definir",
            "time": "",
            "venue": " -  - ",
        },
    ]


def test_played_flag_reflects_whether_score_is_known():
    df = parse_matches(_sample_round())
    played = df.set_index("home")["played"]
    assert played["Flamengo"]
    assert not played["Palmeiras"]


def test_goals_are_parsed_as_numeric():
    df = parse_matches(_sample_round())
    row = df.set_index("home").loc["Flamengo"]
    assert row["home_goals"] == 2
    assert row["away_goals"] == 1


def test_unplayed_match_has_no_goals():
    df = parse_matches(_sample_round())
    row = df.set_index("home").loc["Palmeiras"]
    assert pd.isna(row["home_goals"])
    assert pd.isna(row["away_goals"])


def test_date_placeholder_becomes_missing_timestamp():
    df = parse_matches(_sample_round())
    row = df.set_index("home").loc["Palmeiras"]
    assert pd.isna(row["date_parsed"])


def test_valid_date_is_parsed_correctly():
    df = parse_matches(_sample_round())
    row = df.set_index("home").loc["Flamengo"]
    assert row["date_parsed"] == pd.Timestamp("2026-03-29")
