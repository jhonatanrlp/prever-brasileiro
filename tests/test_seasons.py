import pandas as pd

from src.seasons import assign_season_ids


def test_splits_seasons_on_large_gap():
    dates = pd.to_datetime(
        [
            "2020-04-01", "2020-04-08", "2020-04-15",
            "2021-04-01", "2021-04-08",
        ]
    )
    seasons = assign_season_ids(pd.Series(dates))
    assert list(seasons) == [2020, 2020, 2020, 2021, 2021]


def test_covid_season_crossing_calendar_year_is_not_split():
    """Caso real: Série A 2020 disputada de agosto/2020 a fevereiro/2021 não pode
    virar duas temporadas só porque cruza o ano-calendário.
    """
    dates = pd.to_datetime(
        [
            "2020-08-08", "2020-09-15", "2020-10-20", "2020-12-01", "2021-01-10", "2021-02-18",  # temporada 2020
            "2021-05-29", "2021-08-01",  # temporada 2021
        ]
    )
    seasons = assign_season_ids(pd.Series(dates))
    assert list(seasons) == [2020, 2020, 2020, 2020, 2020, 2020, 2021, 2021]


def test_unordered_input_is_handled():
    dates = pd.to_datetime(["2021-04-08", "2020-04-01", "2021-04-01", "2020-04-08"])
    seasons = assign_season_ids(pd.Series(dates))
    assert list(seasons) == [2021, 2020, 2021, 2020]
