import pandas as pd
import pytest

from src.normalize_teams import TeamNormalizer, UnknownTeamError


@pytest.fixture
def mapping_csv(tmp_path):
    path = tmp_path / "team_mapping.csv"
    pd.DataFrame(
        [
            {
                "team_id": "BRA001",
                "team_name": "Athletico-PR",
                "canonical_name": "Athletico Paranaense",
                "state": "PR",
                "city": "",
                "aliases": "Atletico-PR;Atlético-PR;CAP",
                "first_season": 2003,
                "last_season": 2024,
            },
            {
                "team_id": "BRA002",
                "team_name": "Flamengo",
                "canonical_name": "Flamengo",
                "state": "RJ",
                "city": "",
                "aliases": "CR Flamengo",
                "first_season": 2003,
                "last_season": 2024,
            },
        ]
    ).to_csv(path, index=False)
    return path


def test_resolves_exact_team_name(mapping_csv):
    normalizer = TeamNormalizer(mapping_csv)
    assert normalizer.to_team_id("Flamengo") == "BRA002"


def test_resolves_alias_case_and_accent_insensitive(mapping_csv):
    normalizer = TeamNormalizer(mapping_csv)
    assert normalizer.to_team_id("atlético-pr") == "BRA001"
    assert normalizer.to_team_id("CAP") == "BRA001"


def test_unknown_team_raises(mapping_csv):
    normalizer = TeamNormalizer(mapping_csv)
    with pytest.raises(UnknownTeamError):
        normalizer.to_team_id("Time Inexistente FC")


def test_normalize_series(mapping_csv):
    normalizer = TeamNormalizer(mapping_csv)
    result = normalizer.normalize_series(pd.Series(["Flamengo", "CAP"]))
    assert list(result) == ["BRA002", "BRA001"]
