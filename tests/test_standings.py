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


def _rules_with_full_tiebreakers() -> CompetitionRules:
    rules = _rules()
    rules.tiebreakers = [
        "points",
        "wins",
        "goal_difference",
        "goals_for",
        "head_to_head",
        "fewer_red_cards",
        "fewer_yellow_cards",
    ]
    return rules


def test_head_to_head_breaks_tie_between_exactly_two_teams():
    # A e B empatam em tudo (pontos, vitórias, saldo, gols pró), mas A venceu B no
    # confronto direto — A deve ficar na frente.
    matches = pd.DataFrame(
        [
            {"home_team_id": "A", "away_team_id": "B", "home_goals": 2, "away_goals": 0},
            {"home_team_id": "B", "away_team_id": "A", "home_goals": 0, "away_goals": 0},
            {"home_team_id": "A", "away_team_id": "C", "home_goals": 0, "away_goals": 2},
            {"home_team_id": "B", "away_team_id": "C", "home_goals": 2, "away_goals": 2},
        ]
    )
    table = build_standings(matches, _rules_with_full_tiebreakers())
    a_pos = table.set_index("team_id").loc["A", "position"]
    b_pos = table.set_index("team_id").loc["B", "position"]
    assert a_pos < b_pos


def test_head_to_head_not_applied_to_three_way_tie():
    """Confronto direto só vale "entre duas equipes" — com 3 empatados, deve pular
    direto para o próximo critério (cartões) em vez de tentar decidir por H2H.
    """
    matches = pd.DataFrame(
        [
            {"home_team_id": "A", "away_team_id": "B", "home_goals": 1, "away_goals": 1},
            {"home_team_id": "B", "away_team_id": "C", "home_goals": 1, "away_goals": 1},
            {"home_team_id": "C", "away_team_id": "A", "home_goals": 1, "away_goals": 1},
        ]
    )
    cards = pd.DataFrame(
        [
            {"team_id": "A", "cartao": "Vermelho"},
            {"team_id": "B", "cartao": "Amarelo"},
        ]
    )
    table = build_standings(matches, _rules_with_full_tiebreakers(), cards=cards)
    ranked = list(table["team_id"])
    # A tem cartão vermelho (pior), C não tem nenhum cartão (melhor); B fica no meio.
    assert ranked.index("C") < ranked.index("B") < ranked.index("A")


def test_fewer_cards_breaks_tie_when_no_head_to_head_criterion():
    rules = _rules()
    rules.tiebreakers = ["points", "wins", "goal_difference", "goals_for", "fewer_red_cards"]
    matches = pd.DataFrame(
        [
            {"home_team_id": "A", "away_team_id": "B", "home_goals": 1, "away_goals": 1},
        ]
    )
    cards = pd.DataFrame([{"team_id": "A", "cartao": "Vermelho"}])
    table = build_standings(matches, rules, cards=cards)
    assert table.iloc[0]["team_id"] == "B"  # B não tem cartão vermelho, fica na frente


def test_build_standings_works_without_cards_argument():
    matches = pd.DataFrame(
        [{"home_team_id": "A", "away_team_id": "B", "home_goals": 1, "away_goals": 1}]
    )
    table = build_standings(matches, _rules_with_full_tiebreakers())
    assert len(table) == 2
