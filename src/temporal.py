"""Estado por time (Elo + forma) e construção de features pré-partida, sem
vazamento temporal.

`EloState` e `FormState` são atualizados partida a partida, em ordem cronológica:
seus métodos de leitura (`get`, `snapshot`, `expected_home_win_prob`) só veem o
estado ANTES da partida; `update` é quem aplica o resultado revelado e avança o
estado. `build_pre_match_features` percorre as partidas cronologicamente lendo o
estado antes de cada uma e só então atualizando — isso garante estruturalmente que
a feature da partida na posição i depende apenas das partidas 0..i-1.

Não crie datasets por rodada manualmente — sempre passe o histórico completo de
partidas (já ordenado ou não; a função ordena internamente) por
`build_pre_match_features`.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass, field

import pandas as pd

FORM_WINDOWS = (3, 5, 10)
POINTS = {"W": 3, "D": 1, "L": 0}


@dataclass
class EloConfig:
    initial_rating: float = 1500.0
    home_advantage: float = 60.0
    k_factor: float = 20.0
    goal_diff_multiplier: bool = True


@dataclass
class EloState:
    config: EloConfig = field(default_factory=EloConfig)
    ratings: dict[str, float] = field(default_factory=dict)

    def get(self, team_id: str) -> float:
        return self.ratings.get(team_id, self.config.initial_rating)

    def expected_home_win_prob(self, home_team_id: str, away_team_id: str) -> float:
        home_rating = self.get(home_team_id) + self.config.home_advantage
        away_rating = self.get(away_team_id)
        return 1.0 / (1.0 + 10 ** ((away_rating - home_rating) / 400.0))

    def update(self, home_team_id: str, away_team_id: str, home_goals: int, away_goals: int) -> None:
        expected_home = self.expected_home_win_prob(home_team_id, away_team_id)
        if home_goals > away_goals:
            actual_home = 1.0
        elif home_goals < away_goals:
            actual_home = 0.0
        else:
            actual_home = 0.5

        k = self.config.k_factor
        if self.config.goal_diff_multiplier:
            margin = abs(home_goals - away_goals)
            # Multiplicador estilo FiveThirtyEight: vitórias por margem maior pesam
            # mais, com retornos decrescentes (log), para goleadas não dominarem o rating.
            k *= math.log(margin + 1) + 1.0 if margin > 0 else 1.0

        delta = k * (actual_home - expected_home)
        self.ratings[home_team_id] = self.get(home_team_id) + delta
        self.ratings[away_team_id] = self.get(away_team_id) - delta


@dataclass
class TeamRecord:
    matches_played: int = 0
    points: int = 0
    goals_for: int = 0
    goals_against: int = 0
    home_matches_played: int = 0
    home_points: int = 0
    home_goals_for: int = 0
    home_goals_against: int = 0
    away_matches_played: int = 0
    away_points: int = 0
    away_goals_for: int = 0
    away_goals_against: int = 0
    recent_results: deque = field(default_factory=lambda: deque(maxlen=10))


@dataclass
class FormState:
    records: dict[str, TeamRecord] = field(default_factory=lambda: defaultdict(TeamRecord))

    def get(self, team_id: str) -> TeamRecord:
        return self.records[team_id]

    def snapshot(self, team_id: str, form_windows: tuple[int, ...] = FORM_WINDOWS) -> dict:
        rec = self.get(team_id)
        if rec.matches_played == 0:
            return _empty_snapshot(form_windows)

        snapshot = {
            "matches_played": rec.matches_played,
            "points_per_game": rec.points / rec.matches_played,
            "goal_diff_per_game": (rec.goals_for - rec.goals_against) / rec.matches_played,
            "goals_for_per_game": rec.goals_for / rec.matches_played,
            "goals_against_per_game": rec.goals_against / rec.matches_played,
            "home_points_per_game": (
                rec.home_points / rec.home_matches_played if rec.home_matches_played else None
            ),
            "home_goals_for_per_game": (
                rec.home_goals_for / rec.home_matches_played if rec.home_matches_played else None
            ),
            "home_goals_against_per_game": (
                rec.home_goals_against / rec.home_matches_played if rec.home_matches_played else None
            ),
            "away_points_per_game": (
                rec.away_points / rec.away_matches_played if rec.away_matches_played else None
            ),
            "away_goals_for_per_game": (
                rec.away_goals_for / rec.away_matches_played if rec.away_matches_played else None
            ),
            "away_goals_against_per_game": (
                rec.away_goals_against / rec.away_matches_played if rec.away_matches_played else None
            ),
        }
        for window in form_windows:
            recent = list(rec.recent_results)[-window:]
            snapshot[f"form_points_last_{window}"] = (
                sum(POINTS[r] for r in recent) / len(recent) if recent else None
            )
        return snapshot

    def update(self, home_team_id: str, away_team_id: str, home_goals: int, away_goals: int) -> None:
        home = self.get(home_team_id)
        away = self.get(away_team_id)

        home.matches_played += 1
        away.matches_played += 1
        home.goals_for += home_goals
        home.goals_against += away_goals
        away.goals_for += away_goals
        away.goals_against += home_goals

        home.home_matches_played += 1
        home.home_goals_for += home_goals
        home.home_goals_against += away_goals
        away.away_matches_played += 1
        away.away_goals_for += away_goals
        away.away_goals_against += home_goals

        if home_goals > away_goals:
            home_result, away_result = "W", "L"
        elif home_goals < away_goals:
            home_result, away_result = "L", "W"
        else:
            home_result = away_result = "D"

        home.points += POINTS[home_result]
        away.points += POINTS[away_result]
        home.home_points += POINTS[home_result]
        away.away_points += POINTS[away_result]
        home.recent_results.append(home_result)
        away.recent_results.append(away_result)


def _empty_snapshot(form_windows: tuple[int, ...]) -> dict:
    base = {
        "matches_played": 0,
        "points_per_game": None,
        "goal_diff_per_game": None,
        "goals_for_per_game": None,
        "goals_against_per_game": None,
        "home_points_per_game": None,
        "home_goals_for_per_game": None,
        "home_goals_against_per_game": None,
        "away_points_per_game": None,
        "away_goals_for_per_game": None,
        "away_goals_against_per_game": None,
    }
    for window in form_windows:
        base[f"form_points_last_{window}"] = None
    return base


def build_pre_match_features(
    matches: pd.DataFrame,
    elo_config: EloConfig | None = None,
) -> pd.DataFrame:
    """Recebe o DataFrame de `matches` (colunas: match_id, date, home_team_id,
    away_team_id, home_goals, away_goals) e retorna um DataFrame com uma linha por
    partida, na mesma ordem cronológica, contendo as features conhecidas
    imediatamente ANTES daquela partida acontecer.
    """
    required = {"match_id", "date", "home_team_id", "away_team_id", "home_goals", "away_goals"}
    missing = required - set(matches.columns)
    if missing:
        raise ValueError(f"Colunas obrigatórias ausentes em matches: {missing}")

    passthrough_cols = [c for c in ("season", "round") if c in matches.columns]
    ordered = matches.sort_values(["date", "match_id"]).reset_index(drop=True)

    elo = EloState(config=elo_config or EloConfig())
    form = FormState()
    opponent_elo_sum: dict[str, float] = {}
    opponent_elo_count: dict[str, int] = {}
    rows: list[dict] = []

    for row in ordered.itertuples(index=False):
        home_id = row.home_team_id
        away_id = row.away_team_id

        home_elo_pre = elo.get(home_id)
        away_elo_pre = elo.get(away_id)
        home_form = form.snapshot(home_id)
        away_form = form.snapshot(away_id)

        opponent_strength_home = _average_opponent_elo(opponent_elo_sum, opponent_elo_count, home_id)
        opponent_strength_away = _average_opponent_elo(opponent_elo_sum, opponent_elo_count, away_id)

        feature_row = {
            "match_id": row.match_id,
            "date": row.date,
            **{col: getattr(row, col) for col in passthrough_cols},
            "home_team_id": home_id,
            "away_team_id": away_id,
            "home_elo_pre": home_elo_pre,
            "away_elo_pre": away_elo_pre,
            "elo_diff": home_elo_pre - away_elo_pre,
            "home_win_prob_elo": elo.expected_home_win_prob(home_id, away_id),
            "home_strength_of_schedule": opponent_strength_home,
            "away_strength_of_schedule": opponent_strength_away,
        }
        for key, value in home_form.items():
            feature_row[f"home_{key}"] = value
        for key, value in away_form.items():
            feature_row[f"away_{key}"] = value

        feature_row["attack_vs_defense_home"] = _safe_diff(
            home_form["goals_for_per_game"], away_form["goals_against_per_game"]
        )
        feature_row["attack_vs_defense_away"] = _safe_diff(
            away_form["goals_for_per_game"], home_form["goals_against_per_game"]
        )
        feature_row["form_diff_last_5"] = _safe_diff(
            home_form["form_points_last_5"], away_form["form_points_last_5"]
        )

        # Alvos de treino (só usar em treino; nunca como feature de inferência).
        feature_row["target_result"] = row.result if hasattr(row, "result") else _result(
            row.home_goals, row.away_goals
        )
        feature_row["target_home_goals"] = row.home_goals
        feature_row["target_away_goals"] = row.away_goals

        rows.append(feature_row)

        # Registra, para cada time, o Elo pré-jogo do adversário ENFRENTADO nesta
        # partida — usado para calcular a força média dos adversários já
        # enfrentados nas próximas partidas (strength-of-schedule).
        opponent_elo_sum[home_id] = opponent_elo_sum.get(home_id, 0.0) + away_elo_pre
        opponent_elo_count[home_id] = opponent_elo_count.get(home_id, 0) + 1
        opponent_elo_sum[away_id] = opponent_elo_sum.get(away_id, 0.0) + home_elo_pre
        opponent_elo_count[away_id] = opponent_elo_count.get(away_id, 0) + 1

        elo.update(home_id, away_id, row.home_goals, row.away_goals)
        form.update(home_id, away_id, row.home_goals, row.away_goals)

    return pd.DataFrame(rows)


def _result(home_goals: int, away_goals: int) -> str:
    if home_goals > away_goals:
        return "H"
    if home_goals < away_goals:
        return "A"
    return "D"


def _safe_diff(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return a - b


def _average_opponent_elo(
    elo_sum: dict[str, float], count: dict[str, int], team_id: str
) -> float | None:
    n = count.get(team_id, 0)
    if n == 0:
        return None
    return elo_sum[team_id] / n
