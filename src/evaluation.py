"""Validação temporal (walk-forward) e métricas probabilísticas.

Nunca `train_test_split` aleatório para avaliar modelos deste projeto: partidas do
mesmo campeonato são correlacionadas no tempo (forma, elenco, técnico), e um split
aleatório vazaria informação futura para o treino. `expanding_window_splits` treina
com temporadas passadas e testa na próxima, avançando uma temporada por vez.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

RESULT_CLASSES = ["H", "D", "A"]


def expanding_window_splits(
    features: pd.DataFrame, season_col: str = "season", min_train_seasons: int = 5
) -> Iterator[tuple[pd.DataFrame, pd.DataFrame, int]]:
    """Gera (treino, teste, temporada_de_teste) para cada temporada a partir da
    (min_train_seasons + 1)-ésima disponível.

    Exemplo com temporadas 2003..2026 e min_train_seasons=5:
        treino=2003-2008, teste=2009
        treino=2003-2009, teste=2010
        ...
        treino=2003-2025, teste=2026
    """
    seasons = sorted(features[season_col].unique())
    if len(seasons) <= min_train_seasons:
        raise ValueError(
            f"Apenas {len(seasons)} temporadas disponíveis; "
            f"min_train_seasons={min_train_seasons} exige pelo menos {min_train_seasons + 1}."
        )

    for i in range(min_train_seasons, len(seasons)):
        test_season = seasons[i]
        train_seasons = seasons[:i]
        train = features[features[season_col].isin(train_seasons)]
        test = features[features[season_col] == test_season]
        yield train, test, test_season


def brier_score_multiclass(y_true: np.ndarray, y_proba: np.ndarray, classes: list[str] = RESULT_CLASSES) -> float:
    """Brier score multiclasse: média do erro quadrático entre a probabilidade
    prevista e o indicador one-hot do resultado real, somado sobre as classes.
    """
    one_hot = np.array([[1.0 if cls == y else 0.0 for cls in classes] for y in y_true])
    return float(np.mean(np.sum((y_proba - one_hot) ** 2, axis=1)))


def ranked_probability_score(y_true: np.ndarray, y_proba: np.ndarray, classes: list[str] = RESULT_CLASSES) -> float:
    """RPS (Epstein 1969): a métrica padrão da literatura de forecasting esportivo
    para resultados com ORDEM natural — aqui, derrota < empate < vitória (a ordem de
    `classes` é usada como a ordem ordinal; RPS é invariante à direção em que essa
    ordem é lida). Ao contrário do log loss/Brier, penaliza menos um erro entre
    classes "vizinhas" (achar que ia empatar quando o mandante venceu) do que um
    erro entre extremos (achar que o mandante venceria quando o visitante venceu) —
    o log loss e o Brier tratam as 3 classes como não-ordenadas e não fazem essa
    distinção.
    """
    one_hot = np.array([[1.0 if cls == y else 0.0 for cls in classes] for y in y_true])
    cum_proba = np.cumsum(y_proba, axis=1)
    cum_true = np.cumsum(one_hot, axis=1)
    n_classes = len(classes)
    return float(np.mean(np.sum((cum_proba - cum_true) ** 2, axis=1) / (n_classes - 1)))


def evaluate_probabilistic(
    y_true: np.ndarray, y_proba: np.ndarray, classes: list[str] = RESULT_CLASSES
) -> dict[str, float]:
    y_pred = [classes[i] for i in np.argmax(y_proba, axis=1)]

    # sklearn.metrics.log_loss SEMPRE lê as colunas de y_proba em ordem alfabética
    # das classes, não na ordem passada em `labels` (`labels` só valida quais
    # classes existem) — por isso reordenamos as colunas explicitamente para a
    # ordem alfabética antes de chamar. Sem isso, o log_loss sai calculado com as
    # colunas trocadas (bug real, não só um aviso cosmético).
    sorted_classes = sorted(classes)
    column_order = [classes.index(c) for c in sorted_classes]
    loss = float(log_loss(y_true, y_proba[:, column_order], labels=sorted_classes))

    return {
        "log_loss": loss,
        "brier_score": brier_score_multiclass(y_true, y_proba, classes),
        "rps": ranked_probability_score(y_true, y_proba, classes),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }
