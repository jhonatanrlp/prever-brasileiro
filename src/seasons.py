"""Reconstrução da temporada (season) a partir de datas de partida.

Descoberta da auditoria (ver docs/DATA_SOURCES.md): a temporada NÃO pode ser
derivada do ano-calendário da data, porque a Série A 2020 foi disputada entre
agosto/2020 e fevereiro/2021 (pandemia). Em vez disso, detectamos os intervalos
("off-season") entre uma temporada e a próxima: ordenando as partidas por data,
um gap grande (por padrão 45 dias) entre duas partidas consecutivas marca o fim
de uma temporada e o início da seguinte. Esse método foi validado contra os dados
reais 2003-2024 e produz exatamente uma fronteira por virada de temporada,
inclusive fundindo corretamente o período 2020-08 a 2021-02 em uma única
temporada "2020".
"""

from __future__ import annotations

import pandas as pd

DEFAULT_GAP_DAYS = 45


def assign_season_ids(dates: pd.Series, gap_days: int = DEFAULT_GAP_DAYS) -> pd.Series:
    """Recebe uma Series de datas (não necessariamente ordenada) e retorna, no
    mesmo índice/ordem de entrada, o ano de início da temporada a que cada data
    pertence.
    """
    order = dates.sort_values().index
    sorted_dates = dates.loc[order]
    gap = sorted_dates.diff().dt.days.fillna(0)
    season_break = gap > gap_days
    season_ordinal = season_break.cumsum()

    season_start_year = (
        sorted_dates.groupby(season_ordinal).transform("min").dt.year
    )
    season_start_year.index = order
    return season_start_year.reindex(dates.index)
