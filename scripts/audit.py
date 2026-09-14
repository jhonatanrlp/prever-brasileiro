"""Auditoria automática do CSV bruto do Campeonato Brasileiro (2003-2024).

Gera um relatório em Markdown com shape, tipos, % de missing, duplicatas,
cobertura temporal, times por temporada e partidas por rodada.

Uso:
    python scripts/audit.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "adaoduque"
REPORT_PATH = Path(__file__).resolve().parents[1] / "outputs" / "data_audit.md"

DATE_FORMAT = "%d/%m/%Y"


def _file_section(label: str, df: pd.DataFrame) -> str:
    n_rows, n_cols = df.shape
    missing = (df.isna().mean() * 100).round(2).sort_values(ascending=False)
    lines = [
        f"### {label}",
        "",
        f"- Shape: {n_rows} linhas x {n_cols} colunas",
        f"- Linhas duplicadas: {int(df.duplicated().sum())}",
        "",
        "| coluna | dtype | % missing | valores únicos |",
        "|---|---|---|---|",
    ]
    for col in df.columns:
        lines.append(f"| {col} | {df[col].dtype} | {missing.get(col, 0.0)} | {df[col].nunique(dropna=True)} |")
    return "\n".join(lines)


def _matches_section(df: pd.DataFrame) -> str:
    bad_dates = int(df["data_parsed"].isna().sum())
    season_counts = df.groupby("season").size()
    matches_per_round = df.groupby(["season", "rodata"]).size().groupby(level=0).agg(["min", "max"])
    impossible_scores = df[(df["mandante_Placar"] < 0) | (df["visitante_Placar"] < 0)]
    teams = pd.unique(df[["mandante", "visitante"]].values.ravel("K"))
    winner_mismatch = df[
        (df["mandante_Placar"] > df["visitante_Placar"]) & (df["vencedor"] != df["mandante"])
        | (df["visitante_Placar"] > df["mandante_Placar"]) & (df["vencedor"] != df["visitante"])
    ]

    lines = [
        "## Auditoria de partidas",
        "",
        f"- Período: {df['data_parsed'].min().date()} a {df['data_parsed'].max().date()}",
        f"- Temporadas distintas: {sorted(season_counts.index.dropna().astype(int).tolist())}",
        f"- Total de partidas: {len(df)}",
        f"- Datas não parseáveis: {bad_dates}",
        f"- Times distintos: {len(teams)}",
        f"- Placares negativos (impossíveis): {len(impossible_scores)}",
        f"- Inconsistências vencedor x placar: {len(winner_mismatch)}",
        "",
        "### Partidas por temporada e por rodada (min/max; esperado ~10 em turno duplo com 20 times)",
        "",
        "| temporada | partidas | min/rodada | max/rodada |",
        "|---|---|---|---|",
    ]
    for season, count in season_counts.items():
        row = matches_per_round.loc[season]
        lines.append(f"| {int(season)} | {count} | {row['min']} | {row['max']} |")
    return "\n".join(lines)


def run_audit() -> Path:
    matches = pd.read_csv(RAW_DIR / "campeonato-brasileiro-full.csv")
    matches["data_parsed"] = pd.to_datetime(matches["data"], format=DATE_FORMAT, errors="coerce")
    matches["season"] = matches["data_parsed"].dt.year

    sections = [
        "# Relatório de auditoria de dados — Campeonato Brasileiro",
        "",
        "Gerado automaticamente por `scripts/audit.py`.",
        "",
        _matches_section(matches),
    ]
    for filename, label in [
        ("campeonato-brasileiro-estatisticas-full.csv", "estatísticas"),
        ("campeonato-brasileiro-gols.csv", "gols"),
        ("campeonato-brasileiro-cartoes.csv", "cartões"),
    ]:
        path = RAW_DIR / filename
        if path.exists():
            sections.append(_file_section(label, pd.read_csv(path)))

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n\n".join(sections), encoding="utf-8")
    return REPORT_PATH


if __name__ == "__main__":
    out = run_audit()
    print(f"Relatório gravado em {out}")
