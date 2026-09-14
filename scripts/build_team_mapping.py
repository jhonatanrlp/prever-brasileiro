"""Gera data/external/team_mapping.csv a partir dos nomes reais encontrados em
data/raw/adaoduque/campeonato-brasileiro-full.csv, mais um dicionário curado de
nomes canônicos e aliases conhecidos (grafias usadas por outras fontes: Wikipedia,
Sofascore, API-Football, imprensa).

Uso:
    python scripts/build_team_mapping.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RAW_MATCHES = ROOT / "data" / "raw" / "adaoduque" / "campeonato-brasileiro-full.csv"
OUT_PATH = ROOT / "data" / "external" / "team_mapping.csv"

# team_name (grafia exata da fonte principal) -> (canonical_name, aliases conhecidos)
CANONICAL = {
    "America-MG": ("América Mineiro", "America Mineiro;América-MG;America FC"),
    "America-RN": ("América de Natal", "América-RN;America Natal"),
    "Athletico-PR": ("Athletico Paranaense", "Atletico-PR;Atlético-PR;Atlético Paranaense;Athletico Paranaense;CAP"),
    "Atletico-GO": ("Atlético Goianiense", "Atletico Goianiense;Atlético-GO"),
    "Atletico-MG": ("Atlético Mineiro", "Atletico Mineiro;Atlético-MG;Galo"),
    "Avai": ("Avaí", "Avai FC"),
    "Bahia": ("Bahia", "EC Bahia"),
    "Barueri": ("Grêmio Barueri", "Gremio Barueri"),
    "Botafogo-RJ": ("Botafogo", "Botafogo FR;Botafogo RJ;Botafogo"),
    "Bragantino": ("Red Bull Bragantino", "RB Bragantino;Bragantino-SP"),
    "Brasiliense": ("Brasiliense", "Brasiliense FC"),
    "CSA": ("CSA", "Centro Sportivo Alagoano"),
    "Ceara": ("Ceará", "Ceara SC"),
    "Chapecoense": ("Chapecoense", "Chapecoense-SC;ACF"),
    "Corinthians": ("Corinthians", "SC Corinthians Paulista"),
    "Coritiba": ("Coritiba", "Coritiba FC;Coxa;Coritiba SAF"),
    "Criciuma": ("Criciúma", "Criciuma EC"),
    "Cruzeiro": ("Cruzeiro", "Cruzeiro EC"),
    "Cuiaba": ("Cuiabá", "Cuiaba EC"),
    "Figueirense": ("Figueirense", "Figueirense FC"),
    "Flamengo": ("Flamengo", "CR Flamengo;Mengo"),
    "Fluminense": ("Fluminense", "Fluminense FC;Flu"),
    "Fortaleza": ("Fortaleza", "Fortaleza EC;Fortaleza SAF"),
    "Goias": ("Goiás", "Goias EC"),
    "Gremio": ("Grêmio", "Gremio FBPA"),
    "Gremio Prudente": ("Grêmio Prudente", "Gremio Barueri Prudente"),
    "Guarani": ("Guarani", "Guarani FC"),
    "Internacional": ("Internacional", "SC Internacional;Inter;Internacional-RS"),
    "Ipatinga": ("Ipatinga", "Ipatinga FC"),
    "Joinville": ("Joinville", "Joinville EC"),
    "Juventude": ("Juventude", "EC Juventude"),
    "Nautico": ("Náutico", "Nautico Capibaribe"),
    "Palmeiras": ("Palmeiras", "SE Palmeiras;Verdao"),
    "Parana": ("Paraná Clube", "Parana Clube"),
    "Paysandu": ("Paysandu", "Paysandu SC"),
    "Ponte Preta": ("Ponte Preta", "AA Ponte Preta"),
    "Portuguesa": ("Portuguesa", "Associação Portuguesa de Desportos"),
    "Santa Cruz": ("Santa Cruz", "Santa Cruz FC"),
    "Santo Andre": ("Santo André", "EC Santo Andre"),
    "Santos": ("Santos", "Santos FC"),  # já cobre "Santos FC" da API da CBF
    "Sao Caetano": ("São Caetano", "AA São Caetano;Sao Caetano"),
    "Sao Paulo": ("São Paulo", "Sao Paulo FC;SPFC"),
    "Sport": ("Sport Recife", "Sport Club do Recife;Sport-PE"),
    "Vasco": ("Vasco da Gama", "CR Vasco da Gama;Vasco da Gama Saf"),
    "Vitoria": ("Vitória", "EC Vitoria"),
}


# Times que estreiam na Série A a partir de 2025/2026 e por isso não aparecem no
# histórico 2003-2024 da fonte principal. Adicionados manualmente porque não há
# como derivá-los do CSV histórico — ver docs/DATA_SOURCES.md.
NEW_TEAMS_2025_PLUS = {
    "Mirassol": {
        "canonical_name": "Mirassol",
        "state": "SP",
        "aliases": "Mirassol FC;MIR",
        "first_season": 2025,
    },
    "Remo": {
        "canonical_name": "Remo",
        "state": "PA",
        "aliases": "Clube do Remo;REM",
        "first_season": 2026,
    },
}


def build() -> pd.DataFrame:
    df = pd.read_csv(RAW_MATCHES)
    df["data_parsed"] = pd.to_datetime(df["data"], format="%d/%m/%Y")
    long = pd.concat(
        [
            df[["mandante", "mandante_Estado", "data_parsed"]].rename(
                columns={"mandante": "team", "mandante_Estado": "state"}
            ),
            df[["visitante", "visitante_Estado", "data_parsed"]].rename(
                columns={"visitante": "team", "visitante_Estado": "state"}
            ),
        ]
    )
    agg = long.groupby("team").agg(
        first_season=("data_parsed", lambda s: int(s.min().year)),
        last_season=("data_parsed", lambda s: int(s.max().year)),
        state=("state", lambda s: s.mode().iat[0]),
    )

    rows = []
    for idx, (team_name, row) in enumerate(agg.sort_index().iterrows(), start=1):
        canonical_name, aliases = CANONICAL.get(team_name, (team_name, ""))
        rows.append(
            {
                "team_id": f"BRA{idx:03d}",
                "team_name": team_name,
                "canonical_name": canonical_name,
                "state": row["state"],
                "city": "",
                "aliases": aliases,
                "first_season": row["first_season"],
                "last_season": row["last_season"],
            }
        )

    next_idx = len(rows) + 1
    for team_name, info in NEW_TEAMS_2025_PLUS.items():
        rows.append(
            {
                "team_id": f"BRA{next_idx:03d}",
                "team_name": team_name,
                "canonical_name": info["canonical_name"],
                "state": info["state"],
                "city": "",
                "aliases": info["aliases"],
                "first_season": info["first_season"],
                "last_season": info["first_season"],
            }
        )
        next_idx += 1

    return pd.DataFrame(rows)


if __name__ == "__main__":
    out = build()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"{len(out)} times gravados em {OUT_PATH}")
