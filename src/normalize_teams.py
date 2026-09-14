"""Padronização de nomes de times contra o dicionário mestre em
data/external/team_mapping.csv.

Nunca use `team_name` (string livre) como chave de junção entre fontes distintas —
use sempre `team_id`, resolvido a partir de qualquer grafia conhecida via
`TeamNormalizer.to_team_id`.
"""

from __future__ import annotations

import unicodedata
from pathlib import Path

import pandas as pd

DEFAULT_MAPPING_PATH = Path(__file__).resolve().parents[1] / "data" / "external" / "team_mapping.csv"


def _fold(name: str) -> str:
    """Normaliza uma string para comparação: sem acento, minúscula, sem espaços extras."""
    stripped = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    return " ".join(stripped.lower().split())


class UnknownTeamError(KeyError):
    """Levantado quando um nome de time não é encontrado no dicionário mestre."""


class TeamNormalizer:
    def __init__(self, mapping_path: Path | str = DEFAULT_MAPPING_PATH) -> None:
        self._mapping = pd.read_csv(mapping_path)
        self._lookup: dict[str, str] = {}
        for _, row in self._mapping.iterrows():
            team_id = row["team_id"]
            candidates = [row["team_name"], row["canonical_name"]]
            aliases = row.get("aliases", "")
            if isinstance(aliases, str) and aliases:
                candidates.extend(aliases.split(";"))
            for candidate in candidates:
                self._lookup[_fold(str(candidate))] = team_id

    def to_team_id(self, name: str) -> str:
        key = _fold(name)
        if key not in self._lookup:
            raise UnknownTeamError(
                f"Nome de time desconhecido: {name!r}. "
                "Adicione um alias em data/external/team_mapping.csv."
            )
        return self._lookup[key]

    def normalize_series(self, names: pd.Series) -> pd.Series:
        return names.map(self.to_team_id)

    def canonical_name(self, team_id: str) -> str:
        row = self._mapping.loc[self._mapping["team_id"] == team_id]
        if row.empty:
            raise UnknownTeamError(f"team_id desconhecido: {team_id!r}")
        return row.iloc[0]["canonical_name"]

    @property
    def mapping(self) -> pd.DataFrame:
        return self._mapping.copy()
