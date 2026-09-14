"""Baixa o calendário oficial da CBF (data, rodada e placar por partida) para as
temporadas 2025 e 2026 da Série A (ver src/cbf_calendario.py e docs/DATA_SOURCES.md).

Uso:
    python scripts/fetch_cbf_calendar.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cbf_calendario import CAMPEONATO_IDS, fetch_season  # noqa: E402

RAW_DIR = ROOT / "data" / "raw" / "cbf"


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    metadata = {"source": "cbf.com.br (API não documentada)", "seasons": []}

    for year in sorted(CAMPEONATO_IDS):
        print(f"=== Temporada {year} (campeonato_id={CAMPEONATO_IDS[year]}) ===")
        matches = fetch_season(year)
        out_path = RAW_DIR / f"season_{year}.json"
        out_path.write_text(json.dumps(matches, ensure_ascii=False, indent=2), encoding="utf-8")

        n_played = sum(1 for m in matches if m["home_goals"] is not None)
        print(f"{n_played}/{len(matches)} jogos com placar definido")
        metadata["seasons"].append(
            {
                "season": year,
                "campeonato_id": CAMPEONATO_IDS[year],
                "n_matches": len(matches),
                "n_played": n_played,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
            }
        )

    (RAW_DIR / "_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
