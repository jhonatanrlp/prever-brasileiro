"""Download automatizado do histórico 2003-2024 do Brasileirão (fonte: adaoduque/Brasileirao_Dataset).

Uso:
    python scripts/download_data.py [--force]

Baixa cada arquivo para data/raw/adaoduque/, grava metadados de origem
(_metadata.json) e evita rebaixar um arquivo cujo conteúdo (hash sha256) não mudou,
a menos que --force seja passado. Ver docs/DATA_SOURCES.md para por que essa é a
fonte escolhida.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("download_data")

BASE_URL = "https://raw.githubusercontent.com/adaoduque/Brasileirao_Dataset/master"
FILENAMES = [
    "campeonato-brasileiro-full.csv",
    "campeonato-brasileiro-estatisticas-full.csv",
    "campeonato-brasileiro-gols.csv",
    "campeonato-brasileiro-cartoes.csv",
    "Legenda.txt",
]

DEST_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "adaoduque"
TIMEOUT = 30.0
MAX_RETRIES = 3


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _fetch(client: httpx.Client, url: str) -> bytes:
    last_exc: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.get(url, timeout=TIMEOUT, follow_redirects=True)
            response.raise_for_status()
            return response.content
        except (httpx.HTTPError, httpx.TransportError) as exc:
            last_exc = exc
            logger.warning("Tentativa %d/%d falhou para %s: %s", attempt, MAX_RETRIES, url, exc)
            time.sleep(1.5 * attempt)
    raise RuntimeError(f"Falha ao baixar {url} após {MAX_RETRIES} tentativas") from last_exc


def download_file(client: httpx.Client, filename: str, force: bool) -> dict:
    url = f"{BASE_URL}/{filename}"
    dest_path = DEST_DIR / filename
    existing_hash = _sha256(dest_path.read_bytes()) if dest_path.exists() else None

    try:
        content = _fetch(client, url)
    except RuntimeError as exc:
        logger.error(str(exc))
        return {"filename": filename, "url": url, "status": "error", "error": str(exc)}

    new_hash = _sha256(content)
    if not force and existing_hash == new_hash:
        logger.info("Sem mudanças, mantendo arquivo existente: %s", dest_path)
        status = "unchanged"
    else:
        DEST_DIR.mkdir(parents=True, exist_ok=True)
        dest_path.write_bytes(content)
        status = "updated" if existing_hash else "downloaded"
        logger.info("%s (%d bytes) -> %s", status, len(content), dest_path)

    return {"filename": filename, "url": url, "status": status, "sha256": new_hash, "size_bytes": len(content)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Rebaixar mesmo se o conteúdo não mudou")
    args = parser.parse_args()

    DEST_DIR.mkdir(parents=True, exist_ok=True)
    with httpx.Client(headers={"User-Agent": "prever-brasileiro-downloader/0.1"}) as client:
        results = [download_file(client, name, args.force) for name in FILENAMES]

    (DEST_DIR / "_metadata.json").write_text(
        json.dumps(
            {"source": "adaoduque/Brasileirao_Dataset", "downloaded_at": datetime.now(timezone.utc).isoformat(), "files": results},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    if any(r["status"] == "error" for r in results):
        logger.error("Uma ou mais partidas falharam no download. Veja os logs acima.")
        return 1

    logger.info("Download concluído com sucesso.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
