"""Ingestão da API pública de calendário/jogos da CBF (cbf.com.br).

Descoberta (2026-09-14, inspecionando as chamadas de rede que o próprio site da CBF
faz ao renderizar a página de tabelas): existe um endpoint JSON não documentado,
`GET /api/cbf/jogos/campeonato/{campeonato_id}/rodada/{n}/fase`, que devolve TODAS as
partidas de uma rodada de um campeonato, com placar, data, hora e local (ver
docs/DATA_SOURCES.md).

`campeonato_id` da Série A: 12606 (2025), 1260611 (2026). Descobertos navegando o
site e inspecionando as chamadas de API — não há uma forma documentada de listá-los;
se a CBF mudar esses ids em anos futuros, será preciso redescobri-los da mesma forma
(abrir a tabela do ano em um navegador e olhar as chamadas de rede para
`/api/cbf/jogos/campeonato/`).

Nota de segurança sobre TLS: o servidor da CBF serve uma cadeia de certificado
incompleta (falta um intermediário), que navegadores toleram porque buscam o
intermediário faltante via AIA, mas o OpenSSL usado por `httpx` não faz isso e
rejeita a conexão com CERTIFICATE_VERIFY_FAILED. Por isso as requisições aqui usam
`verify=False`. Isso é aceitável neste caso porque (a) é uma leitura pública, sem
credenciais nem dados sensíveis trafegando, e (b) o problema é a cadeia de
certificado do servidor, não a ausência de HTTPS — mas é uma troca deliberada, não
uma prática a copiar sem pensar para outra integração.
"""

from __future__ import annotations

import logging
import time

import httpx
import pandas as pd

logger = logging.getLogger(__name__)

BASE_URL = "https://www.cbf.com.br/api/cbf/jogos/campeonato/{campeonato_id}/rodada/{round_number}/fase"
USER_AGENT = "prever-brasileiro-bot/0.1 (contato: jhonatanramosleite21@gmail.com; uso academico/portfolio)"
TIMEOUT = 30.0
MAX_RETRIES = 3

CAMPEONATO_IDS = {2025: "12606", 2026: "1260611"}
N_ROUNDS = 38


def fetch_round(campeonato_id: str, round_number: int) -> list[dict]:
    url = BASE_URL.format(campeonato_id=campeonato_id, round_number=round_number)
    last_exc: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = httpx.get(
                url, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT, verify=False
            )
            response.raise_for_status()
            payload = response.json()
            break
        except (httpx.HTTPError, ValueError) as exc:
            last_exc = exc
            logger.warning("Tentativa %d/%d falhou para %s: %s", attempt, MAX_RETRIES, url, exc)
            time.sleep(1.5 * attempt)
    else:
        raise RuntimeError(f"Falha ao buscar {url} após {MAX_RETRIES} tentativas") from last_exc

    matches = []
    for grupo in payload.get("jogos", []):
        for jogo in grupo.get("jogo", []):
            matches.append(
                {
                    "cbf_id": jogo["id_jogo"],
                    "num_jogo": jogo["num_jogo"],
                    "round": int(jogo["rodada"]),
                    "home": jogo["mandante"]["nome"],
                    "away": jogo["visitante"]["nome"],
                    "home_goals": jogo["mandante"]["gols"],
                    "away_goals": jogo["visitante"]["gols"],
                    "date": jogo["data"].strip(),
                    "time": jogo["hora"],
                    "venue": jogo["local"],
                }
            )
    return matches


def fetch_season(year: int) -> list[dict]:
    campeonato_id = CAMPEONATO_IDS.get(year)
    if campeonato_id is None:
        raise ValueError(f"campeonato_id desconhecido para {year}. Descubra e adicione a CAMPEONATO_IDS.")

    all_matches = []
    for round_number in range(1, N_ROUNDS + 1):
        all_matches.extend(fetch_round(campeonato_id, round_number))
    return all_matches


def parse_matches(raw_matches: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(raw_matches)
    df["played"] = df["home_goals"].notna() & df["away_goals"].notna()
    df["home_goals"] = pd.to_numeric(df["home_goals"], errors="coerce")
    df["away_goals"] = pd.to_numeric(df["away_goals"], errors="coerce")
    df["date_parsed"] = pd.to_datetime(df["date"], format="%d/%m/%Y", errors="coerce")
    return df
