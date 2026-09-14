# prever-brasileiro

Sistema probabilístico de previsão do Campeonato Brasileiro Série A: Vitória/Empate/
Derrota, gols esperados, e simulação de Monte Carlo da temporada — pontos esperados,
posição final, probabilidade de título, Libertadores, Sul-Americana e rebaixamento.

O campeonato é tratado como um processo temporal (Elo dinâmico + forma recente,
atualizados partida a partida), com uma prioridade explícita: qualidade dos dados e
ausência de data leakage antes de qualquer treinamento de modelo.

**Resultado publicado:** [Brasileirão 2026 — tabela e projeção](https://claude.ai/code/artifact/4a39de7c-56be-45c6-a933-2b44773376f5)

## Como rodar

```bash
python -m venv .venv
.venv\Scripts\activate          # Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt

python scripts/download_data.py         # histórico 2003-2024
python scripts/audit.py                 # outputs/data_audit.md
python scripts/build_team_mapping.py    # dicionário de times
python scripts/build_dataset.py         # modelo dimensional em data/processed/
python scripts/fetch_cbf_calendar.py    # 2025/2026 (API da CBF, data/rodada reais)
python scripts/build_recent_seasons.py  # matches_2025_2026 + estado atual + jogos restantes
python scripts/predict_current.py       # outputs/current_prediction.csv + match_predictions_2026.csv

pytest                                  # 32 testes
```

`download_data.py` e `fetch_cbf_calendar.py` são idempotentes: não rebaixam se o
conteúdo não mudou (hash sha256). Use `--force` em `download_data.py` para forçar.

## Estrutura

```
configs/config.yaml   parâmetros do projeto (pontuação, simulação) — nunca hardcoded
data/
  raw/                 exatamente como baixado (adaoduque/ = histórico, cbf/ = 2025-2026)
  processed/           modelo dimensional, pronto para features
  external/            team_mapping.csv — dicionário mestre de times
scripts/               pontos de entrada do pipeline (um arquivo por etapa)
src/                   lógica (times, temporal/Elo, modelo de gols, simulação) — 8 módulos soltos
tests/                 pytest, incluindo os testes de não-vazamento
outputs/               data_audit.md, current_prediction.csv, match_predictions_2026.csv, tabela.html
```

## Fontes de dados

### Histórico 2003–2024 — `adaoduque/Brasileirao_Dataset`

https://github.com/adaoduque/Brasileirao_Dataset — 4 CSVs ligados por `ID`/`partida_ID`
(partidas, estatísticas, gols, cartões), documentados em `Legenda.txt`. Escolhida por
ser a fonte canônica (outros repositórios encontrados na pesquisa replicam exatamente
o mesmo schema e números — indicando que derivam dela), acessível sem autenticação via
`raw.githubusercontent.com`, e por cobrir 2003–2024 (8.785 partidas — confirmado pela
auditoria automática em `outputs/data_audit.md`, não pelo texto do README da fonte).

Descartadas por não agregarem cobertura real: `leo-amaral/Brasileirao-2003-a-2025` e
`leeofernandes1980/brasileirao-dataset` (mesmo conteúdo do adaoduque, mesma
proveniência), e datasets do Kaggle com cobertura menor ou sobreposta.

### 2025/2026 — API não documentada da CBF

Inspecionando as chamadas de rede da própria página de tabelas do site da CBF
(`cbf.com.br/futebol-brasileiro/tabelas/...`), encontramos
`GET /api/cbf/jogos/campeonato/{campeonato_id}/rodada/{n}/fase`: devolve todas as
partidas de uma rodada com placar, data e hora. `campeonato_id` da Série A é `12606`
(2025) e `1260611` (2026) — descobertos navegando o site, não documentados
publicamente; se mudarem em anos futuros, precisam ser redescobertos da mesma forma.

Isso é superior a alternativas avaliadas antes (Wikipédia, que só tinha o placar
agregado por confronto sem data/rodada; Sofascore/API-Football, que exigiriam scraping
de API não pública ou uma chave paga): com data e rodada reais, 2025/2026 têm o mesmo
formato do histórico e alimentam o mesmo pipeline de features sem tratamento especial.

Implementado em `src/cbf_calendario.py` / `scripts/fetch_cbf_calendar.py` /
`scripts/build_recent_seasons.py`. As requisições usam `verify=False` no `httpx`
porque o servidor da CBF serve uma cadeia de certificado TLS incompleta que
navegadores toleram (via AIA) mas o OpenSSL não busca automaticamente — decisão
deliberada e documentada, aceitável aqui por ser leitura pública sem credenciais.

### Times sem histórico 2003–2024

Mirassol e Remo estrearam na Série A em 2025/2026 e não aparecem no dataset
histórico. Foram adicionados manualmente a `data/external/team_mapping.csv` (ver
`scripts/build_team_mapping.py`) — não há como derivá-los de uma fonte que termina
antes da estreia deles.

## Dicionário de dados (`data/processed/`)

Todas as chaves de time são `team_id` (ver `data/external/team_mapping.csv`), nunca
strings livres.

- **`matches.csv`** — histórico 2003–2024, 1 linha por partida: `match_id`, `season`
  (reconstruído, ver Metodologia — não é o ano-calendário da data), `competition`,
  `round`, `date`, `home_team_id`, `away_team_id`, `home_goals`, `away_goals`,
  `arena`, `result` (`H`/`D`/`A`, recalculado do placar — a coluna `vencedor` da fonte
  bruta tem inconsistências, ver `outputs/data_audit.md`).
- **`team_match_stats.csv` / `goals.csv` / `cards.csv`** — estatísticas por (partida,
  time), gols e cartões individuais, grão fino, ligados por `match_id`.
  `posse_de_bola` e `precisao_passes` têm ~60–75% de missing — não usar sem decidir
  uma estratégia de ausência explícita.
- **`matches_2025_2026.csv`** — CBF, com data e rodada reais. Mesmas colunas centrais
  de `matches.csv` mais `played` (bool) e `venue`. Pode ser concatenado a
  `matches.csv` para `build_pre_match_features`.
- **`current_standings_2026.csv` / `remaining_fixtures_2026.csv`** — classificação
  atual (calculada, não copiada) e jogos ainda não disputados de 2026, entrada direta
  de `scripts/predict_current.py`.
- **`teams.csv`** — cópia de `data/external/team_mapping.csv`: `team_id`, `team_name`
  (grafia da fonte), `canonical_name`, `state`, `aliases`, `first_season`,
  `last_season`.

## Metodologia

**Zero data leakage.** O modelo prevê uma partida antes dela acontecer. Nenhuma
feature usada para prever a partida `M` pode depender de informação só disponível em
`M` ou em partidas posteriores. Garantido estruturalmente por
`build_pre_match_features` (`src/temporal.py`): percorre as partidas em ordem
cronológica e, para cada uma, calcula as features a partir de um estado acumulado
(Elo, forma) que só é atualizado *depois* de revelar o resultado daquela partida.
Verificado em `tests/test_no_leakage.py`.

**Reconstrução de temporada (season).** A coluna `season` não existe no dado bruto;
é derivada da data. `season = ano(data)` está errado: a Série A 2020 foi disputada de
agosto/2020 a fevereiro/2021 (pandemia), então parte de suas rodadas cai no
ano-calendário 2021. `src/seasons.assign_season_ids` detecta a entressafra: um gap
maior que 45 dias entre duas partidas consecutivas (por data) marca a fronteira entre
temporadas. Validado contra 2003–2024: produz exatamente 21 fronteiras (22
temporadas), com 380 partidas em cada temporada de 20 times em turno e returno —
inclusive fundindo corretamente 2020/08–2021/02 em uma única temporada "2020".

**Elo dinâmico** (`src/temporal.py`: `EloConfig`/`EloState`). Cada time começa com um
rating inicial (`EloConfig.initial_rating`, default 1500). Antes de cada partida, a
expectativa de vitória do mandante vem da fórmula logística padrão de Elo com
vantagem de mando somada ao seu rating; depois do resultado, os ratings são
atualizados com um fator `K` escalado pela margem de vitória (estilo
FiveThirtyEight). O rating "pré-jogo" é sempre o estado do dicionário de ratings no
momento em que a partida é processada.

**Features temporais.** Para cada time, antes de cada partida (quando há histórico):
pontos por jogo, saldo por jogo, gols marcados/sofridos por jogo, aproveitamento em
casa/fora, forma nas últimas 3/5/10 partidas, e força média dos adversários já
enfrentados (Elo pré-jogo médio dos adversários). Estreias recebem features nulas —
tratamento explícito de cold start, não um valor arbitrário.

**2025/2026: dados oficiais da CBF.** `data/processed/matches_2025_2026.csv` tem o
mesmo schema de `matches.csv`, com data e rodada reais de cada partida. Isso permite
concatenar com o histórico e passar por `build_pre_match_features` normalmente —
inclusive um backtest rodada-a-rodada de 2026 (ainda sem script próprio, mas os dados
já suportam). A classificação atual de 2026 (`current_standings_2026.csv`) é
CALCULADA a partir das partidas de 2026 já disputadas (`src/standings.py`), nunca
copiada de uma tabela pronta.

**Modelo de gols (Poisson) e simulação (Monte Carlo).** `src/poisson_goals.py`: força
de ataque/defesa de cada time relativa à média da liga —
`lambda_home = média de gols do mandante na liga × ataque do mandante × defesa do
visitante` (e simetricamente para o visitante). Ajustado com todas as partidas
disponíveis (2003–2024 + 2025/2026 disputadas), com peso maior para temporadas
recentes (decaimento exponencial, meia-vida configurável).
`src/monte_carlo.py` simula o restante da temporada a partir da tabela real atual: em
cada uma das N temporadas simuladas, sorteia um placar Poisson por jogo restante,
acumula pontos/saldo, e classifica com os mesmos critérios de `src/standings.py`.
Vetorizado em NumPy sobre o eixo das simulações — 100.000 temporadas em <1s.
`scripts/predict_current.py` amarra as duas peças e gera
`outputs/current_prediction.csv` e `outputs/match_predictions_2026.csv`.

**Regras de competição como configuração.** Pontuação, número de rebaixados e vagas
continentais vêm de `configs/config.yaml`, lidos por `src/standings.py`
(`CompetitionRules`) — nunca hardcoded no código. Critérios de desempate
implementados: pontos, vitórias, saldo de gols, gols pró. Confronto direto e cartões
(também em `configs/config.yaml`) não estão implementados — exigiriam uma mini-liga
par a par; resolvem a esmagadora maioria dos casos reais sem isso.

**Validação temporal (walk-forward) — pendente.** Quando o projeto comparar modelos
de ML, a validação principal precisa ser expanding window por temporada (treina com
temporadas passadas, testa na próxima), nunca `train_test_split` aleatório —
partidas do mesmo campeonato são correlacionadas no tempo. Ainda não implementado
(não há modelo de ML para validar ainda).

## O que existe

1. Download automatizado e idempotente do histórico 2003–2024 —
   [`scripts/download_data.py`](scripts/download_data.py)
2. Auditoria automática de qualidade — [`scripts/audit.py`](scripts/audit.py) →
   [`outputs/data_audit.md`](outputs/data_audit.md)
3. Dicionário de times com `team_id` estável — [`data/external/team_mapping.csv`](data/external/team_mapping.csv)
4. Modelo dimensional (partidas, estatísticas, gols, cartões) —
   [`scripts/build_dataset.py`](scripts/build_dataset.py)
5. Reconstrução correta de `season` — [`src/seasons.py`](src/seasons.py)
6. Feature engineering temporal sem vazamento: Elo, forma, força do adversário —
   [`src/temporal.py`](src/temporal.py)
7. 2025/2026 com data e rodada reais via API da CBF —
   [`src/cbf_calendario.py`](src/cbf_calendario.py),
   [`scripts/fetch_cbf_calendar.py`](scripts/fetch_cbf_calendar.py),
   [`scripts/build_recent_seasons.py`](scripts/build_recent_seasons.py)
8. Modelo de gols de Poisson — [`src/poisson_goals.py`](src/poisson_goals.py)
9. Simulação de Monte Carlo (100.000 temporadas em <1s, vetorizada) —
   [`src/monte_carlo.py`](src/monte_carlo.py)
10. Regras de competição como configuração e classificação —
    [`src/standings.py`](src/standings.py) (`CompetitionRules` + `build_standings`)
11. 32 testes `pytest`, incluindo testes que travam a garantia de não haver data leakage

## O que falta

- Nenhum modelo de ML foi treinado/comparado ainda (Logistic Regression, Random
  Forest, Gradient Boosting) — nem os baselines simples nem a validação walk-forward
  existem no código hoje.
- Backtest rodada-a-rodada de 2026 não tem script próprio, mas os dados já suportam
  (`data/processed/matches_2025_2026.csv` tem data/rodada reais).
- Confronto direto e cartões como critério de desempate não estão implementados em
  `src/standings.py` (só pontos → vitórias → saldo → gols pró).
