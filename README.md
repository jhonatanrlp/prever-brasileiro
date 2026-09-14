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

python scripts/run_pipeline.py          # roda o pipeline inteiro, na ordem certa
pytest                                  # 49 testes
```

`run_pipeline.py` chama, em sequência, todos os scripts abaixo e para no primeiro
erro real (sem mascarar falha). `--fast` pula `compare_models.py` e
`backtest_2026.py` (~5 min cada); `--force` força redownload do histórico mesmo sem
mudança. Cada etapa também roda sozinha, se preferir:

```bash
python scripts/download_data.py         # histórico 2003-2024
python scripts/audit.py                 # outputs/data_audit.md
python scripts/build_team_mapping.py    # dicionário de times
python scripts/build_dataset.py         # modelo dimensional em data/processed/
python scripts/fetch_cbf_calendar.py    # 2025/2026 (API da CBF, data/rodada reais)
python scripts/build_recent_seasons.py  # matches_2025_2026 + estado atual + jogos restantes
python scripts/predict_current.py       # outputs/current_prediction.csv + match_predictions_2026.csv
python scripts/compare_models.py        # outputs/model_comparison.csv (baselines vs. ML, walk-forward)
python scripts/backtest_2026.py         # outputs/backtest_2026.csv (log loss rodada a rodada de 2026)
```

## Estrutura

```
configs/config.yaml   parâmetros do projeto (pontuação, simulação) — nunca hardcoded
data/
  raw/                 exatamente como baixado (adaoduque/ = histórico, cbf/ = 2025-2026)
  processed/           modelo dimensional, pronto para features
  external/            team_mapping.csv — dicionário mestre de times
scripts/               pontos de entrada do pipeline (run_pipeline.py roda tudo em ordem)
src/                   lógica (times, temporal/Elo, modelo de gols, simulação, avaliação) — módulos soltos
tests/                 pytest, incluindo os testes de não-vazamento
outputs/               data_audit.md, current_prediction.csv, match_predictions_2026.csv,
                       model_comparison.csv, backtest_2026.csv, tabela.html
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
  `posse_de_bola` e `precisao_passes` têm ~60–75% de missing — não usadas por isso;
  `chutes`, `chutes_no_alvo` e `escanteios` têm 0% de missing e alimentam
  `build_pre_match_features` como médias móveis por time (só para o histórico
  2003–2024 — a API da CBF usada em 2025/2026 não expõe essas estatísticas).
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
casa/fora, forma nas últimas 3/5/10 partidas, força média dos adversários já
enfrentados (Elo pré-jogo médio dos adversários), e média móvel de chutes/chutes a
gol/escanteios por jogo (`StatsState` em `src/temporal.py`, opcional — só populada
quando `team_stats` é passado, hoje só para o histórico 2003–2024). Estreias e
partidas sem estatística disponível recebem features nulas — tratamento explícito
de ausência, nunca um valor inventado (zero incluído).

**2025/2026: dados oficiais da CBF.** `data/processed/matches_2025_2026.csv` tem o
mesmo schema de `matches.csv`, com data e rodada reais de cada partida. Isso permite
concatenar com o histórico e passar por `build_pre_match_features` normalmente. A
classificação atual de 2026 (`current_standings_2026.csv`) é CALCULADA a partir das
partidas de 2026 já disputadas (`src/standings.py`), nunca copiada de uma tabela
pronta.

**Backtest rodada-a-rodada de 2026** (`scripts/backtest_2026.py`). Para cada rodada
já disputada, o modelo de gols é reajustado só com o que era conhecido ANTES daquela
rodada (histórico + 2025 + rodadas anteriores de 2026) e a previsão gerada é
comparada com o resultado real, rodada por rodada — responde "o que o modelo diria
na rodada X, sabendo só o que sabíamos até então". Só é possível porque
`matches_2025_2026.csv` tem data e rodada reais (a tentativa anterior via Wikipédia
não permitia isso). Resultado em `outputs/backtest_2026.csv`.

**Modelo de gols (Poisson + Dixon-Coles) e simulação (Monte Carlo).**
`src/poisson_goals.py`: força de ataque/defesa de cada time relativa à média da liga
— `lambda_home = média de gols do mandante na liga × ataque do mandante × defesa do
visitante` (e simetricamente para o visitante). Ajustado com todas as partidas
disponíveis (2003–2024 + 2025/2026 disputadas), com peso maior para temporadas
recentes (decaimento exponencial, meia-vida configurável). Poisson independente
sozinha subestima placares baixos e correlacionados (0x0, 1x0, 0x1, 1x1) — Dixon &
Coles (1997) corrigem isso com um fator `tau`, controlado por um parâmetro `rho`
ajustado por máxima verossimilhança nos dados de treino (nunca escolhido a dedo).
Achado real, honesto: nos dados do Brasileirão o `rho` ajustado fica perto de zero
(~-0.005) — o efeito de correlação de placares baixos é bem mais fraco aqui do que
no dataset original de Dixon-Coles (futebol inglês dos anos 90). A correção continua
correta (some a zero quando os dados não sustentam correlação), só que o ganho
prático é pequeno neste caso — reportado, não inflado.

`src/monte_carlo.py` simula o restante da temporada a partir da tabela real atual:
em cada uma das N temporadas simuladas, sorteia um placar direto da matriz de
probabilidade conjunta do modelo (`score_matrix`, já com o ajuste Dixon-Coles — não
duas Poisson independentes, que jogariam fora a correlação), acumula pontos/saldo, e
classifica com os mesmos critérios de `src/standings.py`. Vetorizado em NumPy sobre
o eixo das simulações (amostragem por transformada inversa da distribuição
achatada) — 100.000 temporadas em <1s. `scripts/predict_current.py` amarra as duas
peças e gera `outputs/current_prediction.csv` e `outputs/match_predictions_2026.csv`.

**Regras de competição como configuração.** Pontuação, número de rebaixados e vagas
continentais vêm de `configs/config.yaml`, lidos por `src/standings.py`
(`CompetitionRules`) — nunca hardcoded no código. Critérios de desempate
implementados, na ordem oficial da CBF: pontos, vitórias, saldo de gols, gols pró,
confronto direto (`_head_to_head_winner`, aplicado só quando exatamente 2 times
estão empatados — a regra oficial não vale para grupos de 3+, então nesse caso o
desempate pula direto para cartões) e menos cartões vermelhos/amarelos (`cards`
opcional, `data/processed/cards.csv` para o histórico; ainda não há esse dado para
2025/2026 — nesse caso o critério é ignorado e o empate remanescente fica por
sorteio, não simulado).

**Validação temporal (walk-forward) e comparação de modelos**
(`src/evaluation.py`, `scripts/compare_models.py`). Nunca `train_test_split`
aleatório: partidas do mesmo campeonato são correlacionadas no tempo.
`expanding_window_splits` treina com temporadas passadas e testa na próxima,
avançando uma de cada vez (15 temporadas mínimas de treino, testado em 2018–2026).
Comparados dois baselines (frequência histórica de mando; Elo com taxa de empate
constante), o modelo de produção (`poisson_dixon_coles`, refeito a cada temporada de
teste só com dados anteriores a ela) e Logistic Regression, Random Forest e Gradient
Boosting treinados sobre as features de `build_pre_match_features` (Elo, forma,
força do adversário, chutes/chutes a gol). Média de log loss por temporada
(2018–2026, menor é melhor; `log(3) ≈ 1.099` é o "chute" uniforme entre H/D/A):

| modelo | log loss | brier score | acurácia |
|---|---|---|---|
| logistic_regression | 1.022 | 0.613 | 0.496 |
| random_forest | 1.025 | 0.615 | 0.487 |
| **poisson_dixon_coles (produção)** | **1.028** | **0.617** | **0.489** |
| elo (baseline) | 1.031 | 0.619 | 0.478 |
| gradient_boosting | 1.044 | 0.627 | 0.474 |
| home_advantage (baseline) | 1.057 | 0.637 | 0.475 |

Todos os modelos batem o "chute" uniforme por margem clara. Logistic Regression e
Random Forest ficam um pouco à frente do modelo de produção, mas por margem pequena
— o Poisson+Dixon-Coles é competitivo (3º de 6) e tem a vantagem de gerar um placar
completo (não só H/D/A), o que o Monte Carlo precisa. Números completos por
temporada em `outputs/model_comparison.csv`.

## O que existe

1. Pipeline único que roda tudo em ordem, parando no primeiro erro real —
   [`scripts/run_pipeline.py`](scripts/run_pipeline.py)
2. Download automatizado e idempotente do histórico 2003–2024 —
   [`scripts/download_data.py`](scripts/download_data.py)
3. Auditoria automática de qualidade — [`scripts/audit.py`](scripts/audit.py) →
   [`outputs/data_audit.md`](outputs/data_audit.md)
4. Dicionário de times com `team_id` estável — [`data/external/team_mapping.csv`](data/external/team_mapping.csv)
5. Modelo dimensional (partidas, estatísticas, gols, cartões) —
   [`scripts/build_dataset.py`](scripts/build_dataset.py)
6. Reconstrução correta de `season` — [`src/seasons.py`](src/seasons.py)
7. Feature engineering temporal sem vazamento: Elo, forma, força do adversário,
   chutes/chutes a gol — [`src/temporal.py`](src/temporal.py)
8. 2025/2026 com data e rodada reais via API da CBF —
   [`src/cbf_calendario.py`](src/cbf_calendario.py),
   [`scripts/fetch_cbf_calendar.py`](scripts/fetch_cbf_calendar.py),
   [`scripts/build_recent_seasons.py`](scripts/build_recent_seasons.py)
9. Modelo de gols de Poisson com ajuste Dixon-Coles para correlação de placares
   baixos, `rho` ajustado por máxima verossimilhança (não escolhido a dedo) —
   [`src/poisson_goals.py`](src/poisson_goals.py)
10. Simulação de Monte Carlo amostrando direto da distribuição conjunta do modelo
    (preserva a correlação Dixon-Coles; 100.000 temporadas em <1s, vetorizada) —
    [`src/monte_carlo.py`](src/monte_carlo.py)
11. Regras de competição como configuração e classificação, com todos os critérios
    de desempate oficiais (pontos, vitórias, saldo, gols pró, confronto direto,
    cartões) — [`src/standings.py`](src/standings.py) (`CompetitionRules` + `build_standings`)
12. Validação temporal (walk-forward) e comparação de 6 modelos (2 baselines +
    Poisson-Dixon-Coles de produção + 3 de ML) —
    [`src/evaluation.py`](src/evaluation.py), [`scripts/compare_models.py`](scripts/compare_models.py) →
    [`outputs/model_comparison.csv`](outputs/model_comparison.csv)
13. Backtest rodada-a-rodada de 2026 —
    [`scripts/backtest_2026.py`](scripts/backtest_2026.py) →
    [`outputs/backtest_2026.csv`](outputs/backtest_2026.csv)
14. 49 testes `pytest`, incluindo testes que travam a garantia de não haver data
    leakage e testes específicos do ajuste Dixon-Coles

## Limitações conhecidas (honestas, não escondidas)

- Cartões de 2025/2026 não são coletados (a API da CBF usada não expõe isso), então
  esse critério de desempate fica sempre inativo para a temporada atual — cai para
  sorteio (não simulado) se um empate chegar até ali.
- O mesmo vale para chutes/chutes a gol/escanteios: só existem no histórico
  2003–2024, então essas features ficam `None` (imputadas pela mediana do treino) em
  2025/2026.
- O `rho` do Dixon-Coles sai perto de zero nos dados do Brasileirão — a correção é
  metodologicamente correta, mas o ganho prático medido é pequeno (ver tabela acima).
- Nenhum hiperparâmetro dos modelos de ML foi ajustado (usei os defaults do
  scikit-learn) — há espaço para tuning se algum desses modelos for promovido a
  produção no lugar do Poisson.
- O Monte Carlo desempata só por pontos/saldo/gols pró — confronto direto e cartões
  (já implementados em `src/standings.py` para a tabela real) não entram na
  simulação porque rodar isso 100.000 vezes ficaria caro; é uma simplificação
  documentada, não um bug.
