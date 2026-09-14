# prever-brasileiro

Sistema probabilístico de previsão do Campeonato Brasileiro Série A: Vitória/Empate/
Derrota, gols esperados, e simulação de Monte Carlo da temporada — pontos esperados,
posição final, probabilidade de título, Libertadores, Sul-Americana e rebaixamento.

O campeonato é tratado como um processo temporal (Elo dinâmico + forma recente,
atualizados partida a partida), com uma prioridade explícita: qualidade dos dados e
ausência de data leakage antes de qualquer treinamento de modelo.

## Resultado publicado

**[Brasileirão 2026 — tabela e projeção](https://claude.ai/artifact/AAcafYXLJPdhxxyo465CEg)**

Classificação atual + projeção final (título, Libertadores, Sul-Americana,
rebaixamento) + **todos os 115 jogos restantes** até a rodada 38, com o placar mais
provável de cada um — não só a próxima rodada. Jogos sem data confirmada pela CBF
aparecem agrupados por rodada no fim da página. Atualizado rodando
`scripts/predict_current.py` de novo e republicando este mesmo link.

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
- **`outputs/match_predictions_2026.csv`** — por jogo restante: `home_win_probability`/
  `draw_probability`/`away_win_probability`, `expected_goals_home`/`expected_goals_away`
  (os `lambda` do Poisson) e `top1_score`..`top5_score` com suas probabilidades +
  `top5_coverage` (soma dessas 5 probabilidades — normalmente ~50%, nunca 100%:
  futebol tem muitos placares plausíveis, o "mais provável" não é "o esperado").
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
visitante` (e simetricamente para o visitante). Poisson independente sozinha
subestima placares baixos e correlacionados (0x0, 1x0, 0x1, 1x1) — Dixon & Coles
(1997) corrigem isso com um fator `tau`, controlado por um parâmetro `rho`.

Duas formas de ajustar os parâmetros, as duas no código:

- **`fit_poisson_goals_model`** (produção): ataque/defesa por média ponderada
  (método dos momentos, peso maior pra temporadas recentes), `rho` depois, sozinho,
  por perfil de máxima verossimilhança.
- **`fit_poisson_goals_model_mle`**: ataque, defesa, mando de campo e `rho` TODOS
  ajustados de uma vez, maximizando uma única log-verossimilhança (otimização
  L-BFGS-B, ~95 parâmetros para 47 times, com uma penalização ridge leve pra evitar
  divergência numérica em casos degenerados — ex.: time que nunca marcou na
  amostra). Metodologicamente mais correto: os parâmetros não são estimados em dois
  passos desacoplados.

**Comparei os dois no mesmo walk-forward (2018–2026, mesmas partidas) antes de
decidir qual usar em produção** — não assumi que o método mais sofisticado seria
melhor:

| | log loss | RPS | acurácia |
|---|---|---|---|
| `fit_poisson_goals_model` (produção) | 1.0281 | 0.2090 | 0.4892 |
| `fit_poisson_goals_model_mle` | 1.0284 | 0.2091 | 0.4903 |

**Resultado: estatisticamente empatados** (diferença de 0.0002 em log loss, contra
um desvio-padrão de ~0.02 entre temporadas — ruído, não sinal). A MLE conjunta
convergiu nas 9 temporadas testadas (`outputs/dixon_coles_mle_diagnostics.csv` tem
log-verossimilhança/AIC/BIC/convergência por temporada; `rho` variou entre -0.007 e
+0.035 dependendo da janela de treino), mas não trouxe ganho preditivo mensurável, e
custa ~20s por ajuste contra <1s do método antigo. **Por isso a produção continua
usando `fit_poisson_goals_model`** — a versão mais simples e mais rápida, já que a
mais complexa não entregou vantagem real. `fit_poisson_goals_model_mle` fica
disponível para quem quiser validar/pesquisar, e é comparada de novo a cada rodada
de `scripts/compare_models.py`.

Essa comparação também responde a uma pergunta maior: com o `rho` do Dixon-Coles
perto de zero nos dois métodos e a MLE conjunta não melhorando nada, o gargalo
atual não é o método de ajuste — é a informação disponível (features). Empilhar
mais modelos num ensemble agora, antes de resolver isso, criaria complexidade sem
saber se ela ajuda; ver "Próximos passos" abaixo.

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
constante), os dois ajustes do modelo de gols (produção e MLE conjunta, ver acima) e
Logistic Regression, Random Forest e Gradient Boosting treinados sobre as features
de `build_pre_match_features` (Elo, forma, força do adversário, chutes/chutes a
gol). Métricas: log loss e Brier tratam H/D/A como categorias sem ordem; **RPS**
(Ranked Probability Score) usa a ordem natural derrota→empate→vitória e penaliza
menos um erro "vizinho" (achar que ia empatar) do que um erro entre extremos (achar
que o time perdedor venceria) — é a métrica padrão da literatura de forecasting
esportivo por causa disso. Média por temporada (2018–2026, menor é melhor exceto
acurácia; `log(3) ≈ 1.099` é o "chute" uniforme entre H/D/A):

| modelo | log loss | brier score | RPS | acurácia |
|---|---|---|---|---|
| logistic_regression | 1.0220 | 0.6135 | 0.2070 | 0.4957 |
| random_forest | 1.0247 | 0.6148 | 0.2077 | 0.4868 |
| **poisson_dixon_coles (produção)** | **1.0281** | **0.6172** | **0.2090** | **0.4892** |
| poisson_dixon_coles_mle | 1.0284 | 0.6174 | 0.2091 | 0.4903 |
| elo (baseline) | 1.0310 | 0.6195 | 0.2098 | 0.4782 |
| gradient_boosting | 1.0438 | 0.6271 | 0.2117 | 0.4738 |
| home_advantage (baseline) | 1.0567 | 0.6373 | 0.2187 | 0.4750 |

Todos os modelos batem o "chute" uniforme por margem clara, e os dois ajustes do
Poisson+Dixon-Coles ficam essencialmente empatados entre si (ver seção acima).
Logistic Regression e Random Forest ficam um pouco à frente em log loss/RPS, mas por
margem pequena (desvio-padrão entre temporadas é ~0.02-0.03 — a diferença está no
limiar do ruído). O Poisson+Dixon-Coles continua em produção apesar disso porque
gera um placar completo (não só H/D/A), o que o Monte Carlo precisa para simular
saldo de gols — trocar por um classificador V/E/D exigiria um modelo de gols
separado de qualquer forma. Números completos por temporada em
`outputs/model_comparison.csv`.

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
9. Modelo de gols de Poisson com ajuste Dixon-Coles, com DOIS métodos de ajuste
   comparados empiricamente (método dos momentos + perfil de MLE vs. MLE conjunta)
   — [`src/poisson_goals.py`](src/poisson_goals.py)
10. Top-5 placares mais prováveis + gols esperados por jogo, expostos no output
    (não só V/E/D) — [`scripts/predict_current.py`](scripts/predict_current.py) →
    [`outputs/match_predictions_2026.csv`](outputs/match_predictions_2026.csv)
11. Simulação de Monte Carlo amostrando direto da distribuição conjunta do modelo
    (preserva a correlação Dixon-Coles; 100.000 temporadas em <1s, vetorizada) —
    [`src/monte_carlo.py`](src/monte_carlo.py)
12. Regras de competição como configuração e classificação, com todos os critérios
    de desempate oficiais (pontos, vitórias, saldo, gols pró, confronto direto,
    cartões) — [`src/standings.py`](src/standings.py) (`CompetitionRules` + `build_standings`)
13. Validação temporal (walk-forward) e comparação de 7 modelos (2 baselines + 2
    ajustes do Poisson-Dixon-Coles + 3 de ML), com log loss/Brier/**RPS**/acurácia —
    [`src/evaluation.py`](src/evaluation.py), [`scripts/compare_models.py`](scripts/compare_models.py) →
    [`outputs/model_comparison.csv`](outputs/model_comparison.csv),
    [`outputs/dixon_coles_mle_diagnostics.csv`](outputs/dixon_coles_mle_diagnostics.csv)
14. Backtest rodada-a-rodada de 2026 —
    [`scripts/backtest_2026.py`](scripts/backtest_2026.py) →
    [`outputs/backtest_2026.csv`](outputs/backtest_2026.csv)
15. 62 testes `pytest`, incluindo testes que travam a garantia de não haver data
    leakage, testes do ajuste Dixon-Coles (escalar e vetorizado), da MLE conjunta
    (convergência, não-divergência em casos degenerados) e do RPS

## Limitações conhecidas (honestas, não escondidas)

- Cartões de 2025/2026 não são coletados (a API da CBF usada não expõe isso), então
  esse critério de desempate fica sempre inativo para a temporada atual — cai para
  sorteio (não simulado) se um empate chegar até ali.
- O mesmo vale para chutes/chutes a gol/escanteios: só existem no histórico
  2003–2024, então essas features ficam `None` (imputadas pela mediana do treino) em
  2025/2026.
- O `rho` do Dixon-Coles sai perto de zero nos dados do Brasileirão nos dois métodos
  de ajuste — a correção é metodologicamente correta, mas o ganho prático medido é
  pequeno (ver comparação acima).
- A MLE conjunta não superou o método dos momentos + perfil de MLE (diferença dentro
  do ruído entre temporadas) — ver seção "Modelo de gols" acima para os números e a
  decisão de manter o método mais simples/rápido em produção.
- Nenhum hiperparâmetro dos modelos de ML foi ajustado (usei os defaults do
  scikit-learn) — há espaço para tuning se algum desses modelos for promovido a
  produção no lugar do Poisson.
- O Monte Carlo desempata só por pontos/saldo/gols pró — confronto direto e cartões
  (já implementados em `src/standings.py` para a tabela real) não entram na
  simulação porque rodar isso 100.000 vezes ficaria caro; é uma simplificação
  documentada, não um bug.
- Sem dado de xG (nenhuma das fontes disponíveis — histórico ou CBF — expõe isso),
  então um "Elo aumentado por xG" não é possível sem inventar o dado.

## Próximos passos — por que não ensemble ainda

A pergunta natural depois de comparar 7 modelos é "por que não combinar todos num
ensemble?". Decidi não fazer isso agora porque a comparação acima já respondeu uma
pergunta mais importante primeiro: **o gargalo não é o método de estimação** (MLE
conjunta ≈ método dos momentos) **nem falta de modelos mais sofisticados** (Random
Forest/Gradient Boosting não superam o Poisson por margem relevante) — os 7 modelos
convergem para a mesma faixa de log loss (~1.02–1.06). Isso sugere que o limite
atual é a **informação disponível** (Elo, forma, chutes históricos), não a técnica
de modelagem. Empilhar os mesmos 7 modelos num ensemble tende a herdar esse teto
compartilhado, não superá-lo. Os candidatos mais prováveis para romper esse teto,
na ordem em que eu tentaria:

1. xG real (não temos a fonte ainda).
2. Estatísticas de jogo (posse, passes) além de chutes/escanteios — já
   parcialmente disponíveis, mas com muito missing (60-75%) no histórico.
3. Só então, ensemble e/ou modelo hierárquico Bayesiano com shrinkage entre times
   com poucos jogos — ambos ficam mais fáceis de justificar depois de esgotar as
   features disponíveis.
