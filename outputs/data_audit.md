# Relatório de auditoria de dados — Campeonato Brasileiro



Gerado automaticamente por `scripts/audit.py`.



## Auditoria de partidas

- Período: 2003-03-29 a 2024-12-08
- Temporadas distintas: [2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
- Total de partidas: 8785
- Datas não parseáveis: 0
- Times distintos: 45
- Placares negativos (impossíveis): 0
- Inconsistências vencedor x placar: 1

### Partidas por temporada e por rodada (min/max; esperado ~10 em turno duplo com 20 times)

| temporada | partidas | min/rodada | max/rodada |
|---|---|---|---|
| 2003 | 552 | 11 | 13 |
| 2004 | 552 | 12 | 12 |
| 2005 | 462 | 11 | 11 |
| 2006 | 380 | 10 | 10 |
| 2007 | 380 | 10 | 10 |
| 2008 | 380 | 10 | 10 |
| 2009 | 380 | 10 | 10 |
| 2010 | 380 | 10 | 10 |
| 2011 | 380 | 10 | 10 |
| 2012 | 380 | 10 | 10 |
| 2013 | 380 | 10 | 10 |
| 2014 | 380 | 10 | 10 |
| 2015 | 380 | 10 | 10 |
| 2016 | 379 | 9 | 10 |
| 2017 | 380 | 10 | 10 |
| 2018 | 380 | 10 | 10 |
| 2019 | 380 | 10 | 10 |
| 2020 | 268 | 9 | 10 |
| 2021 | 492 | 10 | 20 |
| 2022 | 380 | 10 | 10 |
| 2023 | 380 | 10 | 10 |
| 2024 | 380 | 10 | 10 |

### estatísticas

- Shape: 17570 linhas x 13 colunas
- Linhas duplicadas: 0

| coluna | dtype | % missing | valores únicos |
|---|---|---|---|
| partida_id | int64 | 0.0 | 8785 |
| rodata | int64 | 0.0 | 46 |
| clube | str | 0.0 | 45 |
| chutes | int64 | 0.0 | 37 |
| chutes_no_alvo | int64 | 0.0 | 17 |
| posse_de_bola | str | 61.18 | 61 |
| passes | int64 | 0.0 | 523 |
| precisao_passes | str | 74.32 | 42 |
| faltas | int64 | 0.0 | 32 |
| cartao_amarelo | int64 | 0.0 | 11 |
| cartao_vermelho | int64 | 0.0 | 4 |
| impedimentos | int64 | 0.0 | 11 |
| escanteios | int64 | 0.0 | 21 |

### gols

- Shape: 9861 linhas x 6 colunas
- Linhas duplicadas: 0

| coluna | dtype | % missing | valores únicos |
|---|---|---|---|
| partida_id | int64 | 0.0 | 3819 |
| rodata | int64 | 0.0 | 38 |
| clube | str | 0.0 | 34 |
| atleta | str | 0.0 | 1532 |
| minuto | str | 0.0 | 116 |
| tipo_de_gol | str | 88.0 | 2 |

### cartões

- Shape: 20953 linhas x 8 colunas
- Linhas duplicadas: 0

| coluna | dtype | % missing | valores únicos |
|---|---|---|---|
| partida_id | int64 | 0.0 | 4116 |
| rodata | int64 | 0.0 | 38 |
| clube | str | 0.0 | 34 |
| cartao | str | 0.0 | 2 |
| atleta | str | 0.03 | 2290 |
| num_camisa | float64 | 1.84 | 101 |
| posicao | str | 5.72 | 5 |
| minuto | str | 0.0 | 122 |