"""Roda o pipeline inteiro, do download dos dados brutos até a previsão final,
na ordem correta. Cada etapa é um script independente (pode ser rodado sozinho —
ver README.md); este arquivo só garante a ordem certa e para no primeiro erro
real, sem esconder falhas.

Uso:
    python scripts/run_pipeline.py            # roda tudo, inclusive comparação de
                                               # modelos e backtest (mais lento)
    python scripts/run_pipeline.py --fast     # pula compare_models.py e
                                               # backtest_2026.py (~5 min cada)
    python scripts/run_pipeline.py --force    # força redownload mesmo sem mudança
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable

# (nome do arquivo, argumentos extras, descrição)
CORE_STEPS = [
    ("download_data.py", [], "Baixa o histórico 2003-2024"),
    ("audit.py", [], "Audita a qualidade dos dados brutos"),
    ("build_team_mapping.py", [], "Gera o dicionário mestre de times"),
    ("build_dataset.py", [], "Constrói o modelo dimensional histórico"),
    ("fetch_cbf_calendar.py", [], "Baixa 2025/2026 da API da CBF"),
    ("build_recent_seasons.py", [], "Constrói matches_2025_2026 + estado atual"),
    ("predict_current.py", [], "Gera a previsão do restante de 2026"),
]
SLOW_STEPS = [
    ("compare_models.py", [], "Compara baselines contra modelos de ML (walk-forward)"),
    ("backtest_2026.py", [], "Backtest rodada a rodada de 2026"),
]


def run_step(filename: str, extra_args: list[str], description: str) -> float:
    # flush=True é essencial aqui: sem isso, o stdout do processo pai fica em
    # buffer de bloco quando a saída é redirecionada (pipe/arquivo), e os prints
    # deste script aparecem só no final, depois de todo o output dos subprocessos
    # — dando a impressão de que a ordem das etapas está embaralhada.
    print(f"\n{'=' * 60}\n{description} ({filename})\n{'=' * 60}", flush=True)
    start = time.monotonic()
    result = subprocess.run([PYTHON, str(ROOT / filename), *extra_args])
    elapsed = time.monotonic() - start
    if result.returncode != 0:
        raise SystemExit(
            f"\nPipeline interrompido: '{filename}' falhou (código {result.returncode})."
        )
    print(f"-- {filename} concluído em {elapsed:.1f}s", flush=True)
    return elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fast", action="store_true", help="pula compare_models.py e backtest_2026.py")
    parser.add_argument("--force", action="store_true", help="passa --force para download_data.py")
    args = parser.parse_args()

    steps = list(CORE_STEPS)
    if not args.fast:
        steps += SLOW_STEPS

    total_start = time.monotonic()
    for filename, extra_args, description in steps:
        if filename == "download_data.py" and args.force:
            extra_args = [*extra_args, "--force"]
        run_step(filename, extra_args, description)

    total_elapsed = time.monotonic() - total_start
    print(f"\n{'=' * 60}\nPipeline completo em {total_elapsed / 60:.1f} min.\n{'=' * 60}")
    print(
        "Saídas: outputs/current_prediction.csv, outputs/match_predictions_2026.csv"
        + ("" if args.fast else ", outputs/model_comparison.csv, outputs/backtest_2026.csv")
    )


if __name__ == "__main__":
    main()
