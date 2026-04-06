"""
Ejecuta todos los experimentos del paper secuencialmente usando dvc exp run.

Por cada experimento:
  1. dvc exp run --name {name} --set-param experiment.name={name} ...
  2. dvc exp apply {name}          → restaura params.yaml + outputs al directorio
  3. python src/archive_experiment.py  → archiva en plots/{name}/ y metrics/{name}/

Al terminar todos los experimentos:
  - git checkout params.yaml        → restaura params originales
  - python src/plot_model_comparison.py
  - python src/plot_backfill_impact.py

Uso:
    python src/run_all_experiments.py
    python src/run_all_experiments.py --dry-run
    python src/run_all_experiments.py --only xgb_linear,gru_linear
"""

import argparse
import subprocess
import sys

EXPERIMENTS = [
    {"name": "xgb_zero",              "model_type": "xgboost",     "backfill": "zero"},
    {"name": "xgb_linear",            "model_type": "xgboost",     "backfill": "linear"},
    {"name": "xgb_xgb_backcast",      "model_type": "xgboost",     "backfill": "xgboost_backcast"},
    {"name": "mlp_zero",              "model_type": "mlp",         "backfill": "zero"},
    {"name": "mlp_linear",            "model_type": "mlp",         "backfill": "linear"},
    {"name": "mlp_xgb_backcast",      "model_type": "mlp",         "backfill": "xgboost_backcast"},
    {"name": "gru_zero",              "model_type": "gru",         "backfill": "zero"},
    {"name": "gru_linear",            "model_type": "gru",         "backfill": "linear"},
    {"name": "gru_xgb_backcast",      "model_type": "gru",         "backfill": "xgboost_backcast"},
    {"name": "cnngru_zero",           "model_type": "cnn_gru",     "backfill": "zero"},
    {"name": "cnngru_linear",         "model_type": "cnn_gru",     "backfill": "linear"},
    {"name": "cnngru_xgb_backcast",   "model_type": "cnn_gru",     "backfill": "xgboost_backcast"},
    {"name": "rescnngru_zero",        "model_type": "res_cnn_gru", "backfill": "zero"},
    {"name": "rescnngru_linear",      "model_type": "res_cnn_gru", "backfill": "linear"},
    {"name": "rescnngru_xgb_backcast","model_type": "res_cnn_gru", "backfill": "xgboost_backcast"},
]


def run(cmd: list[str], dry_run: bool) -> int:
    print(f"  $ {' '.join(cmd)}")
    if dry_run:
        return 0
    result = subprocess.run(cmd)
    return result.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description="Corre todos los experimentos del paper.")
    parser.add_argument("--dry-run", action="store_true", help="Muestra comandos sin ejecutarlos.")
    parser.add_argument(
        "--only",
        help="Lista de experimentos separados por coma (ej: xgb_linear,gru_linear).",
    )
    args = parser.parse_args()

    exps = EXPERIMENTS
    if args.only:
        names = set(args.only.split(","))
        exps = [e for e in EXPERIMENTS if e["name"] in names]
        if not exps:
            print(f"ERROR: ningún experimento coincide con: {args.only}")
            sys.exit(1)

    failed = []

    for exp in exps:
        name = exp["name"]
        print(f"\n{'='*60}")
        print(f"Experimento: {name}")
        print(f"  model.type = {exp['model_type']}")
        print(f"  backfill   = {exp['backfill']}")
        print()

        # 1. dvc exp run
        rc = run(
            [
                "dvc", "exp", "run",
                "--name", name,
                "--set-param", f"experiment.name={name}",
                "--set-param", f"model.type={exp['model_type']}",
                "--set-param", f"features.indicators.backfill_method={exp['backfill']}",
            ],
            args.dry_run,
        )
        if rc != 0:
            print(f"  WARN: dvc exp run terminó con código {rc}. Continuando...")
            failed.append(name)
            continue

        # 2. dvc exp apply
        rc = run(["dvc", "exp", "apply", name], args.dry_run)
        if rc != 0:
            print(f"  WARN: dvc exp apply terminó con código {rc}. Saltando archivado.")
            failed.append(name)
            continue

        # 3. archive
        rc = run([sys.executable, "src/archive_experiment.py"], args.dry_run)
        if rc != 0:
            print(f"  WARN: archive_experiment.py terminó con código {rc}.")

    # Restaurar params.yaml
    print(f"\n{'='*60}")
    print("Restaurando params.yaml al estado original...")
    run(["git", "checkout", "params.yaml"], args.dry_run)

    # Gráficas de comparación
    if not args.dry_run:
        print("\nGenerando gráficas de comparación...")
        run([sys.executable, "src/plot_model_comparison.py"], dry_run=False)
        run([sys.executable, "src/plot_backfill_impact.py"], dry_run=False)
    else:
        print("\n[dry-run] Se generarían:")
        print("  $ python src/plot_model_comparison.py")
        print("  $ python src/plot_backfill_impact.py")

    print(f"\n{'='*60}")
    if failed:
        print(f"Terminado con errores en: {', '.join(failed)}")
    else:
        print("¡Todos los experimentos completados correctamente!")
    print("\nVer resultados con: dvc exp show")


if __name__ == "__main__":
    main()
