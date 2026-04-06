"""
Archiva todos los experimentos ya ejecutados via dvc exp apply + archive_experiment.py.

Usar cuando run_all_experiments.py completó los dvc exp run pero saltó el archivado
(ej. por el error de rmdir en Windows).

Uso:
    python src/archive_all_experiments.py
    python src/archive_all_experiments.py --only xgb_zero,xgb_linear
"""

import argparse
import subprocess
import sys

from run_all_experiments import EXPERIMENTS


def run(cmd: list[str]) -> int:
    print(f"  $ {' '.join(cmd)}")
    return subprocess.run(cmd).returncode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", help="Subconjunto separado por coma.")
    args = parser.parse_args()

    exps = EXPERIMENTS
    if args.only:
        names = set(args.only.split(","))
        exps = [e for e in EXPERIMENTS if e["name"] in names]

    failed = []
    for exp in exps:
        name = exp["name"]
        print(f"\n{'='*50}")
        print(f"Archivando: {name}")

        rc = run(["dvc", "exp", "apply", name])
        if rc != 0:
            print(f"  WARN: dvc exp apply falló para {name}.")
            failed.append(name)
            continue

        rc = run([sys.executable, "src/archive_experiment.py"])
        if rc != 0:
            print(f"  WARN: archive_experiment.py falló para {name}.")
            failed.append(name)

    # Restaurar params.yaml
    print(f"\n{'='*50}")
    print("Restaurando params.yaml...")
    run(["git", "checkout", "params.yaml"])

    if failed:
        print(f"\nFallaron: {', '.join(failed)}")
    else:
        print("\n¡Todos los experimentos archivados!")


if __name__ == "__main__":
    main()
