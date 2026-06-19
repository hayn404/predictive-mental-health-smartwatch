#!/usr/bin/env python3
"""
Download each model's figures (logged as MLflow artifacts under `figures/`) from
the DagsHub MLflow into docs/thesis/figures/<model>/ for the thesis.

Auth (env):
  MLFLOW_TRACKING_URI=https://dagshub.com/hayn404/predictive-mental-health-smartwatch.mlflow
  MLFLOW_TRACKING_USERNAME=hayn404
  MLFLOW_TRACKING_PASSWORD=<your DagsHub token>

Run:
  python ml/ci/pull_figures.py
"""
import os
import shutil
import sys
from pathlib import Path

MODELS = ["stress", "depression", "sleep", "focus", "bioage"]
OUT = Path("docs/thesis/figures")


def main():
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not uri:
        sys.exit("Set MLFLOW_TRACKING_URI (+ MLFLOW_TRACKING_USERNAME / _PASSWORD).")
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(uri)
    client = MlflowClient()
    pulled = {}
    for m in MODELS:
        exp = client.get_experiment_by_name(f"seren-{m}-train")
        if exp is None:
            print(f"[{m}] no experiment 'seren-{m}-train' — skipping")
            continue
        runs = client.search_runs([exp.experiment_id],
                                  order_by=["attributes.start_time DESC"], max_results=10)
        # pick the most recent run that actually has figures
        got = 0
        for run in runs:
            try:
                local = mlflow.artifacts.download_artifacts(
                    run_id=run.info.run_id, artifact_path="figures")
            except Exception:
                continue
            pngs = list(Path(local).glob("*.png"))
            if not pngs:
                continue
            dest = OUT / m
            dest.mkdir(parents=True, exist_ok=True)
            for f in pngs:
                shutil.copy(f, dest / f.name)
            got = len(pngs)
            print(f"[{m}] {got} figures <- run {run.info.run_id[:8]} -> {dest}")
            break
        pulled[m] = got
        if not got:
            print(f"[{m}] no figures artifact found in any recent run")

    print("\nSummary:", {k: f"{v} figs" for k, v in pulled.items()})
    missing = [k for k, v in pulled.items() if not v]
    if missing:
        print(f"Missing figures for: {missing} "
              "(re-run those models in CI so they log figures to MLflow).")


if __name__ == "__main__":
    main()
