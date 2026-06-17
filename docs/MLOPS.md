# Seren MLOps — train on GitHub Actions, version with DVC + DagsHub, ship via EAS

A new model is **trained on a self-hosted GitHub Actions runner** (your staging
machine), its data + artifacts are **versioned with DVC on DagsHub**, metrics are
logged to **DagsHub MLflow**, and the model **ships only if it beats the deployed
one** — then it goes to phones **over-the-air via EAS Update**.

```
 train.yml (self-hosted runner)         model-gate.yml (PR)            deploy.yml (main)
 ──────────────────────────────         ───────────────────           ─────────────────
 dvc pull  (data from DagsHub)   ─PR─▶  gate.py: candidate vs   ─merge─▶ dvc pull (model)
 dvc repro (train+eval+export)          champion on main;                eas update
 dvc push  (artifacts → DagsHub)        FAIL if any primary              (OTA → installs
 MLflow log + open candidate PR         metric regressed                 get the model)
```

## Files
| File | Role |
|---|---|
| `dvc.yaml` | Pipeline stages (train → eval → export). `dvc repro` runs them. |
| `params.yaml` | Hyperparameters per model. |
| `ml/<model>/metrics.json` | Held-out metrics + a `gate` block. **Git-tracked** (small) so the gate can diff it. |
| `ml/ci/gate.py` | Champion/challenger comparison. Exit 1 = regression = PR blocked. |
| `ml/ci/requirements.txt` | CI training deps (extends `ml/requirements.txt` + DVC + torch). |
| `.github/workflows/train.yml` | Manual trigger → train on self-hosted runner → candidate PR. |
| `.github/workflows/model-gate.yml` | PR check (make it **required** on `main`). |
| `.github/workflows/deploy.yml` | On merge → `dvc pull` model → `eas update`. |
| `scripts/setup-mlops.sh` | One-time DVC + DagsHub bootstrap (run locally). |

## One-time setup
1. `export DAGSHUB_USER=hayn404 DAGSHUB_TOKEN=<token> && bash scripts/setup-mlops.sh`
   then commit the `.dvc` pointers.
2. **GitHub secrets** (Settings → Secrets → Actions): `DAGSHUB_TOKEN`, `EXPO_TOKEN`,
   and `GH_PAT` (a PAT with `repo` + `workflow` scope — needed so the auto-PR
   triggers the gate).
3. **Self-hosted runner:** Settings → Actions → Runners → New self-hosted runner,
   label it `gpu`. Install Python, Node, DVC, and your CUDA stack on it once.
4. **Branch protection:** protect `main`, require the `gate` status check.
5. **EAS Update:** ensure `expo-updates` is configured (update URL + `production`
   channel) so `eas update` reaches installs.

## The training-script contract (wire each model once)
A `dvc.yaml` stage calls `python ml/<model>/train.py ...`. That script must:
- read hyperparameters from `params.yaml` (the stage's `params:` block),
- read its data from `data/<model>/` (DVC-pulled — **not** Kaggle),
- write the exported model into `assets/ml/<model>/`,
- write `ml/<model>/metrics.json` with the held-out metrics **and** a `gate` block:
  ```json
  { "eval_set_id": "walch_v1", "walch_kappa": 0.482,
    "gate": { "primary": "walch_kappa", "direction": "max",
              "min_delta": 0.0, "no_regress": ["walch_weightedF1"] } }
  ```
`direction` is `max` for accuracy/AUC/F1/κ and `min` for MAE/loss. Changing
`eval_set_id` blocks the gate (prevents "better" scores on an easier test set).

Sleep is wired (`train_sleep` stage + `ml/sleep/metrics.json` baseline). Replicate
the stage for stress / focus / bio-age / depression by uncommenting the templates
in `dvc.yaml` once each `train.py` honours the contract above.

## Notes / limits
- **GitHub-hosted runners have no GPU** — the sleep deep model needs the
  self-hosted `gpu` runner. XGBoost models run on any runner.
- **DagsHub free tier** has storage/compute limits — version the extracted
  feature caches (tens–hundreds of MB), not multi-GB raw dumps.
- The deployed `.tflite`/JSON are DVC-tracked; `deploy.yml` `dvc pull`s them before
  `eas update`. Keep them small.
