#!/usr/bin/env bash
# One-time MLOps bootstrap: DVC + DagsHub. Run LOCALLY, once, with the training
# data present on disk. Afterwards, training/gating/deploy run in GitHub Actions.
#
#   export DAGSHUB_USER=hayn404
#   export DAGSHUB_TOKEN=<your dagshub token>
#   bash scripts/setup-mlops.sh
set -euo pipefail

: "${DAGSHUB_USER:?set DAGSHUB_USER}"
: "${DAGSHUB_TOKEN:?set DAGSHUB_TOKEN}"
REPO="predictive-mental-health-smartwatch"

pip install "dvc[s3]" dagshub

# init DVC (no-op if already initialised)
dvc init || true

# DagsHub S3-compatible DVC remote
dvc remote add -d origin s3://dvc 2>/dev/null || true
dvc remote modify origin endpointurl "https://dagshub.com/${DAGSHUB_USER}/${REPO}.s3"
dvc remote modify origin --local access_key_id     "${DAGSHUB_TOKEN}"
dvc remote modify origin --local secret_access_key "${DAGSHUB_TOKEN}"

# Track the training DATA (edit these paths to match your local caches).
# These are the feature caches that used to live as Kaggle datasets.
for f in \
  data/sleep/bidsleep_features.pkl \
  data/sleep/walch_features.pkl \
  data/stress \
  data/depresjon \
  data/cogwear ; do
  [ -e "$f" ] && dvc add "$f" || echo "skip (missing): $f"
done

# data/stress/ should hold the three source datasets (download once, then this
# tracks them on DagsHub so CI never touches Kaggle again):
#   - WESAD            (kaggle: orvile/wesad-wearable-stress-affect-detection-dataset) -> held-out test
#   - Stress-Predict   (kaggle: dohahemdan17/stress-predict-dataset)                   -> SIPD (train)
#   - PhysioStress     (kaggle: dohahemdan17/wearable-dataset)                          -> train (fusion)

git add .dvc .dvcignore dvc.yaml params.yaml 2>/dev/null || true
git add data/*.dvc data/**/*.dvc data/.gitignore 2>/dev/null || true

dvc push   # upload data to DagsHub

cat <<'EOF'

Done. Next:
  1. git commit -m "chore: track training data with DVC on DagsHub" && git push
  2. Add GitHub secrets: DAGSHUB_TOKEN, EXPO_TOKEN, GH_PAT (repo+workflow scope)
  3. Register your staging machine: Settings -> Actions -> Runners -> New self-hosted runner
     (label it `gpu` for the sleep model)
  4. Protect main: require the `gate` status check before merge
EOF
