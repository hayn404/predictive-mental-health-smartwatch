# ML — per-component layout

Each model/component is self-contained under its own folder:

```
ml/
  stress/        # PPG/HRV stress model — the primary, most developed component
    src/         #   feature extraction, preprocessing, training
    kaggle/      #   self-contained Kaggle notebook (sole model producer) + validator
    config/      #   pipeline config
    models/      #   exported stress_model.json + metadata (joblib artifacts are git-ignored)
    tests/       #   pytest unit tests
  sleep/         # on-device sleep-stage model (training + export pipeline)
  anxiety/
    src/         #   feature extraction + training (reuses ml/stress/src/features.py)
    models/      #   exported anxiety_model*.json + training log
  focus/
    src/         #   "academic"/cognitive-readiness model (reuses ml/stress/src/features.py)
    models/      #   exported focus_model.json + training log
  voice/
    src/         #   acoustic feature extraction (pitch, energy, speech rate)
    tests/
  depression/    # placeholder — see depression/README.md (code on feature/depression_model)

  data/          # shared raw datasets (git-ignored)
  docs/          # cross-model analysis & thesis references
  scripts/       # shared utilities (e.g. convert_models.py)
  requirements.txt, Dockerfile
```

**Shared code:** anxiety and focus reuse the stress component's HRV feature extractor
(`ml/stress/src/features.py`) via a `sys.path` insert — keep that one extractor as the
single source of truth for the 29-feature vector.

**On-device side:** TypeScript inference services live in `services/ai/` (flat), and bundled
model files the app loads live in `assets/ml/<component>/`.
