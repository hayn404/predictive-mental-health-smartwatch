# Depression model

Placeholder for the depression model component.

The model and training notebook currently live on the unmerged branch
`feature/depression_model` (a raw upload: notebook + `.pkl` + figures at the repo root).
Before merging, move those files here to match the per-component layout:

```
ml/depression/
  src/        # training / feature-extraction scripts
  models/     # exported model + metadata (large binaries should be git-ignored or use Git LFS)
  kaggle/     # training notebook(s)
```

The on-device inference service (when added) belongs in `services/ai/` alongside the
other model services, and any bundled model file under `assets/ml/depression/`.
