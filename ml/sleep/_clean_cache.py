"""Sanitize NaN/Inf in the prepare_features.py cache pickles.

Apple-Watch HR/motion streams have small gaps that propagate to a tiny fraction of
feature values. Training would NaN-out on the first such batch. This script replaces
NaN/Inf with 0 (= the median in normalized space) in-place.

Usage:  python _clean_cache.py <cache_dir>
"""
import pickle
import sys
from pathlib import Path
import numpy as np

cache = Path(sys.argv[1] if len(sys.argv) > 1
             else r"C:\Users\Haneen\Downloads\seren-sleep-cache")

for name in ["walch_features.pkl", "bidsleep_features.pkl"]:
    p = cache / name
    if not p.exists():
        print(f"  skip (missing): {p}")
        continue
    with open(p, "rb") as fp:
        data = pickle.load(fp)
    n_bad = 0
    cleaned = []
    for sid, feats, labs in data:
        bad = int(np.isnan(feats).sum() + np.isinf(feats).sum())
        n_bad += bad
        f = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        cleaned.append((sid, f, labs))
    with open(p, "wb") as fp:
        pickle.dump(cleaned, fp, protocol=4)
    print(f"  {name}: {len(data)} recordings, cleaned {n_bad} NaN/Inf values "
          f"({p.stat().st_size/1e6:.2f} MB)")

print("\nDone. Cache is now safe to upload to Kaggle.")
