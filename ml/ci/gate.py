#!/usr/bin/env python3
"""
Champion/challenger model gate.

Fails (exit 1) if any model's PRIMARY metric regressed versus the version on
`origin/main`. Run in CI on pull requests that touch model metrics/assets.

Contract: every trained model writes a small, git-tracked `ml/<model>/metrics.json`:

    {
      "eval_set_id": "walch_v1",          # pin the test set — changing it blocks the gate
      "walch_kappa": 0.482,               # ...the metrics, flat keys...
      "walch_weightedF1": 0.665,
      "gate": {
        "primary": "walch_kappa",         # the metric that decides promotion
        "direction": "max",               # "max" (accuracy/AUC/F1/κ) or "min" (MAE/loss)
        "min_delta": 0.0,                 # require at least this much improvement
        "no_regress": ["walch_weightedF1"]# secondary metrics that may not drop
      }
    }

The model file itself can be DVC-tracked (large); this metrics.json stays in git
so the gate can diff it against main without pulling artifacts.
"""
import glob
import json
import subprocess
import sys


def champion(path: str):
    """The metrics.json as it exists on origin/main, or None if the model is new."""
    git_path = path.replace("\\", "/")  # git wants forward slashes (Windows glob gives \)
    try:
        blob = subprocess.check_output(
            ["git", "show", f"origin/main:{git_path}"], stderr=subprocess.DEVNULL
        )
        return json.loads(blob)
    except subprocess.CalledProcessError:
        return None


def is_better(direction: str, cand: float, champ: float, min_delta: float) -> bool:
    if direction == "max":
        return cand >= champ + min_delta
    return cand <= champ - min_delta


def main() -> int:
    paths = sorted(glob.glob("ml/**/metrics.json", recursive=True))
    if not paths:
        print("no ml/**/metrics.json found — nothing to gate")
        return 0

    failed = False
    for path in paths:
        cand = json.load(open(path))
        gate = cand.get("gate")
        if not gate:
            print(f"  -  {path}: no `gate` block, skipping")
            continue

        champ = champion(path)
        if champ is None:
            print(f"  OK {path}: new model (no champion on main) -> allowed")
            continue

        if cand.get("eval_set_id") != champ.get("eval_set_id"):
            print(
                f"  XX {path}: eval_set_id changed "
                f"({champ.get('eval_set_id')} -> {cand.get('eval_set_id')}) "
                "-> not comparable, blocking"
            )
            failed = True
            continue

        key = gate["primary"]
        direction = gate.get("direction", "max")
        min_delta = float(gate.get("min_delta", 0.0))
        cv, bv = float(cand[key]), float(champ[key])
        ok = is_better(direction, cv, bv, min_delta)
        mark = "OK" if ok else "XX"
        print(f"  {mark} {path}: {key} {cv:.4f} vs champion {bv:.4f} "
              f"({direction}, min_delta {min_delta})")
        if not ok:
            failed = True

        for k in gate.get("no_regress", []):
            if k in cand and k in champ and float(cand[k]) < float(champ[k]):
                print(f"       XX secondary `{k}` regressed: {cand[k]:.4f} < {champ[k]:.4f}")
                failed = True

    if failed:
        print("\nGATE FAILED: a model regressed. The new model will NOT ship.")
        return 1
    print("\nGATE PASSED: all models better-or-equal. Safe to merge + deploy.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
