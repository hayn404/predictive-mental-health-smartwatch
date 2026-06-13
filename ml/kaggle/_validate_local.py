"""
Validation: execute the notebook's CODE cells against the LOCAL WESAD data to
catch any copy-paste bug in the embedded functions. Not shipped — dev-only.
Patches the Kaggle-specific paths to local equivalents, then execs each cell.
"""
import json, sys, tempfile, types
import numpy as np
from pathlib import Path

NB = Path(__file__).parent / "seren_stress_kaggle.ipynb"
WESAD_LOCAL = Path(__file__).parents[1] / "data" / "WESAD"
OUT = Path(tempfile.mkdtemp())
SIPD_FAKE = Path(tempfile.mkdtemp()) / "sipd"


def _make_fake_sipd(root, participants=(2, 3, 4, 5, 6), seconds=240, start=1_000_000):
    """Minimal synthetic SIPD (E4 raw + labels) so the SIPD/cross-dataset cells run."""
    proc = root / "Processed_data"; proc.mkdir(parents=True)
    rows = ["Participant,HR,respr,Time(sec),Label"]
    for pid in participants:
        raw = root / "Raw_data" / f"S{pid}"; raw.mkdir(parents=True)
        t = np.arange(seconds * 64) / 64
        ph = (t % (60/70)) / (60/70)
        bvp = np.where(ph < 0.15, ph/0.15, np.exp(-(ph-0.15)*4))
        with open(raw/"BVP.csv", "w") as f:
            f.write(f"{start:.2f}\n64.00\n"); np.savetxt(f, bvp, fmt="%.5f")
        with open(raw/"TEMP.csv", "w") as f:
            f.write(f"{start:.2f}\n4.00\n"); np.savetxt(f, np.full(seconds*4, 32.0), fmt="%.3f")
        with open(raw/"ACC.csv", "w") as f:
            f.write(f"{start:.2f},{start:.2f},{start:.2f}\n32.00,32.00,32.00\n")
            np.savetxt(f, np.ones((seconds*32, 3))/np.sqrt(3), fmt="%.4f", delimiter=",")
        for s in range(seconds):
            rows.append(f"{pid},80,12,{start+s},{0 if s < seconds//2 else 1}")
    (proc/"Improved_All_Combined_hr_rsp_binary.csv").write_text("\n".join(rows))


_make_fake_sipd(SIPD_FAKE)

cells = [c for c in json.load(open(NB))["cells"] if c["cell_type"] == "code"]
g = {}
for i, c in enumerate(cells):
    src = "".join(c["source"])
    # Patch config + export paths for local execution.
    src = src.replace('WESAD_ROOT = None', f'WESAD_ROOT = r"{WESAD_LOCAL}"')
    # Point SIPD at the synthetic dataset so the SIPD + cross-dataset cells execute.
    src = src.replace('SIPD_ROOT = None', f'SIPD_ROOT = r"{SIPD_FAKE}"')
    # Shrink the tuning budget so validation runs fast (we only verify it executes).
    src = src.replace('TUNE_POP   = 8', 'TUNE_POP   = 4')
    src = src.replace('TUNE_ITERS = 12', 'TUNE_ITERS = 3')
    # Force LOCAL MLflow mode (None branch) so validation never touches DagsHub,
    # and redirect the hard-coded /kaggle/working paths to a temp dir.
    src = src.replace(
        'MLFLOW_TRACKING_URI = "https://dagshub.com/hayn404/predictive-mental-health-smartwatch.mlflow"',
        'MLFLOW_TRACKING_URI = None')
    src = src.replace('"file:/kaggle/working/mlruns"', f'"file:{OUT.as_posix()}/mlruns"')
    src = src.replace('"/kaggle/working/mlruns"', f'"{OUT.as_posix()}/mlruns"')
    src = src.replace('Path("/kaggle/working")', f'Path(r"{OUT}")')
    src = src.replace('"/kaggle/working/', f'"{OUT.as_posix()}/')
    try:
        exec(compile(src, f"<cell {i}>", "exec"), g)
    except Exception as e:
        print(f"\n!!! CELL {i} FAILED: {type(e).__name__}: {e}")
        print(src[:400])
        sys.exit(1)

print("\n=== VALIDATION OK ===")
print("exported files:", [p.name for p in OUT.glob('*.json')])
m = json.load(open(OUT / "model_metadata.json"))
print("version:", m["version"], "| n features:", len(m["features"]))
print("metrics:", {k: round(v, 4) for k, v in m["metrics"].items()})
# Confirm the SIPD + cross-dataset cells actually executed
sipd_df = g.get("sipd_df")
xds = g.get("xds") or {}
tr = g.get("tuning_results") or {}
hold = g.get("holdout")
print("SIPD rows:", 0 if sipd_df is None else len(sipd_df), "| cross-dataset results:", len(xds))
print("tuning_results:", {k: round(v["cv_auc"], 3) for k, v in tr.items()})
print("BEST_PARAMS:", g.get("BEST_PARAMS"))
print("held-out (train SIPD->test WESAD):", {k: round(v, 3) for k, v in (hold or {}).items()})
thr = g.get("BEST_THRESHOLD"); hold_cal = g.get("holdout_cal")
print("BEST_THRESHOLD:", thr, "| holdout_cal:", {k: round(v, 3) for k, v in (hold_cal or {}).items() if isinstance(v, float)})
assert sipd_df is not None and len(sipd_df) > 0, "SIPD cell did not produce data"
assert len(xds) > 0, "cross-dataset experiment did not run"
assert len(tr) == 2, "PSO+GWO tuning did not both run"
assert hold is not None, "held-out cross-dataset eval did not run"
assert thr is not None and 0 < thr < 1, "threshold calibration did not run"
assert hold_cal is not None, "calibrated held-out eval did not run"
