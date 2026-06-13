"""Local validation of the Walch feature pipeline on the downloaded dataset.
Mirrors the notebook's feature extraction so we catch bugs before Kaggle."""
import numpy as np
from pathlib import Path

WALCH_DIR = Path(r"C:\Users\Haneen\Downloads\motion-and-heart-rate-from-a-wrist-worn-wearable-and-labeled-sleep-from-polysomnography-1.0.0")

EPOCH_SEC, HR_WIN_SEC, MOVE_THRESH_G = 30, 120, 0.02
NUM_CLASSES = 4
STAGE_NAMES = ["Wake", "Light", "Deep", "REM"]
WALCH_STAGE_REMAP = {0: 0, 1: 1, 2: 1, 3: 2, 4: 2, 5: 3}


def acc_magnitude(xyz):
    return np.sqrt((xyz.astype(np.float64) ** 2).sum(axis=1))


def extract_features_for_night(hr_t, hr_v, acc_t, acc_mag, label_t, label_raw, remap):
    feats, labs = [], []
    prev = None
    o = np.argsort(hr_t); hr_t, hr_v = hr_t[o], hr_v[o]
    o = np.argsort(acc_t); acc_t, acc_mag = acc_t[o], acc_mag[o]
    for et, raw in zip(label_t, label_raw):
        cls = remap.get(int(raw), None) if not np.isnan(raw) else None
        if cls is None:
            prev = None; continue
        c = et + EPOCH_SEC / 2.0
        lo, hi = np.searchsorted(hr_t, c - HR_WIN_SEC), np.searchsorted(hr_t, c + HR_WIN_SEC)
        hr_seg = hr_v[lo:hi]
        alo, ahi = np.searchsorted(acc_t, et), np.searchsorted(acc_t, et + EPOCH_SEC)
        a_seg = acc_mag[alo:ahi]
        if len(hr_seg) == 0:
            prev = None; continue
        hr_mean = float(np.mean(hr_seg)); hr_std = float(np.std(hr_seg)) if len(hr_seg) > 1 else 0.0
        hr_min, hr_max = float(np.min(hr_seg)), float(np.max(hr_seg))
        hr_sdsd = float(np.std(np.diff(hr_seg))) if len(hr_seg) > 2 else 0.0
        hr_dlt = (hr_mean - prev) if prev is not None else 0.0
        if len(a_seg) > 1:
            d = np.abs(np.diff(a_seg))
            act_count, immob, act_max, act_std = float(d.sum()), float((d < MOVE_THRESH_G).mean()), float(d.max()), float(a_seg.std())
        else:
            act_count = immob = act_max = act_std = 0.0
        feats.append([hr_mean, hr_std, hr_min, hr_max, hr_max - hr_min, hr_sdsd, hr_dlt,
                      act_count, immob, act_max, act_std])
        labs.append(cls); prev = hr_mean
    if not feats:
        return None, None
    return np.asarray(feats, np.float32), np.asarray(labs, np.int64)


out, n_accel_samples = [], []
label_files = sorted((WALCH_DIR / "labels").glob("*labeled_sleep.txt"))
for lf in label_files:
    sid = lf.name.split("_")[0]
    hf = WALCH_DIR / "heart_rate" / f"{sid}_heartrate.txt"
    mf = WALCH_DIR / "motion" / f"{sid}_acceleration.txt"
    if not (hf.exists() and mf.exists()):
        print(f"  {sid}: missing hr/motion"); continue
    lab = np.loadtxt(lf); lab = lab.reshape(-1, 2) if lab.ndim == 1 else lab
    hr = np.loadtxt(hf, delimiter=","); hr = hr.reshape(-1, 2) if hr.ndim == 1 else hr
    mo = np.loadtxt(mf)
    f, l = extract_features_for_night(hr[:, 0], hr[:, 1], mo[:, 0], acc_magnitude(mo[:, 1:4]),
                                      lab[:, 0], lab[:, 1], WALCH_STAGE_REMAP)
    if f is None:
        print(f"  {sid}: 0 epochs kept"); continue
    out.append((sid, f, l))
    # how many accel samples fall in the labeled (positive-time) window?
    n_accel_in = int(((mo[:, 0] >= 0) & (mo[:, 0] <= lab[:, 0].max())).sum())
    n_accel_samples.append(n_accel_in)

print(f"\nParsed {len(out)} / {len(label_files)} subjects")
if out:
    all_l = np.concatenate([l for _, _, l in out])
    all_f = np.concatenate([f for _, f, _ in out])
    counts = np.bincount(all_l, minlength=NUM_CLASSES)
    print("Total epochs:", len(all_l))
    print("Class dist:", dict(zip(STAGE_NAMES, counts.tolist())),
          "| Deep%% = %.1f" % (100 * counts[2] / counts.sum()))
    epcs = [len(l) for _, _, l in out]
    print(f"Epochs/subject: min={min(epcs)} median={int(np.median(epcs))} max={max(epcs)}")
    print("NaN/Inf in features:", int(np.isnan(all_f).sum() + np.isinf(all_f).sum()))
    print("Accel samples in labeled window (per subj): min=%d median=%d max=%d"
          % (min(n_accel_samples), int(np.median(n_accel_samples)), max(n_accel_samples)))
    FEATURE_NAMES = ["hr_mean", "hr_std", "hr_min", "hr_max", "hr_range", "hr_sdsd",
                     "hr_delta", "act_count", "immob_frac", "act_max", "act_std"]
    print("\nPer-feature mean / std (pre-normalization):")
    for i, nm in enumerate(FEATURE_NAMES):
        print(f"  {nm:12s} mean={all_f[:, i].mean():10.4f}  std={all_f[:, i].std():10.4f}")
