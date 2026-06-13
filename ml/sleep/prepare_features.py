#!/usr/bin/env python3
"""
prepare_features.py — Stream Walch + BIDSleep from PhysioNet, extract the 11-feature
sleep-staging dataset, and write Kaggle-ready cache pickles.

Run this on a high-bandwidth machine (no GPU required). The output is the same .pkl
format that kaggle_train.ipynb's `cached()` helper expects, so you just upload it to a
Kaggle Dataset and the notebook will load it instantly instead of streaming.

Usage
-----
    pip install numpy pandas scipy requests
    export PHYSIONET_USER=your_pn_username
    export PHYSIONET_PASS=your_pn_password
    python prepare_features.py --out ./seren-sleep-cache --workers 20

Smoke test (only 3 nights of BIDSleep + 3 Walch subjects, ~2 min):
    python prepare_features.py --out ./tmp_cache --smoke

Then on Kaggle:
    1. Upload the contents of --out as a Kaggle Dataset (e.g. "seren-sleep-cache").
    2. In kaggle_train.ipynb, "+ Add Input" → that dataset.
    3. Insert a one-line cell BEFORE the Config cell:
           import shutil
           shutil.copytree("/kaggle/input/seren-sleep-cache",
                           "/kaggle/working/cache", dirs_exist_ok=True)
    4. Run the notebook. `cached()` finds the pickles, skips streaming.

Resources
---------
RAM: ~200 MB per worker peak (motion.csv parse). With --workers 20 → ~4 GB peak. Final
output ≈ 15 MB total. Comfortably under any 30 GB cap.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import pickle
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import scipy.io as sio


# ─────────────────────────────────────────────────────────────────────────────
# Constants — MUST match kaggle_train.ipynb (the on-device TS code mirrors these)
# ─────────────────────────────────────────────────────────────────────────────
WALCH_BASE    = "https://physionet.org/files/sleep-accel/1.0.0/"
BIDSLEEP_BASE = "https://physionet.org/files/bidsleep-dataset/1.0.0/"

EPOCH_SEC     = 30
HR_WIN_SEC    = 120
MOVE_THRESH_G = 0.02

NUM_CLASSES = 4
STAGE_NAMES = ["Wake", "Light", "Deep", "REM"]

WALCH_STAGE_REMAP    = {0: 0, 1: 1, 2: 1, 3: 2, 4: 2, 5: 3}      # 5=REM, -1 skipped
BIDSLEEP_STAGE_REMAP = {0: 0, 1: 1, 2: 1, 3: 2, 4: 3}             # 4=REM (Dreem AASM)
BIDSLEEP_LABEL_KEY = "expert_label"
BIDSLEEP_START_KEY = "recStart"

FEATURE_NAMES = [
    "hr_mean", "hr_std", "hr_min", "hr_max", "hr_range",
    "hr_succdiff_std", "hr_delta_prev",
    "act_count", "immobility_frac", "act_max", "act_std",
]
NUM_FEATURES = len(FEATURE_NAMES)   # 11

_NUM_START = set(b"-.0123456789")


# ─────────────────────────────────────────────────────────────────────────────
# HTTP helpers
# ─────────────────────────────────────────────────────────────────────────────
def make_session(user: str | None, password: str | None) -> requests.Session:
    s = requests.Session()
    if user and password:
        s.auth = (user, password)
    # Bigger pool so many concurrent workers don't queue
    a = requests.adapters.HTTPAdapter(pool_connections=64, pool_maxsize=64)
    s.mount("http://",  a)
    s.mount("https://", a)
    return s


SESSION: requests.Session  # set in main()


def http_get(url: str, tries: int = 4, timeout: int = 180) -> bytes:
    last = None
    for k in range(tries):
        try:
            r = SESSION.get(url, timeout=timeout)
            if r.status_code == 200:
                return r.content
            if r.status_code in (401, 403):
                raise RuntimeError(
                    f"HTTP {r.status_code} on {url} — set PHYSIONET_USER/PHYSIONET_PASS env vars."
                )
            last = f"HTTP {r.status_code}"
        except requests.RequestException as e:
            last = str(e)
        time.sleep(2 * (k + 1))
    raise RuntimeError(f"Failed to fetch {url} ({last})")


def list_files(base: str) -> list[str]:
    """All file paths in a PhysioNet dataset, parsed from its SHA256SUMS manifest."""
    txt = http_get(base + "SHA256SUMS.txt").decode("utf-8", "ignore")
    out = []
    for line in txt.splitlines():
        parts = line.split(None, 1)
        if len(parts) == 2:
            out.append(parts[1].strip().lstrip("*"))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction (identical to the notebook — keep in sync!)
# ─────────────────────────────────────────────────────────────────────────────
def acc_magnitude(xyz: np.ndarray) -> np.ndarray:
    return np.sqrt((xyz.astype(np.float64) ** 2).sum(axis=1))


def extract_features_for_night(hr_t, hr_v, acc_t, acc_mag, label_t, label_raw, remap):
    feats, labs = [], []
    prev = None
    o = np.argsort(hr_t);  hr_t,  hr_v    = hr_t[o],  hr_v[o]
    o = np.argsort(acc_t); acc_t, acc_mag = acc_t[o], acc_mag[o]

    for et, raw in zip(label_t, label_raw):
        cls = remap.get(int(raw), None) if not np.isnan(raw) else None
        if cls is None:
            prev = None
            continue
        c = et + EPOCH_SEC / 2.0
        lo, hi = np.searchsorted(hr_t, c - HR_WIN_SEC), np.searchsorted(hr_t, c + HR_WIN_SEC)
        hr_seg = hr_v[lo:hi]
        a_lo, a_hi = np.searchsorted(acc_t, et), np.searchsorted(acc_t, et + EPOCH_SEC)
        a_seg = acc_mag[a_lo:a_hi]
        if len(hr_seg) == 0:
            prev = None
            continue

        hr_mean = float(np.mean(hr_seg))
        hr_std  = float(np.std(hr_seg))         if len(hr_seg) > 1 else 0.0
        hr_min  = float(np.min(hr_seg))
        hr_max  = float(np.max(hr_seg))
        hr_rng  = hr_max - hr_min
        hr_sdsd = float(np.std(np.diff(hr_seg))) if len(hr_seg) > 2 else 0.0
        hr_dlt  = (hr_mean - prev) if prev is not None else 0.0

        if len(a_seg) > 1:
            d = np.abs(np.diff(a_seg))
            act_count = float(d.sum())
            immob     = float((d < MOVE_THRESH_G).mean())
            act_max   = float(d.max())
            act_std   = float(a_seg.std())
        else:
            act_count = immob = act_max = act_std = 0.0

        feats.append([hr_mean, hr_std, hr_min, hr_max, hr_rng, hr_sdsd, hr_dlt,
                      act_count, immob, act_max, act_std])
        labs.append(cls)
        prev = hr_mean

    if not feats:
        return None, None
    return np.asarray(feats, dtype=np.float32), np.asarray(labs, dtype=np.int64)


def robust_normalize_features(feats: np.ndarray) -> np.ndarray:
    """Per-night median/MAD z-score + replace any NaN/Inf with 0 (= median in normalized
    space). Apple-Watch HR/motion gaps cause occasional NaN; this keeps training stable."""
    med = np.median(feats, axis=0)
    mad = np.median(np.abs(feats - med), axis=0)
    scale = np.where(mad > 1e-8, 1.4826 * mad, 1.0)
    out = ((feats - med) / scale).astype(np.float32)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Walch / Sleep-Accel
# ─────────────────────────────────────────────────────────────────────────────
def _loadtxt_url(url: str, delimiter=None) -> np.ndarray:
    raw = http_get(url).decode("utf-8", "ignore")
    a = np.loadtxt(io.StringIO(raw), delimiter=delimiter)
    return a.reshape(-1, a.shape[-1]) if a.ndim == 1 else a


def _walch_subject(lp: str, paths: set[str]):
    sid = lp.split("/")[-1].split("_")[0]
    hp, mp = f"heart_rate/{sid}_heartrate.txt", f"motion/{sid}_acceleration.txt"
    if hp not in paths or mp not in paths:
        return None
    lab = _loadtxt_url(WALCH_BASE + lp)
    hr  = _loadtxt_url(WALCH_BASE + hp, delimiter=",")
    mo  = _loadtxt_url(WALCH_BASE + mp)
    f, l = extract_features_for_night(hr[:, 0], hr[:, 1], mo[:, 0], acc_magnitude(mo[:, 1:4]),
                                      lab[:, 0], lab[:, 1], WALCH_STAGE_REMAP)
    if f is None:
        return None
    return (sid, robust_normalize_features(f), l)


def load_walch(workers: int, limit: int | None = None):
    paths = set(list_files(WALCH_BASE))
    labels = sorted(p for p in paths if p.endswith("labeled_sleep.txt"))
    if limit:
        labels = labels[:limit]
    out, done = [], 0
    print(f"\n[walch] streaming {len(labels)} subjects with {workers} workers…", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_walch_subject, lp, paths): lp for lp in labels}
        for fut in as_completed(futs):
            try:
                r = fut.result()
                if r is not None:
                    out.append(r)
            except Exception as e:
                print(f"  ! failed {futs[fut]}: {e}", flush=True)
            done += 1
            if done % 5 == 0 or done == len(labels):
                print(f"  [{done}/{len(labels)}] kept {len(out)}", flush=True)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# BIDSleep
# ─────────────────────────────────────────────────────────────────────────────
def _read_csv_bytes(content: bytes, value_cols: list[int]):
    skiprows = 0 if content[:1] in _NUM_START else 1
    df = pd.read_csv(io.BytesIO(content), header=None, skiprows=skiprows, low_memory=False)
    t = df.iloc[:, 0].to_numpy(dtype=np.float64)
    if t.max() > 1e12:
        t = t / 1000.0
    vals = df.iloc[:, value_cols].to_numpy(dtype=np.float64)
    return t, vals


def _bidsleep_night(mp: str, pset: set[str], verbose: bool = False):
    night = mp.rsplit("/", 1)[0] if "/" in mp else ""
    hr_p = f"{night}/hr.csv"     if night else "hr.csv"
    mo_p = f"{night}/motion.csv" if night else "motion.csv"
    if hr_p not in pset or mo_p not in pset:
        return None

    m = sio.loadmat(io.BytesIO(http_get(BIDSLEEP_BASE + mp)))
    label_raw = np.ravel(np.asarray(m[BIDSLEEP_LABEL_KEY]).astype(np.float64))
    label_t   = np.arange(len(label_raw), dtype=np.float64) * EPOCH_SEC

    hr_t, hr_v = _read_csv_bytes(http_get(BIDSLEEP_BASE + hr_p), [1])
    mo_t, mo_v = _read_csv_bytes(http_get(BIDSLEEP_BASE + mo_p), [1, 2, 3])

    rec_utc  = pd.Timestamp(str(np.ravel(m[BIDSLEEP_START_KEY])[0]), tz="UTC").timestamp()
    first    = float(min(hr_t.min(), mo_t.min()))
    rec_unix = rec_utc + round((first - rec_utc) / 900.0) * 900.0
    hr_t, mo_t = hr_t - rec_unix, mo_t - rec_unix

    if verbose:
        print("  stage codes:", sorted(set(label_raw.astype(int).tolist())),
              "| HR rel [%.0f,%.0f]s | labels [0,%.0f]s"
              % (hr_t.min(), hr_t.max(), label_t[-1]), flush=True)

    f, l = extract_features_for_night(hr_t, hr_v[:, 0], mo_t, acc_magnitude(mo_v),
                                      label_t, label_raw, BIDSLEEP_STAGE_REMAP)
    if f is None:
        return None
    return (night or mp, robust_normalize_features(f), l)


def load_bidsleep(workers: int, limit: int | None = None):
    paths = list_files(BIDSLEEP_BASE)
    pset  = set(paths)
    mats  = sorted(p for p in paths if p.endswith(".mat"))
    if limit:
        mats = mats[:limit]
    if not mats:
        return []

    print(f"\n[bidsleep] streaming {len(mats)} nights with {workers} workers…", flush=True)

    # First night serial → sanity print + format validation
    out, done = [], 1
    first = _bidsleep_night(mats[0], pset, verbose=True)
    if first is not None:
        out.append(first)
    print(f"  [1/{len(mats)}] kept {len(out)} (warm-up)", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_bidsleep_night, mp, pset): mp for mp in mats[1:]}
        for fut in as_completed(futs):
            try:
                r = fut.result()
                if r is not None:
                    out.append(r)
            except Exception as e:
                print(f"  ! failed {futs[fut]}: {e}", flush=True)
            done += 1
            if done % 20 == 0 or done == len(mats):
                print(f"  [{done}/{len(mats)}] kept {len(out)}", flush=True)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Reporting / saving
# ─────────────────────────────────────────────────────────────────────────────
def summarize(name: str, recs):
    if not recs:
        print(f"\n[{name}] EMPTY")
        return {}
    all_l = np.concatenate([l for _, _, l in recs])
    all_f = np.concatenate([f for _, f, _ in recs])
    counts = np.bincount(all_l, minlength=NUM_CLASSES)
    epcs = [len(l) for _, _, l in recs]
    info = {
        "recordings": len(recs),
        "epochs_total": int(len(all_l)),
        "class_counts": dict(zip(STAGE_NAMES, counts.astype(int).tolist())),
        "class_pct":    {n: round(100*c/counts.sum(), 1)
                         for n, c in zip(STAGE_NAMES, counts.tolist())},
        "epochs_per_recording": {"min": min(epcs),
                                 "median": int(np.median(epcs)),
                                 "max": max(epcs)},
        "nan_or_inf": int(np.isnan(all_f).sum() + np.isinf(all_f).sum()),
    }
    print(f"\n[{name}] {info['recordings']} recordings, "
          f"{info['epochs_total']:,} epochs, NaN/Inf={info['nan_or_inf']}")
    print(f"  class %: {info['class_pct']}")
    print(f"  epochs/rec: {info['epochs_per_recording']}")
    return info


def save_pickle(path: Path, data):
    with open(path, "wb") as f:
        pickle.dump(data, f, protocol=4)
    print(f"  → {path} ({path.stat().st_size/1e6:.2f} MB)")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--out", default="./seren-sleep-cache",
                    help="Output directory (will be created). Upload its contents to Kaggle.")
    ap.add_argument("--workers", type=int, default=20,
                    help="Parallel HTTP workers (default 20; high-bandwidth machines can go to 32).")
    ap.add_argument("--datasets", default="walch,bidsleep",
                    help="Comma-separated subset: walch,bidsleep")
    ap.add_argument("--smoke", action="store_true",
                    help="Smoke test: 3 Walch subjects + 3 BIDSleep nights (~2 min).")
    ap.add_argument("--walch-limit",    type=int, default=None)
    ap.add_argument("--bidsleep-limit", type=int, default=None)
    ap.add_argument("--user",     default=os.environ.get("PHYSIONET_USER"))
    ap.add_argument("--password", default=os.environ.get("PHYSIONET_PASS"))
    args = ap.parse_args()

    global SESSION
    SESSION = make_session(args.user, args.password)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        args.walch_limit, args.bidsleep_limit = 3, 3

    datasets = {d.strip() for d in args.datasets.split(",")}
    summary = {}
    t0 = time.time()

    if "walch" in datasets:
        recs = load_walch(args.workers, args.walch_limit)
        summary["walch"] = summarize("walch", recs)
        save_pickle(out / "walch_features.pkl", recs)

    if "bidsleep" in datasets:
        recs = load_bidsleep(args.workers, args.bidsleep_limit)
        summary["bidsleep"] = summarize("bidsleep", recs)
        save_pickle(out / "bidsleep_features.pkl", recs)

    # Manifest documenting what's inside (for the Kaggle Dataset description + the model card)
    manifest = {
        "format_version": "1.0",
        "feature_names":   FEATURE_NAMES,
        "num_features":    NUM_FEATURES,
        "epoch_seconds":   EPOCH_SEC,
        "hr_window_sec":   HR_WIN_SEC,
        "move_threshold_g": MOVE_THRESH_G,
        "num_classes":     NUM_CLASSES,
        "stage_names":     STAGE_NAMES,
        "walch_stage_remap":    {str(k): v for k, v in WALCH_STAGE_REMAP.items()},
        "bidsleep_stage_remap": {str(k): v for k, v in BIDSLEEP_STAGE_REMAP.items()},
        "produced_at_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "summary":         summary,
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    with open(out / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n  → {out / 'manifest.json'}")

    print(f"\n✓ Done in {manifest['elapsed_seconds']}s. Files in {out.resolve()}:")
    for p in sorted(out.iterdir()):
        print(f"    {p.name:30s} {p.stat().st_size/1e6:>8.2f} MB")
    print(
        "\nNext steps on Kaggle:\n"
        "  1. Upload these files as a Kaggle Dataset (e.g. 'seren-sleep-cache').\n"
        "  2. In kaggle_train.ipynb → '+ Add Input' → that dataset.\n"
        "  3. Insert a new cell BEFORE the Config cell:\n"
        "       import shutil\n"
        "       shutil.copytree('/kaggle/input/seren-sleep-cache',\n"
        "                       '/kaggle/working/cache', dirs_exist_ok=True)\n"
        "  4. Run the notebook — load_walch/load_bidsleep will skip streaming.\n"
    )


if __name__ == "__main__":
    sys.exit(main())
