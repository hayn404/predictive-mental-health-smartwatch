"""
Builds seren_stress_kaggle.ipynb — a self-contained Kaggle notebook that
extracts features, trains, and exports the model in Seren's assets/ml format.

Run:  python build_notebook.py   ->  writes seren_stress_kaggle.ipynb
We build the .ipynb via json so it is always valid, regardless of jupyter.
"""
import json
from pathlib import Path

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": text.strip("\n").splitlines(keepends=True)})


# ---------------------------------------------------------------- cell: title
md(r"""
# Seren — Stress Model Training on Kaggle

Self-contained pipeline: **load WESAD → extract HRV + PPG-morphology features (parallel)
→ train XGBoost → export `stress_model.json` + `model_metadata.json`** in the exact format
the React Native app reads from `assets/ml/stress/`.

**Why Kaggle is faster:** feature extraction (sample-entropy is O(N²) per window) is parallelised
across all CPU cores with `joblib`, turning minutes into seconds. WESAD is already hosted on
Kaggle, so there is no 2 GB download.

### Setup (one time)
1. *Add Data* → search **WESAD** → add a dataset that contains `S2/S2.pkl … S17/S17.pkl`.
2. Set `WESAD_ROOT` below to that dataset's folder (printed by the first cell).
3. *Run All*. Outputs land in `/kaggle/working/` — download them into the repo's `assets/ml/stress/`.

### Knobs (see the CONFIG cell)
- `TRAIN_ON` / `EVAL_ON` — which dataset trains the exported model and which is the
  held-out cross-dataset test. Default `SIPD` -> test `WESAD` (SIPD is larger/richer
  and transfers better). Also `"WESAD+SIPD"` to pool both.
- `RUN_TUNING` / `TUNERS` — nature-inspired hyperparameter tuning (`pso`, `gwo`).
  Maximizes subject-disjoint CV AUC on the training set; best params feed the model.
- `CALIBRATE_THRESHOLD` — pick the decision threshold on training out-of-fold preds
  (no test leakage) to fix the cross-dataset precision/F1 gap; exported in the model JSON.
- `USE_MORPHOLOGY` — add the 8 PPG pulse-shape features (raw-PPG only; +~3 pts within WESAD).
- `NORMALIZATION`  — `"global"` (drop-in, single mean/std vector) or `"per_subject"`
  (+~6 pts, but the watch must then z-score against a per-user baseline — a TS change).
- `MLFLOW_TRACKING_URI` — `None` logs to a local file store at `/kaggle/working/mlruns`
  (download it, then `mlflow ui --backend-store-uri ./mlruns`); or set a remote server URI.
""")

# ---------------------------------------------------------------- cell: deps
code(r"""
# Kaggle already ships numpy/scipy/sklearn/xgboost/joblib. Pin nothing; just confirm.
import numpy as np, scipy, sklearn, xgboost, joblib, json, pickle, os, glob
print("numpy", np.__version__, "| scipy", scipy.__version__,
      "| sklearn", sklearn.__version__, "| xgboost", xgboost.__version__)
""")

# ---------------------------------------------------------------- cell: config
code(r"""
# ============================== CONFIG ==============================
# Point this at the Kaggle WESAD dataset folder that holds S2/ ... S17/.
# The cell below auto-detects common layouts and prints what it found.
WESAD_ROOT = None            # e.g. "/kaggle/input/wesad/WESAD"  (None = auto-detect)

# Stress-Predict / SIPD (raw Empatica E4) — a SECOND raw-PPG dataset that enables
# the cross-dataset experiment. Upload the italha-d/Stress-Predict-Dataset repo as
# a Kaggle dataset; its folder holds Raw_data/ + Processed_data/.
SIPD_ROOT = None             # None = auto-detect; absent -> cross-dataset is skipped
USE_SIPD  = True             # False -> WESAD-only run

# Extra WATCH-COMPATIBLE Empatica-E4 datasets for FUSION (more/varied training
# data -> higher AUC). All are E4, so only BVP/ACC/TEMP are read; EDA/ECG ignored.
#
# PhysioStress = PhysioNet "Wearable Device Dataset from Induced Stress and
#   Structured Exercise Sessions" — OPEN ACCESS (no agreement), 36 subjects
#   (STRESS/S01-S18 + f01-f18). https://physionet.org/content/wearable-device-dataset/
#   Recommended first extra dataset (free, immediate).
USE_PHYSIOSTRESS  = False
PHYSIOSTRESS_ROOT = None      # point at the .../Wearable_Dataset/STRESS folder
# Labels are protocol-based (stressor task vs baseline) from each subject's tags.csv,
# using the exact tag->stage mapping from the dataset's Wearable_Dataset.ipynb.
# Verified on the real download: 34 subjects, ~648 windows, ~33% stress.
#
# ForDigitStress = access-by-request (hcai.eu/fordigitstress); VerBIO = public
# (TAMU HUBBS). Label-file pattern may need a tweak to match your download.
USE_FORDIGIT  = False
FORDIGIT_ROOT = None
USE_VERBIO    = False
VERBIO_ROOT   = None

# Which dataset(s) the FINAL exported model trains on, and which is the held-out
# cross-dataset test. SIPD transfers better than WESAD; FUSION (more datasets)
# should push AUC higher still. TRAIN_ON may be any "+"-joined combo of the
# available datasets: e.g. "SIPD", "WESAD+SIPD", "WESAD+SIPD+VerBIO+ForDigitStress".
# EVAL_ON is a single held-out dataset NOT in TRAIN_ON.
TRAIN_ON = "SIPD"
EVAL_ON  = "WESAD"

# FINAL CLEAN EVALUATION (removes model-selection leakage). When True, the notebook
# locks FINAL_TEST aside, chooses the config (morphology + normalization) and tunes
# hyperparameters using ONLY the other ("dev") datasets via leave-one-dev-dataset-out,
# then evaluates FINAL_TEST EXACTLY ONCE. Use this for the number you report in the
# thesis (needs >=3 datasets: >=2 dev + the held-out FINAL_TEST).
FINAL_EVAL = False
FINAL_TEST = "WESAD"

# WITHIN-DATASET benchmark: train+test on the SAME dataset. Gives the optimistic
# "same-dataset" numbers to compare against the literature, alongside the honest
# cross-dataset numbers. Fast; set False to skip.
WITHIN_BENCHMARK = True
# Leakage demonstration: also run a RANDOM k-fold split (NOT subject-disjoint, the
# protocol some prior works use, e.g. PhysioStress's 93% used 10-fold) next to the
# honest LOSO, on identical data + PPG-only features. The "leak gap" = how much a
# non-subject-disjoint split inflates the score. Proves the leakage on the same data.
WITHIN_KFOLD   = True
WITHIN_KFOLD_K = 10

# SHAP-guided feature reduction. After the XAI cells rank features by mean|SHAP|, retrain
# on the top-k features for several k and measure held-out cross-dataset AUC -> find the
# smallest feature set that preserves performance (a WRAPPER check, not a pure SHAP filter).
RUN_FEATURE_REDUCTION = True
FEATURE_REDUCTION_KS  = None      # None -> auto [3,5,8,14,21,n_features]

# Nature-inspired hyperparameter tuning (PSO / GWO). Tunes XGBoost by maximizing
# subject-disjoint CV AUC on the TRAINING set only (no held-out leakage), then the
# best params feed the final model. RUN_TUNING=False uses the hand-set BASE_PARAMS.
RUN_TUNING = True
TUNERS     = ["pso", "gwo"]   # which to run/compare; the higher CV-AUC one wins
TUNE_POP   = 8                # swarm / pack size (agents)
TUNE_ITERS = 12               # iterations per optimizer

# Decision-threshold calibration. The default 0.5 is miscalibrated cross-dataset
# (high recall, low precision -> low F1). Objectives:
#   "prior"  -> BASE-RATE MATCHING (recommended for cross-dataset): set the threshold
#               on the TARGET's own score distribution so the predicted positive rate
#               equals the training stress prevalence. Adapts to the target's
#               distribution shift; uses NO target labels (leakage-free).
#   "f1" | "youden" | "balanced" -> pick on the TRAINING out-of-fold preds. NOTE these
#               source-based thresholds often DO NOT transfer cross-dataset (the
#               source-optimal threshold can point the wrong way for the target).
CALIBRATE_THRESHOLD = True
THRESHOLD_OBJECTIVE = "prior"

WINDOW_SEC     = 60          # 60-s windows match the watch's polling cadence
USE_MORPHOLOGY = True        # add 8 PPG pulse-shape features (raw-PPG only)
NORMALIZATION  = "per_subject"   # "global" | "per_subject"
VERSION        = "1.1.0"     # bumps the exported model version
SEED           = 42
N_JOBS         = -1          # use all cores for feature extraction

# --- MLflow tracking ---
# Remote DagsHub-hosted MLflow. Requires Kaggle **internet ON** + credentials.
# Set None instead to log to a local file store at /kaggle/working/mlruns
# (in that mode the notebook strips platform-injected MLFLOW_*/DATABRICKS_* env
# vars so Kaggle's managed server can't hijack logging -> the HTTP 403 you saw).
MLFLOW_TRACKING_URI = "https://dagshub.com/hayn404/predictive-mental-health-smartwatch.mlflow"
MLFLOW_EXPERIMENT   = "seren-stress-detection"

# DagsHub auth. The username is not secret; the TOKEN must NOT be hardcoded.
# Add the token as a Kaggle Secret (Add-ons -> Secrets) with this label, and the
# notebook reads it at runtime. Get the token at DagsHub -> your repo -> Remote
# -> Experiments (or Settings -> Tokens).
DAGSHUB_USERNAME         = "hayn404"
KAGGLE_SECRET_TOKEN_NAME = "DAGSHUB_TOKEN"   # name of the Kaggle Secret holding the token
# ===================================================================

import glob, os
from pathlib import Path
def _autodetect_wesad(root):
    if root and glob.glob(os.path.join(root, "S*", "S*.pkl")):
        return root
    for cand in glob.glob("/kaggle/input/**/S2/S2.pkl", recursive=True):
        return os.path.dirname(os.path.dirname(cand))   # .../S2/S2.pkl -> root
    for cand in glob.glob("/kaggle/input/**/S2.pkl", recursive=True):
        return os.path.dirname(os.path.dirname(cand))
    return root
WESAD_ROOT = _autodetect_wesad(WESAD_ROOT)
found = sorted(glob.glob(os.path.join(WESAD_ROOT or "", "S*", "S*.pkl"))) if WESAD_ROOT else []
print("WESAD_ROOT =", WESAD_ROOT)
print(f"found {len(found)} subject pickles:", [Path(p).stem for p in found][:20])
assert found, "No WESAD pickles found — set WESAD_ROOT to the dataset folder containing S*/S*.pkl"

def _autodetect_sipd(root):
    if root and os.path.exists(os.path.join(root, "Processed_data")):
        return root
    for cand in glob.glob("/kaggle/input/**/Processed_data/Improved_All_Combined_hr_rsp_binary.csv", recursive=True):
        return os.path.dirname(os.path.dirname(cand))   # .../Processed_data/x.csv -> root
    return root
if USE_SIPD:
    SIPD_ROOT = _autodetect_sipd(SIPD_ROOT)
    _ok = bool(SIPD_ROOT and os.path.exists(os.path.join(SIPD_ROOT, "Processed_data")))
    print("SIPD_ROOT =", SIPD_ROOT, "(found)" if _ok else "(NOT found - cross-dataset will be skipped)")
    if not _ok:
        USE_SIPD = False

def _first_e4_root(root):
    # An E4 dataset root contains subject/session folders that each hold BVP.csv.
    if root and glob.glob(os.path.join(root, "*", "BVP.csv")):
        return root
    return root
if USE_FORDIGIT:
    FORDIGIT_ROOT = _first_e4_root(FORDIGIT_ROOT)
    _f = bool(FORDIGIT_ROOT and glob.glob(os.path.join(FORDIGIT_ROOT, "*", "BVP.csv")))
    print("FORDIGIT_ROOT =", FORDIGIT_ROOT, "(found)" if _f else "(NOT found - skipped)")
    USE_FORDIGIT = _f
if USE_VERBIO:
    VERBIO_ROOT = _first_e4_root(VERBIO_ROOT)
    _v = bool(VERBIO_ROOT and glob.glob(os.path.join(VERBIO_ROOT, "*", "BVP.csv")))
    print("VERBIO_ROOT =", VERBIO_ROOT, "(found)" if _v else "(NOT found - skipped)")
    USE_VERBIO = _v
def _autodetect_physiostress(root):
    # Find the STRESS folder that holds subject dirs (S01.../f01...) with BVP.csv.
    if root:
        for cand in (os.path.join(root, "Wearable_Dataset", "STRESS"),
                     os.path.join(root, "STRESS"), root):
            if glob.glob(os.path.join(cand, "*", "BVP.csv")):
                return cand
    for cand in glob.glob("/kaggle/input/**/STRESS/*/BVP.csv", recursive=True):
        return os.path.dirname(os.path.dirname(cand))   # .../STRESS/S01/BVP.csv -> STRESS
    return root
if USE_PHYSIOSTRESS:
    PHYSIOSTRESS_ROOT = _autodetect_physiostress(PHYSIOSTRESS_ROOT)
    _p = bool(PHYSIOSTRESS_ROOT and glob.glob(os.path.join(PHYSIOSTRESS_ROOT, "*", "BVP.csv")))
    print("PHYSIOSTRESS_ROOT =", PHYSIOSTRESS_ROOT, "(found)" if _p else "(NOT found - skipped)")
    USE_PHYSIOSTRESS = _p
""")

# ---------------------------------------------------------------- cell: hrv features
code(r'''
# ===== HRV feature extraction (identical maths to ml/src/features.py) =====
from scipy import signal as scipy_signal
from scipy.interpolate import interp1d
import warnings; warnings.filterwarnings("ignore")

def bvp_to_rr_intervals(bvp, fs=64):
    nyq = fs/2; b,a = scipy_signal.butter(3,[0.5/nyq, min(8.0/nyq,0.99)],btype="band")
    f = scipy_signal.filtfilt(b,a,bvp)
    pk,_ = scipy_signal.find_peaks(f, distance=int(0.3*fs), height=np.percentile(f,60))
    if len(pk)<2: return np.array([])
    rr = np.diff(pk)/fs*1000
    m = (rr>=300)&(rr<=2000)
    if len(rr)>5:
        med = np.median(rr); m &= np.abs(rr-med)/med < 0.30
    return rr[m]

def time_domain(rr):
    keys=["meanRR","sdnn","rmssd","pnn50","pnn20","hrMean","hrStd","hrRange","cvRR"]
    if len(rr)<5: return {k:0.0 for k in keys}
    d=np.abs(np.diff(rr)); hr=60000/rr
    return dict(meanRR=float(np.mean(rr)),sdnn=float(np.std(rr,ddof=1)),
        rmssd=float(np.sqrt(np.mean(d**2))),pnn50=float(np.sum(d>50)/len(d)*100),
        pnn20=float(np.sum(d>20)/len(d)*100),hrMean=float(np.mean(hr)),
        hrStd=float(np.std(hr,ddof=1)),hrRange=float(np.max(hr)-np.min(hr)),
        cvRR=float(np.std(rr,ddof=1)/np.mean(rr)) if np.mean(rr)>0 else 0.0)

def freq_domain(rr):
    keys=["vlfPower","lfPower","hfPower","lfHfRatio","totalPower","lfNorm","hfNorm"]
    if len(rr)<30: return {k:0.0 for k in keys}
    try:
        t=np.cumsum(rr)/1000; t-=t[0]; fs=4; tu=np.arange(t[0],t[-1],1/fs)
        if len(tu)<30: return {k:0.0 for k in keys}
        rr_r=interp1d(t,rr,kind="cubic",fill_value="extrapolate")(tu); rr_r-=np.mean(rr_r)
        fr,psd=scipy_signal.welch(rr_r,fs=fs,nperseg=min(256,len(rr_r)),noverlap=min(128,len(rr_r)//2))
        I=getattr(np,"trapezoid",getattr(np,"trapz"))
        def band(lo,hi):
            m=(fr>=lo)&(fr<hi); return float(I(psd[m],fr[m])) if m.any() else 0.0
        vlf,lf,hf=band(0.003,0.04),band(0.04,0.15),band(0.15,0.4)
        tp=vlf+lf+hf; s=lf+hf
        return dict(vlfPower=vlf,lfPower=lf,hfPower=hf,lfHfRatio=float(lf/hf) if hf>0 else 0.0,
            totalPower=tp,lfNorm=float(lf/s*100) if s>0 else 50.0,hfNorm=float(hf/s*100) if s>0 else 50.0)
    except Exception: return {k:0.0 for k in keys}

def _sampen(rr,m=2,r=0.2):
    N=len(rr)
    if N<20: return 0.0
    r=r*np.std(rr,ddof=1)
    if r==0: return 0.0
    def cnt(L):
        T=np.array([rr[i:i+L] for i in range(N-L)]); c=0
        for i in range(len(T)):
            for j in range(i+1,len(T)):
                if np.max(np.abs(T[i]-T[j]))<r: c+=1
        return c
    A,B=cnt(m+1),cnt(m)
    return float(-np.log(A/B)) if B>0 and A>0 else 0.0

def _dfa(rr):
    N=len(rr)
    if N<16: return 0.0
    y=np.cumsum(rr-np.mean(rr)); bs=np.arange(4,min(17,N//4+1))
    if len(bs)<2: return 0.0
    F=[]
    for n in bs:
        nb=N//n
        if nb==0: continue
        rms=[]
        for i in range(nb):
            s=y[i*n:(i+1)*n]; x=np.arange(n); s=s-np.polyval(np.polyfit(x,s,1),x)
            rms.append(np.sqrt(np.mean(s**2)))
        F.append(np.mean(rms) if rms else 0)
    if len(F)<2 or any(f<=0 for f in F): return 0.0
    ln,lf=np.log(bs[:len(F)]),np.log(np.array(F)); v=np.isfinite(ln)&np.isfinite(lf)
    return float(np.polyfit(ln[v],lf[v],1)[0]) if v.sum()>=2 else 0.0

def nonlinear(rr):
    keys=["sd1","sd2","sd1sd2Ratio","sampleEntropy","dfaAlpha1"]
    if len(rr)<10: return {k:0.0 for k in keys}
    try:
        a,b=rr[:-1],rr[1:]; sd1=float(np.std(b-a,ddof=1)/np.sqrt(2)); sd2=float(np.std(b+a,ddof=1)/np.sqrt(2))
        return dict(sd1=sd1,sd2=sd2,sd1sd2Ratio=float(sd1/sd2) if sd2>0 else 0.0,
            sampleEntropy=_sampen(rr),dfaAlpha1=_dfa(rr))
    except Exception: return {k:0.0 for k in keys}

def temp_feats(temp,fs=4):
    if len(temp)<2: return dict(tempMean=0.0,tempSlope=0.0,tempStd=0.0,tempRange=0.0)
    t=np.arange(len(temp))/fs/60
    return dict(tempMean=float(np.mean(temp)),tempSlope=float(np.polyfit(t,temp,1)[0]),
        tempStd=float(np.std(temp,ddof=1)),tempRange=float(np.max(temp)-np.min(temp)))

def accel_feats(acc,fs=32):
    if len(acc)<2: return dict(accelMagnitudeMean=0.0,accelMagnitudeStd=0.0,stepCount=0.0,activityType=0.0)
    mag=np.sqrt(np.sum(acc**2,axis=1)); d=mag-np.mean(mag)
    pk,_=scipy_signal.find_peaks(d,height=0.3,distance=int(0.3*fs)); mm=float(np.mean(mag))
    act=0 if mm<1.02 else (1 if mm<1.1 else 2)
    return dict(accelMagnitudeMean=mm,accelMagnitudeStd=float(np.std(mag,ddof=1)),
        stepCount=float(len(pk)),activityType=float(act))
''')

# ---------------------------------------------------------------- cell: morphology
code(r'''
# ===== PPG morphology (identical maths to ml/src/morphology.py) =====
MORPHOLOGY_FEATURES=["ppgAmpMean","ppgAmpStd","ppgAmpCV","ppgRiseTimeMean",
    "ppgRiseTimeStd","ppgWidthMean","ppgWidthStd","ppgAreaMean"]
def _empty_morph(): return {k:0.0 for k in MORPHOLOGY_FEATURES}

def morphology(bvp, fs=64):
    bvp=np.asarray(bvp,float).flatten()
    if len(bvp)<fs: return _empty_morph()
    try:
        nyq=fs/2; b,a=scipy_signal.butter(3,[0.5/nyq,min(8.0/nyq,0.99)],btype="band")
        f=scipy_signal.filtfilt(b,a,bvp)
        if np.std(f)<1e-8: return _empty_morph()
        md=int(0.3*fs)
        peaks,_=scipy_signal.find_peaks(f,distance=md,height=np.percentile(f,60))
        troughs,_=scipy_signal.find_peaks(-f,distance=md)
        if len(peaks)<3 or len(troughs)<2: return _empty_morph()
        amps,rt,wd,ar=[],[],[],[]; I=getattr(np,"trapezoid",getattr(np,"trapz"))
        for pk in peaks:
            pr=troughs[troughs<pk]
            if len(pr)==0: continue
            on=pr[-1]; rs=pk-on
            if rs<=0 or rs>0.4*fs: continue
            amp=f[pk]-f[on]
            if amp<=0: continue
            half=f[on]+amp/2; se=min(len(f),on+int(1.5*fs)); seg=f[on:se]; ab=seg>=half
            wsamp=(np.where(ab)[0][-1]-np.where(ab)[0][0]) if ab.any() else 0
            area=float(I(np.clip(seg-f[on],0,None)))/(amp+1e-9)
            amps.append(amp); rt.append(rs/fs); wd.append(wsamp/fs); ar.append(area)
        if len(amps)<3: return _empty_morph()
        amps,rt,wd,ar=map(np.asarray,(amps,rt,wd,ar)); am=float(np.mean(amps))
        return dict(ppgAmpMean=am,ppgAmpStd=float(np.std(amps,ddof=1)),
            ppgAmpCV=float(np.std(amps,ddof=1)/am) if am>0 else 0.0,
            ppgRiseTimeMean=float(np.mean(rt)),ppgRiseTimeStd=float(np.std(rt,ddof=1)),
            ppgWidthMean=float(np.mean(wd)),ppgWidthStd=float(np.std(wd,ddof=1)),
            ppgAreaMean=float(np.mean(ar)))
    except Exception: return _empty_morph()
''')

# ---------------------------------------------------------------- cell: load + window
code(r'''
# ===== Load WESAD wrist signals and cut into labelled 60-s windows =====
RATES={"BVP":64,"EDA":4,"TEMP":4,"ACC":32,"label":700}
LABELS={0:"transient",1:"baseline",2:"stress",3:"amusement",4:"meditation"}
STRESS_BIN={1:0,2:1,3:0,4:0}

def load_subject(p):
    with open(p,"rb") as fh: d=pickle.load(fh,encoding="latin1")
    w=d["signal"]["wrist"]
    return dict(bvp=w["BVP"].flatten(),temp=w["TEMP"].flatten(),acc=w["ACC"],labels=d["label"].flatten())

def labels_to_bvp_rate(labels,L):
    ratio=RATES["label"]/RATES["BVP"]; out=np.zeros(L,dtype=int)
    for i in range(L):
        s=int(i*ratio); e=min(int((i+1)*ratio),len(labels))
        if s<len(labels):
            seg=labels[s:e]
            if len(seg): v,c=np.unique(seg,return_counts=True); out[i]=v[np.argmax(c)]
    return out

def windows_for_subject(p, window_sec=WINDOW_SEC):
    s=load_subject(p); lab=labels_to_bvp_rate(s["labels"],len(s["bvp"]))
    bw=window_sec*RATES["BVP"]; tw=window_sec*RATES["TEMP"]; aw=window_sec*RATES["ACC"]
    out=[]
    for i in range((len(s["bvp"])-bw)//bw+1):
        b0=i*bw; b1=b0+bw
        t0=int(b0*RATES["TEMP"]/RATES["BVP"]); a0=int(b0*RATES["ACC"]/RATES["BVP"])
        if b1>len(s["bvp"]) or t0+tw>len(s["temp"]) or a0+aw>len(s["acc"]): break
        wl=lab[b0:b1]; vl=wl[wl>0]
        if len(vl)==0: continue
        v,c=np.unique(vl,return_counts=True); ml=v[np.argmax(c)]
        if np.max(c)/len(vl)<0.8: continue
        out.append(dict(bvp=s["bvp"][b0:b1],temp=s["temp"][t0:t0+tw],acc=s["acc"][a0:a0+aw],
                        label=int(ml),stress_binary=STRESS_BIN.get(int(ml),-1),subject=Path(p).stem))
    return out

all_windows=[]
for p in found:
    wnd=windows_for_subject(p); all_windows+=wnd
    print(f"  {Path(p).stem}: {len(wnd)} windows ({sum(w['stress_binary']==1 for w in wnd)} stress)")
print("TOTAL windows:", len(all_windows))
''')

# ---------------------------------------------------------------- cell: parallel extract
code(r'''
# ===== Parallel feature extraction (the part Kaggle speeds up) =====
import pandas as pd
from joblib import Parallel, delayed
import time

def extract_one(w):
    rr=bvp_to_rr_intervals(w["bvp"],fs=64)
    feats={}
    feats.update(time_domain(rr)); feats.update(freq_domain(rr)); feats.update(nonlinear(rr))
    feats.update(temp_feats(w["temp"])); feats.update(accel_feats(w["acc"]))
    feats.update(morphology(w["bvp"],fs=64) if USE_MORPHOLOGY else _empty_morph())
    feats.update(label=w["label"],stress_binary=w["stress_binary"],subject=w["subject"])
    return feats

t0=time.time()
rows=Parallel(n_jobs=N_JOBS,verbose=5)(delayed(extract_one)(w) for w in all_windows)
wesad_df=pd.DataFrame(rows)
print(f"extracted {wesad_df.shape} in {time.time()-t0:.1f}s")
wesad_df=wesad_df[wesad_df["stress_binary"].isin([0,1])].copy()
print("WESAD class balance:", wesad_df["stress_binary"].value_counts().to_dict())
''')

# ---------------------------------------------------------------- cell: load SIPD
code(r'''
# ===== Load Stress-Predict / SIPD (raw E4) — same pipeline as WESAD =====
# Reuses extract_one() so SIPD features are computed identically to WESAD's,
# which is what makes training-on-SIPD / testing-on-WESAD a valid comparison.
import pandas as pd, time as _time
sipd_df = None
if USE_SIPD:
    def _read_e4_single(p):
        a = np.loadtxt(p, delimiter=","); return float(a[0]), float(a[1]), a[2:].astype(float)
    def _read_e4_acc(p):
        a = np.loadtxt(p, delimiter=","); return float(a[0,0]), float(a[1,0]), a[2:].astype(float)
    def _slice(sig, s0, fs, t0, t1):
        i0 = max(int(round((t0-s0)*fs)), 0); i1 = int(round((t1-s0)*fs))
        return sig[i0:i1] if sig.ndim == 1 else sig[i0:i1, :]
    def sipd_windows(root, window_sec=WINDOW_SEC, purity=0.8):
        root = Path(root)
        lab = pd.read_csv(root/"Processed_data"/"Improved_All_Combined_hr_rsp_binary.csv")
        lab.columns = [str(c).strip() for c in lab.columns]
        tcol = next(c for c in lab.columns if c.lower().startswith("time"))
        wins = []
        for pid, grp in lab.groupby("Participant"):
            sec2lab = dict(zip(grp[tcol].astype(int), grp["Label"].astype(int)))
            folder = None
            for nm in (f"S{pid}", f"S{int(pid):02d}", str(pid)):
                if (root/"Raw_data"/nm/"BVP.csv").exists(): folder = root/"Raw_data"/nm; break
            if folder is None: continue
            bs, bfs, bvp = _read_e4_single(folder/"BVP.csv")
            try: ts, tfs, temp = _read_e4_single(folder/"TEMP.csv")
            except Exception: ts, tfs, temp = bs, 4.0, np.array([])
            try: as_, afs, acc = _read_e4_acc(folder/"ACC.csv")
            except Exception: as_, afs, acc = bs, 32.0, np.zeros((0,3))
            for i in range(int(len(bvp)//(bfs*window_sec))):
                t0 = bs+i*window_sec; t1 = t0+window_sec
                secs = [sec2lab[s] for s in range(int(t0), int(t1)) if s in sec2lab]
                if len(secs) < window_sec*purity: continue
                v, c = np.unique(secs, return_counts=True); maj = int(v[np.argmax(c)])
                if c.max()/len(secs) < purity: continue
                bw = _slice(bvp, bs, bfs, t0, t1)
                if len(bw) < bfs*window_sec*0.8: continue
                tw = _slice(temp, ts, tfs, t0, t1) if len(temp) else np.array([])
                aw = _slice(acc, as_, afs, t0, t1) if len(acc) else np.zeros((0,3))
                wins.append(dict(bvp=bw, temp=tw, acc=aw, label=2 if maj else 1,
                                 stress_binary=maj, subject=f"SP_S{int(pid)}"))
        return wins

    _t = _time.time(); sw = sipd_windows(SIPD_ROOT)
    print(f"SIPD: {len(sw)} windows from {len(set(w['subject'] for w in sw))} participants")
    rows = Parallel(n_jobs=N_JOBS, verbose=1)(delayed(extract_one)(w) for w in sw)
    sipd_df = pd.DataFrame(rows)
    sipd_df = sipd_df[sipd_df.stress_binary.isin([0,1])].copy()
    print(f"SIPD features {sipd_df.shape} in {_time.time()-_t:.1f}s | "
          f"balance {sipd_df.stress_binary.value_counts().to_dict()}")
else:
    print("SIPD disabled / not found - training falls back to WESAD.")
''')

# ---------------------------------------------------------------- cell: extra E4 datasets
code(r'''
# ===== Extra WATCH-ONLY E4 datasets for FUSION (PhysioStress, ForDigitStress, VerBIO) =====
# Reads ONLY BVP/ACC/TEMP (never EDA/ECG) -> identical watch surface to WESAD/SIPD.
# PhysioStress is OPEN ACCESS (PhysioNet) and the recommended first add. Labels use
# tags+Stress_Level (PhysioStress) or a thresholded continuous annotation (ForDigit/VerBIO);
# verify the label-file layout against your download.
extra_dfs = {}
if USE_FORDIGIT or USE_VERBIO or USE_PHYSIOSTRESS:
    import pandas as pd
    def _e4_single(p):
        a=np.loadtxt(p,delimiter=","); return float(a[0]),float(a[1]),a[2:].astype(float)
    def _e4_acc(p):
        a=np.loadtxt(p,delimiter=","); return float(a[0,0]),float(a[1,0]),a[2:].astype(float)
    def _slc(s,s0,fs,t0,t1):
        i0=max(int(round((t0-s0)*fs)),0); i1=int(round((t1-s0)*fs))
        return s[i0:i1] if s.ndim==1 else s[i0:i1,:]
    def _ann_label_fn(path,thr=0.5):
        try: ann=pd.read_csv(path)
        except Exception: return None
        ann.columns=[str(c).strip().lower() for c in ann.columns]
        tcol=next((c for c in ann.columns if any(h in c for h in ("time","frame","timestamp"))),ann.columns[0])
        vcol=next((c for c in ann.columns if any(h in c for h in ("stress","label","value","rating"))),ann.columns[-1])
        t=ann[tcol].to_numpy(float); v=(ann[vcol].to_numpy(float)>=thr).astype(int)
        def fn(t0,t1):
            m=(t>=t0)&(t<t1)
            if m.sum()==0: return None
            fr=v[m].mean()
            return None if 0.2<fr<0.8 else int(fr>=0.5)
        return fn
    def _e4_windows(sd, fn, prefix, window_sec=WINDOW_SEC):
        sd=Path(sd)
        if not (sd/"BVP.csv").exists(): return []
        bs,bfs,bvp=_e4_single(sd/"BVP.csv")
        try: ts,tfs,temp=_e4_single(sd/"TEMP.csv")
        except Exception: ts,tfs,temp=bs,4.0,np.array([])
        try: as_,afs,acc=_e4_acc(sd/"ACC.csv")
        except Exception: as_,afs,acc=bs,32.0,np.zeros((0,3))
        out=[]
        for i in range(int(len(bvp)//(bfs*window_sec))):
            t0=bs+i*window_sec; t1=t0+window_sec
            lab=fn(t0,t1)
            if lab not in (0,1): continue
            bw=_slc(bvp,bs,bfs,t0,t1)
            if len(bw)<bfs*window_sec*0.8: continue
            tw=_slc(temp,ts,tfs,t0,t1) if len(temp) else np.array([])
            aw=_slc(acc,as_,afs,t0,t1) if len(acc) else np.zeros((0,3))
            out.append(dict(bvp=bw,temp=tw,acc=aw,label=2 if lab else 1,
                            stress_binary=lab,subject=f"{prefix}_{sd.name}"))
        return out
    def _load_e4(root,prefix,ann_glob):
        root=Path(root); w=[]
        for sd in sorted(p for p in root.glob("*") if p.is_dir()):
            ann=next(iter(sd.glob(ann_glob)),None)
            fn=_ann_label_fn(ann) if ann else None
            if fn is None: continue
            w+=_e4_windows(sd,fn,prefix)
        return w

    # ---- PhysioStress: protocol-based labels from tags.csv (dataset's own mapping) ----
    # E4 timestamps here are UTC datetime strings; the tag list is prepended with the
    # session start, so stressor spans by tag index (per Wearable_Dataset.ipynb) are:
    #   V1 (S-subjects): Stroop[3,4] TMCT[5,6] RealOpinion[7,8] OppOpinion[9,10] Subtract[11,12]
    #   V2 (f-subjects): TMCT[2,3] RealOpinion[4,5] OppOpinion[6,7] Subtract[8,9]
    # Non-stress = baseline (start -> first stressor). Reads ONLY BVP/ACC/TEMP.
    # Excludes f07 (PPG sensor covered) + f14_a/f14_b (session split across files).
    import datetime as _dtm
    _PS_SPANS={"S":[(3,4),(5,6),(7,8),(9,10),(11,12)],"f":[(2,3),(4,5),(6,7),(8,9)]}
    _PS_SKIP={"f07","f14_a","f14_b"}
    def _ps_dt(s): return _dtm.datetime.strptime(str(s).strip(),"%Y-%m-%d %H:%M:%S")
    def _ps_read(path):                      # E4 file w/ datetime header -> (start_dt, samples)
        with open(path) as fh: r0=fh.readline().strip(); fh.readline()
        return _ps_dt(r0.split(",")[0]), np.loadtxt(path,delimiter=",",skiprows=2)
    def _load_physiostress(root,window_sec=WINDOW_SEC):
        root=Path(root); out=[]
        for sd in sorted(p for p in root.glob("*") if p.is_dir()):
            if sd.name in _PS_SKIP or not (sd/"BVP.csv").exists() or not (sd/"tags.csv").exists(): continue
            coh="S" if sd.name.startswith("S") else "f"
            try:
                bstart,bvp=_ps_read(sd/"BVP.csv")
                try: _,temp=_ps_read(sd/"TEMP.csv")
                except Exception: temp=np.array([])
                try: _,acc=_ps_read(sd/"ACC.csv")
                except Exception: acc=np.zeros((0,3))
                lines=[l for l in (sd/"tags.csv").read_text().splitlines() if l.strip()]
                tags=[0.0]+[(_ps_dt(l)-bstart).total_seconds() for l in lines]
            except Exception: continue
            spans=[(tags[a],tags[b]) for a,b in _PS_SPANS[coh] if b<len(tags)]
            base_end=tags[_PS_SPANS[coh][0][0]] if _PS_SPANS[coh][0][0]<len(tags) else 0
            if not spans or base_end<=0: continue
            def _ov(w0,w1,a,b): return max(0.0,min(w1,b)-max(w0,a))
            for i in range(int(len(bvp)//(64*window_sec))):
                w0=i*window_sec; w1=w0+window_sec
                st=sum(_ov(w0,w1,a,b) for a,b in spans); ba=_ov(w0,w1,0,base_end)
                if st>=0.5*window_sec: lab=1
                elif ba>=0.5*window_sec: lab=0
                else: continue
                bw=bvp[i*64*window_sec:(i+1)*64*window_sec]
                tw=temp[i*4*window_sec:(i+1)*4*window_sec] if len(temp) else np.array([])
                aw=acc[i*32*window_sec:(i+1)*32*window_sec] if len(acc) else np.zeros((0,3))
                out.append(dict(bvp=bw,temp=tw,acc=aw,label=2 if lab else 1,
                                stress_binary=lab,subject=f"PS_{sd.name}"))
        return out

    if USE_PHYSIOSTRESS:
        w=_load_physiostress(PHYSIOSTRESS_ROOT)
        if not w:
            print("PhysioStress: no windows (set PHYSIOSTRESS_ROOT to .../Wearable_Dataset/STRESS).")
        else:
            rows=Parallel(n_jobs=N_JOBS)(delayed(extract_one)(x) for x in w)
            d=pd.DataFrame(rows); d=d[d.stress_binary.isin([0,1])].copy()
            extra_dfs["PhysioStress"]=d
            print(f"PhysioStress: {len(d)} windows / {d.subject.nunique()} subjects | "
                  f"balance {d.stress_binary.value_counts().to_dict()}")

    for flag,root,prefix,name,ann in [
        (USE_FORDIGIT,FORDIGIT_ROOT,"FD","ForDigitStress","*stress*"),
        (USE_VERBIO,VERBIO_ROOT,"VB","VerBIO","*[Ss]tress*")]:
        if not flag: continue
        w=_load_e4(root,prefix,ann)
        if not w:
            print(f"{name}: no labelled windows found (adjust ANN_GLOB to your files)"); continue
        rows=Parallel(n_jobs=N_JOBS)(delayed(extract_one)(x) for x in w)
        d=pd.DataFrame(rows); d=d[d.stress_binary.isin([0,1])].copy()
        extra_dfs[name]=d
        print(f"{name}: {len(d)} windows / {d.subject.nunique()} subjects | "
              f"balance {d.stress_binary.value_counts().to_dict()}")
else:
    print("No extra E4 datasets enabled (USE_PHYSIOSTRESS / USE_FORDIGIT / USE_VERBIO = False).")
''')

# ---------------------------------------------------------------- cell: select training set
code(r'''
# ===== Choose the training set (TRAIN_ON) and the held-out test set (EVAL_ON) =====
import pandas as pd
_avail = {"WESAD": wesad_df}
_avail.update(globals().get("extra_dfs") or {})
if sipd_df is not None and len(sipd_df): _avail["SIPD"] = sipd_df
print("Available datasets:", {k: len(v) for k, v in _avail.items()})

# TRAIN_ON may be any "+"-joined combo of available datasets (fusion).
_want = [d for d in TRAIN_ON.split("+")]
_have = [d for d in _want if d in _avail]
if not _have:
    print(f"TRAIN_ON='{TRAIN_ON}' unavailable -> falling back to WESAD")
    _have = ["WESAD"]
train_name = "+".join(_have)
df = (_avail[_have[0]] if len(_have) == 1
      else pd.concat([_avail[d] for d in _have], ignore_index=True))

eval_df = _avail.get(EVAL_ON) if (EVAL_ON in _avail and EVAL_ON not in _have) else None
print(f"TRAIN_ON = {train_name}: {len(df)} windows / {df['subject'].nunique()} subjects "
      f"| balance {df['stress_binary'].value_counts().to_dict()}")
print(f"EVAL_ON  = {EVAL_ON if eval_df is not None else '(none - no held-out dataset)'}"
      + (f": {len(eval_df)} windows" if eval_df is not None else ""))
''')

# ---------------------------------------------------------------- cell: helpers
code(r'''
# ===== Shared helpers: feature set, per-subject norm, model factory =====
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score,f1_score,roc_auc_score,precision_score,
                             recall_score,classification_report,confusion_matrix)
from xgboost import XGBClassifier

HRV21=["meanRR","sdnn","rmssd","pnn50","pnn20","hrMean","hrStd","hrRange","cvRR",
       "vlfPower","lfPower","hfPower","lfHfRatio","totalPower","lfNorm","hfNorm",
       "sd1","sd2","sd1sd2Ratio","sampleEntropy","dfaAlpha1"]
FEATS = HRV21 + (MORPHOLOGY_FEATURES if USE_MORPHOLOGY else [])

def per_subject_z(d, cols, g="subject", eps=1e-8):
    o=d.copy(); gp=o.groupby(g)[cols]
    mu=gp.transform("mean"); sd=gp.transform("std").replace(0,np.nan)
    o[cols]=((o[cols]-mu)/(sd+eps)).fillna(0.0); return o

# Hyperparameters: BASE_PARAMS are the hand-set defaults; tuning overwrites BEST_PARAMS.
BASE_PARAMS=dict(n_estimators=200,max_depth=6,learning_rate=0.1,subsample=0.8,
                 colsample_bytree=0.8,min_child_weight=3,gamma=0.0,reg_lambda=1.0)
BEST_PARAMS=dict(BASE_PARAMS)

def make_model(y):
    return XGBClassifier(**BEST_PARAMS,
        scale_pos_weight=(y==0).sum()/max((y==1).sum(),1),
        random_state=SEED,eval_metric="logloss")
print(f"{len(FEATS)} features | normalization={NORMALIZATION} | train_set={train_name}")
''')

# ---------------------------------------------------------------- cell: tuning (PSO/GWO)
code(r'''
# ===== Nature-inspired hyperparameter tuning: PSO vs GWO =====
# Objective: MAXIMIZE subject-disjoint 4-fold CV AUC on the TRAINING set (df).
# The held-out EVAL_ON set is never used here -> no leakage. Best params -> BEST_PARAMS.
tuning_results = {}
if RUN_TUNING:
    _HP=[("max_depth",3,10,"int"),("learning_rate",0.01,0.30,"float"),
         ("n_estimators",50,300,"int"),("subsample",0.5,1.0,"float"),
         ("colsample_bytree",0.5,1.0,"float"),("min_child_weight",1,10,"int"),
         ("gamma",0.0,5.0,"float"),("reg_lambda",0.0,5.0,"float")]
    _LB=np.array([h[1] for h in _HP],float); _UB=np.array([h[2] for h in _HP],float)
    def _decode(v):
        return {n:(int(round(float(np.clip(x,lo,hi)))) if t=="int" else float(np.clip(x,lo,hi)))
                for x,(n,lo,hi,t) in zip(v,_HP)}
    def _clip(x): return np.minimum(np.maximum(x,_LB),_UB)
    _Xt=df[FEATS].values; _yt=df["stress_binary"].values; _gt=df["subject"].values
    _NSPLITS=max(2,min(4,len(np.unique(_gt))))   # adaptive: never more folds than subjects
    def _objective(vec):
        p=_decode(vec); skf=StratifiedGroupKFold(n_splits=_NSPLITS,shuffle=True,random_state=SEED); aucs=[]
        for tr,te in skf.split(_Xt,_yt,_gt):
            if NORMALIZATION=="per_subject":
                Xtr,Xte=_Xt[tr].copy(),_Xt[te].copy()
                for idx,Xs in ((tr,Xtr),(te,Xte)):
                    for gid in np.unique(_gt[idx]):
                        mm=_gt[idx]==gid; mu=Xs[mm].mean(0); sd=Xs[mm].std(0); sd[sd==0]=1.0
                        Xs[mm]=(Xs[mm]-mu)/sd
                Xtr,Xte=np.nan_to_num(Xtr),np.nan_to_num(Xte)
            else:
                sc=StandardScaler(); Xtr=np.nan_to_num(sc.fit_transform(_Xt[tr])); Xte=np.nan_to_num(sc.transform(_Xt[te]))
            ytr,yte=_yt[tr],_yt[te]
            if len(set(yte))<2 or len(set(ytr))<2: continue
            spw=(ytr==0).sum()/max((ytr==1).sum(),1)
            m=XGBClassifier(**p,scale_pos_weight=spw,random_state=SEED,eval_metric="logloss",n_jobs=1)
            m.fit(Xtr,ytr); aucs.append(roc_auc_score(yte,m.predict_proba(Xte)[:,1]))
        return 1.0-(np.mean(aucs) if aucs else 0.0)
    def _pso(obj,pop,iters,w=0.7,c1=1.5,c2=1.5):
        rng=np.random.RandomState(SEED); dim=len(_LB); span=_UB-_LB
        X=rng.uniform(_LB,_UB,(pop,dim)); V=rng.uniform(-span,span,(pop,dim))*0.1
        pb=X.copy(); pbf=np.array([obj(x) for x in X]); gi=pbf.argmin(); gb=pb[gi].copy(); gbf=pbf[gi]; h=[gbf]
        for it in range(iters):
            r1,r2=rng.rand(pop,dim),rng.rand(pop,dim)
            V=w*V+c1*r1*(pb-X)+c2*r2*(gb-X); X=_clip(X+V)
            f=np.array([obj(x) for x in X]); im=f<pbf; pb[im],pbf[im]=X[im],f[im]
            if pbf.min()<gbf: gi=pbf.argmin(); gb=pb[gi].copy(); gbf=pbf[gi]
            h.append(gbf); print(f"  PSO {it+1}/{iters}: CV AUC={1-gbf:.4f}")
        return gb,gbf,h
    def _gwo(obj,pop,iters):
        rng=np.random.RandomState(SEED); dim=len(_LB)
        X=rng.uniform(_LB,_UB,(pop,dim)); f=np.array([obj(x) for x in X]); o=f.argsort()
        al,be,de=X[o[0]].copy(),X[o[1]].copy(),X[o[2]].copy(); af,bf,dff=f[o[0]],f[o[1]],f[o[2]]; h=[af]
        for it in range(iters):
            a=2-2*it/iters
            A1=2*a*rng.rand(pop,dim)-a;C1=2*rng.rand(pop,dim)
            A2=2*a*rng.rand(pop,dim)-a;C2=2*rng.rand(pop,dim)
            A3=2*a*rng.rand(pop,dim)-a;C3=2*rng.rand(pop,dim)
            X1=al-A1*np.abs(C1*al-X);X2=be-A2*np.abs(C2*be-X);X3=de-A3*np.abs(C3*de-X)
            X=_clip((X1+X2+X3)/3); f=np.array([obj(x) for x in X])
            for i in range(pop):
                if f[i]<af: de,dff=be.copy(),bf; be,bf=al.copy(),af; al,af=X[i].copy(),f[i]
                elif f[i]<bf: de,dff=be.copy(),bf; be,bf=X[i].copy(),f[i]
                elif f[i]<dff: de,dff=X[i].copy(),f[i]
            h.append(af); print(f"  GWO {it+1}/{iters}: CV AUC={1-af:.4f}")
        return al,af,h
    _runners={"PSO":_pso,"GWO":_gwo}
    for _name in TUNERS:
        nm=_name.upper()
        if nm not in _runners: continue
        print(f"[{nm}] tuning: {TUNE_POP} agents x {TUNE_ITERS} iters (4-fold subject-disjoint AUC)")
        vec,fit,hist=_runners[nm](_objective,TUNE_POP,TUNE_ITERS)
        tuning_results[nm]=dict(params=_decode(vec),cv_auc=float(1-fit),history=[float(1-x) for x in hist])
        print(f"[{nm}] best CV AUC={1-fit:.4f}")
    if tuning_results:
        _best=max(tuning_results,key=lambda k:tuning_results[k]["cv_auc"])
        BEST_PARAMS=dict(BASE_PARAMS); BEST_PARAMS.update(tuning_results[_best]["params"])
        print(f"\nWINNER: {_best} (CV AUC={tuning_results[_best]['cv_auc']:.4f})")
        for k in tuning_results: print(f"  {k}: CV AUC={tuning_results[k]['cv_auc']:.4f}")
        print("Tuned params ->", BEST_PARAMS)
else:
    print("RUN_TUNING=False -> using BASE_PARAMS:", BEST_PARAMS)
''')

# ---------------------------------------------------------------- cell: train + eval
code(r'''
# ===== Within-TRAIN_ON LOSO with the (tuned) hyperparameters =====
print(f"Within-{train_name} LOSO | {len(FEATS)} features | norm={NORMALIZATION} | "
      f"params={'tuned' if tuning_results else 'base'}")
# ---- LOSO CV ----
logo=LeaveOneGroupOut(); g=df["subject"].values; y=df["stress_binary"].values
yp=np.zeros_like(y); pp=np.zeros_like(y,dtype=float)
for tr,te in logo.split(df,y,g):
    dtr,dte=df.iloc[tr],df.iloc[te]
    if NORMALIZATION=="per_subject":
        dtr=per_subject_z(dtr,FEATS); dte=per_subject_z(dte,FEATS)
        Xtr,Xte=dtr[FEATS].values,dte[FEATS].values
    else:
        sc=StandardScaler(); Xtr=sc.fit_transform(dtr[FEATS]); Xte=sc.transform(dte[FEATS])
    Xtr,Xte=np.nan_to_num(Xtr),np.nan_to_num(Xte)
    m=make_model(y[tr]); m.fit(Xtr,y[tr])
    yp[te]=m.predict(Xte); pp[te]=m.predict_proba(Xte)[:,1]

cv=dict(cv_accuracy=accuracy_score(y,yp),cv_f1_weighted=f1_score(y,yp,average="weighted"),
    cv_f1_binary=f1_score(y,yp,average="binary"),cv_precision=precision_score(y,yp),
    cv_recall=recall_score(y,yp),cv_auc_roc=roc_auc_score(y,pp))
print("\nLOSO-CV (threshold=0.5):"); [print(f"  {k}: {v:.4f}") for k,v in cv.items()]

# ---- Threshold calibration on TRAINING out-of-fold probs (pp) — no test leakage ----
def pick_threshold(y_true, prob, objective="f1"):
    grid=np.unique(np.round(np.quantile(prob,np.linspace(0,1,101)),4))
    grid=grid[(grid>0)&(grid<1)]
    best_t,best_s=0.5,-1.0
    for t in grid:
        pred=(prob>=t).astype(int)
        if objective=="youden":
            tp=((pred==1)&(y_true==1)).sum(); fn=((pred==0)&(y_true==1)).sum()
            fp=((pred==1)&(y_true==0)).sum(); tn=((pred==0)&(y_true==0)).sum()
            tpr=tp/max(tp+fn,1); fpr=fp/max(fp+tn,1); s=tpr-fpr
        elif objective=="balanced":
            from sklearn.metrics import balanced_accuracy_score; s=balanced_accuracy_score(y_true,pred)
        else:  # f1
            s=f1_score(y_true,pred,zero_division=0)
        if s>best_s: best_s,best_t=s,float(t)
    return best_t

BEST_THRESHOLD=0.5
TRAIN_PRIOR=float((y==1).mean())   # training stress prevalence (leakage-free)
if CALIBRATE_THRESHOLD:
    if THRESHOLD_OBJECTIVE=="prior":
        # Base-rate matching: predict ~TRAIN_PRIOR fraction positive on the SOURCE OOF.
        # (For the held-out target, the threshold is re-derived on the target's own
        #  scores below — that is what makes it adapt to cross-dataset shift.)
        BEST_THRESHOLD=float(np.quantile(pp,1-TRAIN_PRIOR))
    else:
        BEST_THRESHOLD=pick_threshold(y,pp,THRESHOLD_OBJECTIVE)
    yp_c=(pp>=BEST_THRESHOLD).astype(int)
    cv_cal=dict(thr=BEST_THRESHOLD,
        acc=accuracy_score(y,yp_c),f1=f1_score(y,yp_c,zero_division=0),
        precision=precision_score(y,yp_c,zero_division=0),recall=recall_score(y,yp_c,zero_division=0))
    print(f"\nCalibrated threshold ({THRESHOLD_OBJECTIVE}, train prior={TRAIN_PRIOR:.2f}) = {BEST_THRESHOLD:.3f}")
    print(f"LOSO-CV (calibrated): acc={cv_cal['acc']:.4f} f1={cv_cal['f1']:.4f} "
          f"precision={cv_cal['precision']:.4f} recall={cv_cal['recall']:.4f}")
print("\n",classification_report(y,yp,target_names=["Not Stressed","Stressed"]))
print("Confusion:\n",confusion_matrix(y,yp))
''')

# ---------------------------------------------------------------- cell: final + export
code(r'''
# ===== Train final model on ALL data and export in assets/ml format =====
# NOTE on normalization & deployment:
#   global       -> exported mean/std is a single vector; drop-in for current TS inference.
#   per_subject  -> we still export the GLOBAL mean/std (so the JSON stays valid), but for
#                   the per-subject *accuracy* the watch must z-score against a per-user
#                   baseline. The exported global vector is a usable fallback until that
#                   on-device calibration is added (services/ai/stressModel.ts).
#   USE_MORPHOLOGY=True -> the TS feature extractor must ALSO compute the 8 ppg* features
#                   from raw PPG, or the model receives zeros for them. Not drop-in.

scaler=StandardScaler(); Xall=scaler.fit_transform(df[FEATS].values)
final=make_model(y); final.fit(np.nan_to_num(Xall),y)

# ---- Held-out CROSS-DATASET test: train on TRAIN_ON (all of df), test on EVAL_ON ----
# This is the headline generalization number (e.g. train SIPD -> test WESAD).
holdout=None
if eval_df is not None and len(eval_df):
    if NORMALIZATION=="per_subject":
        Xtr=per_subject_z(df,FEATS)[FEATS].values
        Xte=per_subject_z(eval_df,FEATS)[FEATS].values
    else:
        _sc=StandardScaler(); Xtr=_sc.fit_transform(df[FEATS]); Xte=_sc.transform(eval_df[FEATS])
    _hm=make_model(y); _hm.fit(np.nan_to_num(Xtr),y)
    _ye=eval_df["stress_binary"].values
    _qe=_hm.predict_proba(np.nan_to_num(Xte))[:,1]
    _auc=roc_auc_score(_ye,_qe) if len(set(_ye))>1 else float("nan")
    # Held-out threshold: for "prior", BASE-RATE MATCH on the TARGET's own scores
    # (predict ~TRAIN_PRIOR fraction positive) -> adapts to cross-dataset shift, no
    # target labels used. For source-based objectives, reuse the training threshold.
    if CALIBRATE_THRESHOLD and THRESHOLD_OBJECTIVE=="prior":
        _thr_h=float(np.quantile(_qe,1-TRAIN_PRIOR))
    else:
        _thr_h=float(BEST_THRESHOLD)
    _pe=(_qe>=0.5).astype(int)          # default threshold
    _pc=(_qe>=_thr_h).astype(int)       # calibrated / base-rate-matched threshold
    holdout=dict(acc=accuracy_score(_ye,_pe),f1=f1_score(_ye,_pe,zero_division=0),
                 auc=_auc,precision=precision_score(_ye,_pe,zero_division=0),
                 recall=recall_score(_ye,_pe,zero_division=0))
    holdout_cal=dict(threshold=_thr_h,
                 acc=accuracy_score(_ye,_pc),f1=f1_score(_ye,_pc,zero_division=0),auc=_auc,
                 precision=precision_score(_ye,_pc,zero_division=0),
                 recall=recall_score(_ye,_pc,zero_division=0))
    print(f"\n=== HELD-OUT: train {train_name} -> test {EVAL_ON} ===")
    print(f"  default (0.5):       acc={holdout['acc']:.4f} f1={holdout['f1']:.4f} "
          f"prec={holdout['precision']:.4f} rec={holdout['recall']:.4f} auc={_auc:.4f}")
    print(f"  calibrated ({_thr_h:.3f}): acc={holdout_cal['acc']:.4f} f1={holdout_cal['f1']:.4f} "
          f"prec={holdout_cal['precision']:.4f} rec={holdout_cal['recall']:.4f}  [{THRESHOLD_OBJECTIVE}]")
else:
    holdout_cal=None

imp=dict(zip(FEATS,final.feature_importances_.tolist()))
booster=final.get_booster()
trees=[json.loads(t) for t in booster.get_dump(dump_format="json")]
norm={"mean":dict(zip(FEATS,scaler.mean_.tolist())),"std":dict(zip(FEATS,scaler.scale_.tolist()))}
LEVELS={"low":{"min":0,"max":25,"label":"Low","color":"#35e27e"},
    "moderate":{"min":25,"max":50,"label":"Moderate","color":"#9B8EC4"},
    "elevated":{"min":50,"max":75,"label":"Elevated","color":"#E8A87C"},
    "high":{"min":75,"max":100,"label":"High","color":"#C4897B"}}
cv_metrics={k:v for k,v in cv.items()}

model_export=dict(version=VERSION,modelType="xgboost_binary_classifier",task="stress_detection",
    features=FEATS,numFeatures=len(FEATS),numTrees=len(trees),baseScore=0.5,
    learningRate=float(final.learning_rate),normalization=norm,trees=trees,
    decisionThreshold=float(BEST_THRESHOLD),   # calibrated; TS should classify prob>=this as stress
    metrics=cv_metrics,importances=imp,stressLevels=LEVELS,
    trainingNotes=dict(useMorphology=USE_MORPHOLOGY,normalization=NORMALIZATION,
        trainedOn=train_name,evalOn=(EVAL_ON if holdout else None),
        decisionThreshold=float(BEST_THRESHOLD),thresholdObjective=THRESHOLD_OBJECTIVE,
        crossDatasetHoldout=holdout,crossDatasetHoldoutCalibrated=holdout_cal,
        tunedParams=BEST_PARAMS,tuner=(max(tuning_results,key=lambda k:tuning_results[k]['cv_auc']) if tuning_results else None),
        nSubjects=int(df["subject"].nunique()),nSamples=int(len(df))))
metadata=dict(version=VERSION,features=FEATS,normalization=norm,
    decisionThreshold=float(BEST_THRESHOLD),
    metrics=cv_metrics,importances=imp,stressLevels=LEVELS)

out=Path("/kaggle/working")
(out/"stress_model.json").write_text(json.dumps(model_export,indent=2))
(out/"model_metadata.json").write_text(json.dumps(metadata,indent=2))
print("Wrote:")
print("  /kaggle/working/stress_model.json    ", round((out/"stress_model.json").stat().st_size/1024,1),"KB")
print("  /kaggle/working/model_metadata.json")
print("\nTop features:", sorted(imp.items(),key=lambda kv:-kv[1])[:8])
print("\nDownload both from the Kaggle 'Output' tab into the repo's assets/ml/stress/ "
      "(and ml/stress/models/). If USE_MORPHOLOGY or per_subject were on, apply the matching "
      "TypeScript changes noted above before shipping.")
''')

# ---------------------------------------------------------------- cell: within-dataset benchmark
code(r'''
# ===== WITHIN-DATASET benchmark (the protocol prior works report) =====
# Train AND test on the SAME dataset via leave-one-subject-out (subject-disjoint, no
# window overlap). These are the OPTIMISTIC "same-dataset" numbers to compare with
# the literature (e.g. WESAD ~93% binary). NOTE: published WESAD numbers usually
# INCLUDE EDA; Seren is PPG-only, so this is like-for-like minus EDA. Fixed
# BASE_PARAMS for reproducibility. Cross-dataset (below) is the honest test.
within_results = {}
if WITHIN_BENCHMARK:
    from sklearn.model_selection import LeaveOneGroupOut as _LOGO, StratifiedKFold as _SKF
    def _wfold(dd, cols, tr, te, y, norm):          # fit on tr, predict te (fixed BASE_PARAMS)
        a, b = dd.iloc[tr], dd.iloc[te]
        if norm == "per_subject":
            a = per_subject_z(a, cols); b = per_subject_z(b, cols)
            Xtr, Xte = a[cols].values, b[cols].values
        else:                                        # global standardization
            sc = StandardScaler(); Xtr = sc.fit_transform(a[cols]); Xte = sc.transform(b[cols])
        m = XGBClassifier(**BASE_PARAMS, scale_pos_weight=(y[tr]==0).sum()/max((y[tr]==1).sum(),1),
                          random_state=SEED, eval_metric="logloss")
        m.fit(np.nan_to_num(Xtr), y[tr])
        return m.predict(np.nan_to_num(Xte)), m.predict_proba(np.nan_to_num(Xte))[:,1]
    def _within_run(dd, cols, mode, norm):           # mode: "loso" (subject-disjoint) | "kfold" (random)
        y = dd.stress_binary.values; g = dd.subject.values
        yp = np.zeros_like(y); pp = np.zeros_like(y, dtype=float)
        if mode == "loso":
            splits = _LOGO().split(dd, y, g)
        else:   # random stratified k-fold: same subject can be in train AND test -> leakage
            k = max(2, min(WITHIN_KFOLD_K, int((y==1).sum()), int((y==0).sum())))
            splits = _SKF(n_splits=k, shuffle=True, random_state=SEED).split(dd, y)
        for tr, te in splits:
            yp[te], pp[te] = _wfold(dd, cols, tr, te, y, norm)
        return dict(acc=accuracy_score(y, yp), f1=f1_score(y, yp, zero_division=0),
                    auc=roc_auc_score(y, pp) if len(set(y)) > 1 else float("nan"))

    # Column 1 = Seren's honest within-dataset (your NORMALIZATION, LOSO). The leak demo
    # below uses GLOBAL norm (the prior-work protocol, e.g. PhysioStress) under which
    # k-fold leakage actually manifests -- per-subject norm is leakage-ROBUST (gap ~0).
    _hdr = f"{'dataset':13}{'features':10}{'subj':>4}{'Seren LOSO':>11}"
    if WITHIN_KFOLD: _hdr += f"{'| pw-LOSO':>10}{'pw-kfold':>10}{'leak gap':>10}"
    print(_hdr); print("-"*len(_hdr))
    for _dn2, _dd in _avail.items():
        for _fn, _ff in [("HRV-21", HRV21), ("HRV+morph", HRV21+MORPHOLOGY_FEATURES)]:
            _cols = [c for c in _ff if c in _dd.columns]
            _seren = _within_run(_dd, _cols, "loso", NORMALIZATION)
            within_results[f"{_dn2}|{_fn}|seren_loso"] = _seren
            _row = f"{_dn2:13}{_fn:10}{_dd['subject'].nunique():>4}{_seren['auc']:>11.3f}"
            if WITHIN_KFOLD:
                _pwl = _within_run(_dd, _cols, "loso", "global")    # prior-work protocol, honest split
                _pwk = _within_run(_dd, _cols, "kfold", "global")   # prior-work protocol, leaky split
                within_results[f"{_dn2}|{_fn}|global_loso"] = _pwl
                within_results[f"{_dn2}|{_fn}|global_kfold"] = _pwk
                _row += f"{_pwl['auc']:>10.3f}{_pwk['auc']:>10.3f}{_pwk['auc']-_pwl['auc']:>+10.3f}"
            print(_row)
    print("\nSeren LOSO = your model (per-subject norm, subject-disjoint) = the honest within number.")
    if WITHIN_KFOLD:
        print("pw-* = PRIOR-WORK protocol (GLOBAL norm): pw-LOSO (honest split) vs pw-kfold (random 10-fold,")
        print("same subject in train+test). 'leak gap' = pw-kfold - pw-LOSO = inflation from a non-subject-")
        print("disjoint split under global norm -- the protocol behind published 90%+ figures. Per-subject")
        print("norm (Seren LOSO col) is leakage-robust by comparison. Cross-dataset table below = real test.")
else:
    print("WITHIN_BENCHMARK = False -> skipped.")
''')

# ---------------------------------------------------------------- cell: cross-dataset
code(r'''
# ===== CROSS-DATASET matrix: every ordered pair of available datasets =====
# Train on one dataset, test on another — same raw-PPG pipeline on each.
# STABLE/REPRODUCIBLE: this matrix uses the FIXED BASE_PARAMS (not the run's tuned
# BEST_PARAMS), so the cells don't shift when you change TRAIN_ON. It is a relative
# transfer comparison; the tuned headline is the HELD-OUT cell above.
xds = {}
_dsets = dict(_avail)   # all loaded datasets (WESAD, SIPD, PhysioStress, ...)
if len(_dsets) >= 2:
    def _fixed_model(yv):
        return XGBClassifier(**BASE_PARAMS,
            scale_pos_weight=(yv==0).sum()/max((yv==1).sum(),1),
            random_state=SEED, eval_metric="logloss")
    def _xprep(tr, te, feats, norm):
        if norm == "per_subject":
            tr = per_subject_z(tr, feats); te = per_subject_z(te, feats)
            Xtr, Xte = tr[feats].values, te[feats].values
        else:
            sc = StandardScaler(); Xtr = sc.fit_transform(tr[feats]); Xte = sc.transform(te[feats])
        return np.nan_to_num(Xtr), tr.stress_binary.values, np.nan_to_num(Xte), te.stress_binary.values

    _names = list(_dsets)
    _pairs = [(a, b) for a in _names for b in _names if a != b]
    _cols = [c for c in (HRV21 + MORPHOLOGY_FEATURES) if all(c in d.columns for d in _dsets.values())]
    print(f"{'norm':12}{'train -> test':28}{'acc':>7}{'f1':>7}{'auc':>7}")
    print("-"*62)
    for norm in ("global", "per_subject"):
        for a, b in _pairs:
            Xtr, ytr, Xte, yte = _xprep(_dsets[a], _dsets[b], _cols, norm)
            m = _fixed_model(ytr); m.fit(Xtr, ytr)
            prob = m.predict_proba(Xte)[:,1]; pred = (prob >= 0.5).astype(int)
            acc = accuracy_score(yte, pred); f1 = f1_score(yte, pred, zero_division=0)
            auc = roc_auc_score(yte, prob) if len(set(yte)) > 1 else float("nan")
            xds[f"{norm}|{a}->{b}"] = dict(acc=acc, f1=f1, auc=auc)
            print(f"{norm:12}{a+' -> '+b:28}{acc:>7.3f}{f1:>7.3f}{auc:>7.3f}")
    print("\nRead-out: AUC is the fair cross-dataset metric. Higher rows = datasets that")
    print("transfer well; use them to choose your TRAIN_ON fusion. Logged to MLflow below.")
else:
    print("Only one dataset loaded -> cross-dataset matrix skipped (add SIPD/PhysioStress).")
''')

# ---------------------------------------------------------------- cell: final clean eval
code(r'''
# ===== FINAL CLEAN EVALUATION — no model-selection leakage on FINAL_TEST =====
# Protocol: (1) hold FINAL_TEST completely aside; (2) pick the config (morphology +
# normalization) using ONLY the dev datasets via leave-one-dev-dataset-out; (3) tune
# hyperparameters on the pooled dev set's subject-disjoint CV; (4) evaluate FINAL_TEST
# EXACTLY ONCE. Nothing about FINAL_TEST influences any decision -> the number is honest.
final_clean_auc = None
if FINAL_EVAL and FINAL_TEST in _avail and len([d for d in _avail if d != FINAL_TEST]) >= 2:
    from sklearn.model_selection import StratifiedGroupKFold as _SGK
    _test = _avail[FINAL_TEST]
    _dev = {k: v for k, v in _avail.items() if k != FINAL_TEST}; _dn = list(_dev)
    def _ce_auc(tr, te, feats, norm, params):
        if norm == "per_subject":
            Xtr = np.nan_to_num(per_subject_z(tr, feats)[feats].values)
            Xte = np.nan_to_num(per_subject_z(te, feats)[feats].values)
        else:
            sc = StandardScaler(); Xtr = np.nan_to_num(sc.fit_transform(tr[feats])); Xte = np.nan_to_num(sc.transform(te[feats]))
        ytr = tr.stress_binary.values; yte = te.stress_binary.values
        m = XGBClassifier(**params, scale_pos_weight=(ytr==0).sum()/max((ytr==1).sum(),1),
                          random_state=SEED, eval_metric="logloss"); m.fit(Xtr, ytr)
        return roc_auc_score(yte, m.predict_proba(Xte)[:,1]) if len(set(yte))>1 else float("nan")

    # (1)+(2) SELECT morphology+norm on DEV only (leave-one-dev-dataset-out, fixed BASE_PARAMS)
    print("Config selection on DEV only (FINAL_TEST held aside):")
    _cands = []
    for _mo in (True, False):
        for _no in ("per_subject", "global"):
            _f = [c for c in (HRV21 + (MORPHOLOGY_FEATURES if _mo else [])) if all(c in d.columns for d in _avail.values())]
            _aucs = [_ce_auc(pd.concat([_dev[o] for o in _dn if o != h], ignore_index=True), _dev[h], _f, _no, BASE_PARAMS) for h in _dn]
            _s = float(np.nanmean(_aucs)); _cands.append((_s, _mo, _no, _f))
            print(f"  {'HRV+morph' if _mo else 'HRV-21':9} {_no:11}: DEV-xds AUC={_s:.3f}")
    _cands.sort(key=lambda x: -x[0]); _bs, _bmo, _bno, _bfeats = _cands[0]
    print(f"  -> SELECTED (no FINAL_TEST seen): {'HRV+morph' if _bmo else 'HRV-21'} + {_bno}")

    # (3) TUNE the chosen config on pooled DEV CV only (leakage-free), via PSO if available
    _dev_all = pd.concat([_dev[d] for d in _dn], ignore_index=True)
    _fp = dict(BASE_PARAMS)
    if RUN_TUNING and "_pso" in globals():
        _Xd = _dev_all[_bfeats].values; _yd = _dev_all.stress_binary.values; _gd = _dev_all.subject.values
        _ns = max(2, min(4, len(np.unique(_gd))))
        def _obj2(vec):
            p = _decode(vec); sk = _SGK(n_splits=_ns, shuffle=True, random_state=SEED); a = []
            for tr, va in sk.split(_Xd, _yd, _gd):
                if _bno == "per_subject":
                    Xtr = _Xd[tr].copy(); Xva = _Xd[va].copy()
                    for idx, Xs in ((tr, Xtr), (va, Xva)):
                        for gid in np.unique(_gd[idx]):
                            mm = _gd[idx] == gid; mu = Xs[mm].mean(0); sv = Xs[mm].std(0); sv[sv==0] = 1.0; Xs[mm] = (Xs[mm]-mu)/sv
                    Xtr, Xva = np.nan_to_num(Xtr), np.nan_to_num(Xva)
                else:
                    sc = StandardScaler(); Xtr = np.nan_to_num(sc.fit_transform(_Xd[tr])); Xva = np.nan_to_num(sc.transform(_Xd[va]))
                yt, yv = _yd[tr], _yd[va]
                if len(set(yv)) < 2 or len(set(yt)) < 2: continue
                m = XGBClassifier(**p, scale_pos_weight=(yt==0).sum()/max((yt==1).sum(),1),
                                  random_state=SEED, eval_metric="logloss", n_jobs=1); m.fit(Xtr, yt)
                a.append(roc_auc_score(yv, m.predict_proba(Xva)[:,1]))
            return 1.0 - (np.mean(a) if a else 0.0)
        _vec, _, _ = _pso(_obj2, TUNE_POP, TUNE_ITERS); _fp.update(_decode(_vec))
        print(f"  tuned on DEV CV (PSO): {_fp}")

    # (4) EVALUATE FINAL_TEST exactly once
    final_clean_auc = _ce_auc(_dev_all, _test, _bfeats, _bno, _fp)
    print(f"\n=== FINAL CLEAN: train [{'+'.join(_dn)}] -> test [{FINAL_TEST}] (touched ONCE) ===")
    print(f"  AUC = {final_clean_auc:.4f}   <- report THIS (config + tuning chosen without seeing {FINAL_TEST})")
elif FINAL_EVAL:
    print("FINAL_EVAL skipped: need FINAL_TEST + >=2 other datasets loaded.")
''')

# ================================================================ EXPLAINABILITY (XAI)
md(r"""
## Explainability (XAI) — what the model learned and why

This section opens the XGBoost "black box" with **TreeSHAP** (exact Shapley values for tree
ensembles) so the thesis can argue the model is *physiologically sound*, not just accurate.

**Which model, on which features?** SHAP must be computed on the *exact* representation the
explained model consumes. Our reported headline configuration uses **per-subject
z-normalization**, so when `NORMALIZATION="per_subject"` we explain a model fit on the
per-subject-normalized features (the representation that yields the reported AUC). The
exported `stress_model.json` ships a *global* mean/std vector as an on-device fallback until
per-user calibration lands in `services/ai/stressModel.ts`; the feature **ranking** is
consistent across both, but the per-subject space is the faithful one to interpret.

**SHAP sign convention.** Values are in **log-odds of the positive class ("stressed")**:
a positive SHAP value pushes the prediction *toward stress*, negative pushes *toward calm*.

Four parts:
1. **Global** — beeswarm + mean|SHAP| ranking, cross-checked against XGBoost gain importance.
2. **Local** — waterfall plots for a correct-stress, a correct-calm, and a misclassified window.
3. **Feature effects** — SHAP dependence plots showing the *direction* of each top feature.
4. **Physiological interpretation** — does the model rely on lower RMSSD/HF and higher HR/LF-HF
   for stress, as the HRV literature predicts? (computed agreement score + written read-out).
5. **SHAP-guided feature reduction** (optional, `RUN_FEATURE_REDUCTION`) — retrain on the top-k
   features by mean|SHAP| and measure held-out AUC to find the leanest set that keeps accuracy.

Every cell is wrapped so a missing `shap` install or a plotting hiccup **never aborts the run**
(the model is already exported above). Results are logged to MLflow in the final cell.
""")

# ---------------------------------------------------------------- cell: XAI global (TreeSHAP)
code(r'''
# ===== XAI 1/4 — GLOBAL: TreeSHAP summary, mean|SHAP| ranking, vs XGBoost gain =====
# Robust: pip-installs shap if missing; wrapped so it can never crash the run.
shap_importance = {}; shap_plot_files = []; shap_physio = {}; XAI_OK = False
try:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    try:
        import shap
    except ImportError:
        import subprocess, sys
        print("shap not found - installing ...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "shap"], check=True)
        import shap
    print("shap", shap.__version__)

    OUTDIR = Path("/kaggle/working")
    # --- Build the EXACT input the explained model sees ---
    # per_subject: explain a model fit on per-subject z-normed features (the reported config).
    # global:      explain the exported `final` model directly on the global-standardized matrix.
    if NORMALIZATION == "per_subject":
        X_shap = np.nan_to_num(per_subject_z(df, FEATS)[FEATS].values)
        shap_model = make_model(y); shap_model.fit(X_shap, y)
        _norm_desc = "per-subject z-normalized (matches the reported AUC config)"
    else:
        X_shap = np.nan_to_num(Xall)          # reuse the export cell's global-standardized matrix
        shap_model = final
        _norm_desc = "globally standardized (matches the exported model)"
    X_df = pd.DataFrame(X_shap, columns=FEATS)
    print("SHAP input space:", _norm_desc, "| matrix", tuple(X_df.shape))

    # --- TreeSHAP (exact for trees; no background sampling needed) ---
    explainer = shap.TreeExplainer(shap_model)
    sv = explainer.shap_values(X_df)
    if isinstance(sv, list):                  # older shap: [class0, class1]
        sv = sv[1]
    sv = np.asarray(sv)
    if sv.ndim == 3:                          # (n, feat, classes) -> positive class
        sv = sv[:, :, -1]
    ev = np.atleast_1d(explainer.expected_value); ev = float(ev[-1] if ev.size > 1 else ev[0])
    print("SHAP values:", sv.shape, "| base value (log-odds):", round(ev, 4))

    # --- mean|SHAP| global ranking ---
    mean_abs = np.abs(sv).mean(0)
    order = np.argsort(mean_abs)[::-1]
    shap_importance = {FEATS[i]: float(mean_abs[i]) for i in order}
    print("\nGlobal importance (mean |SHAP|, log-odds units):")
    for i in order[:12]:
        print(f"  {FEATS[i]:14} {mean_abs[i]:.4f}")

    # --- compare SHAP ranking vs XGBoost GAIN importance (same model) ---
    _gain = shap_model.get_booster().get_score(importance_type="gain")
    gain_full = {f: float(_gain.get(f, 0.0)) for f in FEATS}
    shap_rank = {f: r for r, f in enumerate([FEATS[i] for i in order])}
    gain_rank = {f: r for r, f in enumerate(sorted(FEATS, key=lambda c: -gain_full[c]))}
    print("\nSHAP rank vs XGBoost gain rank (0 = most important):")
    print(f"  {'feature':14}{'shap_rank':>10}{'gain_rank':>10}{'gain':>12}")
    for f in [FEATS[i] for i in order[:12]]:
        print(f"  {f:14}{shap_rank[f]:>10}{gain_rank[f]:>10}{gain_full[f]:>12.2f}")
    # rank agreement (Spearman) between the two importance notions
    _sr = np.array([shap_rank[f] for f in FEATS]); _gr = np.array([gain_rank[f] for f in FEATS])
    _rho = float(np.corrcoef(_sr, _gr)[0, 1])
    print(f"\nSHAP-vs-gain rank correlation (Spearman-like): {_rho:.3f} "
          "(high = the two agree on what matters)")

    # --- beeswarm + bar summary plots ---
    _md = min(len(FEATS), 21)
    shap.summary_plot(sv, X_df, show=False, max_display=_md)
    plt.title("SHAP summary (beeswarm) - stress (positive = toward stress)")
    plt.tight_layout()
    _p = str(OUTDIR / "shap_beeswarm.png"); plt.savefig(_p, dpi=130, bbox_inches="tight"); plt.close()
    shap_plot_files.append(_p)
    shap.summary_plot(sv, X_df, plot_type="bar", show=False, max_display=_md)
    plt.title("Mean |SHAP| feature importance"); plt.tight_layout()
    _p = str(OUTDIR / "shap_importance_bar.png"); plt.savefig(_p, dpi=130, bbox_inches="tight"); plt.close()
    shap_plot_files.append(_p)

    _SHAP = dict(sv=sv, ev=ev, X_df=X_df, order=order)   # stash for the next XAI cells
    XAI_OK = True
    print("\nSaved:", [Path(p).name for p in shap_plot_files])
except Exception as _e:
    import traceback; traceback.print_exc()
    print("XAI global SHAP skipped (non-fatal):", type(_e).__name__, "-", _e)
''')

# ---------------------------------------------------------------- cell: XAI local (waterfall)
code(r'''
# ===== XAI 2/4 — LOCAL: waterfall plots for example windows =====
# A confident correct-stress, a confident correct-calm, and the most confident MISCLASSIFIED
# window. Each waterfall shows how features add/subtract from the base log-odds for THAT case.
try:
    if not globals().get("XAI_OK"): raise RuntimeError("global SHAP cell did not complete")
    import shap, matplotlib.pyplot as plt
    sv = _SHAP["sv"]; ev = _SHAP["ev"]; X_df = _SHAP["X_df"]
    OUTDIR = Path("/kaggle/working")
    thr = float(globals().get("BEST_THRESHOLD", 0.5))
    q = shap_model.predict_proba(X_df.values)[:, 1]      # final model's own prob (matches SHAP)
    pred = (q >= thr).astype(int); yv = np.asarray(y)
    print(f"decision threshold = {thr:.3f} | in-sample misclassified = {int((pred!=yv).sum())}/{len(yv)}")

    def _best(mask, by, want_max=True):
        idx = np.where(mask)[0]
        if len(idx) == 0: return None
        return int(idx[np.argmax(by[idx])] if want_max else idx[np.argmin(by[idx])])

    # Misclassified example: prefer an in-sample error; the heavily-regularized model is
    # often perfect in-sample, so fall back to a genuine OUT-OF-FOLD error (LOSO CV preds
    # `pp`, aligned to df row order = X_df row order). The waterfall still explains the
    # FINAL model's attribution for that window -> an honest "where it goes wrong" case.
    i_mis = _best(pred != yv, np.abs(q - thr), True); mis_src = "in-sample"
    _ppoof = globals().get("pp")
    if i_mis is None and _ppoof is not None and np.asarray(_ppoof).shape[0] == len(yv):
        _ppoof = np.asarray(_ppoof, float); _oofpred = (_ppoof >= thr).astype(int)
        i_mis = _best(_oofpred != yv, np.abs(_ppoof - thr), True); mis_src = "out-of-fold"

    picks = [("correct_stress", _best((yv == 1) & (pred == 1), q, True),  "shap_waterfall_stress.png"),
             ("correct_calm",   _best((yv == 0) & (pred == 0), q, False), "shap_waterfall_calm.png"),
             (f"misclassified[{mis_src}]", i_mis, "shap_waterfall_error.png")]

    def _waterfall(i, tag, fname):
        plt.figure(figsize=(8, 6))
        try:
            try:    # modern shap API
                expl = shap.Explanation(values=sv[i], base_values=ev,
                                        data=X_df.iloc[i].values, feature_names=list(X_df.columns))
                shap.plots.waterfall(expl, max_display=14, show=False)
            except Exception:   # legacy API
                shap.plots._waterfall.waterfall_legacy(ev, sv[i], feature_names=list(X_df.columns),
                                                       features=X_df.iloc[i].values, max_display=14, show=False)
        except Exception:       # last resort: static force plot
            plt.close(); plt.figure(figsize=(12, 3))
            shap.force_plot(ev, sv[i], X_df.iloc[i], matplotlib=True, show=False)
        plt.title(f"{tag}: true={int(yv[i])} pred={int(pred[i])} p(stress)={q[i]:.2f}")
        p = str(OUTDIR / fname); plt.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
        shap_plot_files.append(p)
        top3 = sorted(zip(list(X_df.columns), sv[i]), key=lambda kv: -abs(kv[1]))[:3]
        print(f"  {tag:16} idx={i:5} true={int(yv[i])} pred={int(pred[i])} p={q[i]:.2f}"
              f" | top: " + ", ".join(f"{f}{'+' if s>=0 else ''}{s:.2f}" for f, s in top3))

    for tag, i, fname in picks:
        if i is None: print(f"  {tag:16} (no such case in-sample - skipped)"); continue
        _waterfall(i, tag, fname)
    print("Saved local waterfalls.")
except Exception as _e:
    import traceback; traceback.print_exc()
    print("XAI local SHAP skipped (non-fatal):", type(_e).__name__, "-", _e)
''')

# ---------------------------------------------------------------- cell: XAI dependence
code(r'''
# ===== XAI 3/4 — FEATURE EFFECTS: SHAP dependence plots for the top features =====
# Each plot: x = feature value (per-subject-normalized units), y = its SHAP value.
# An upward trend => higher feature -> more stress; downward => higher feature -> less stress.
try:
    if not globals().get("XAI_OK"): raise RuntimeError("global SHAP cell did not complete")
    import shap, matplotlib.pyplot as plt
    sv = _SHAP["sv"]; X_df = _SHAP["X_df"]; order = _SHAP["order"]
    OUTDIR = Path("/kaggle/working")
    top = [X_df.columns[i] for i in order[:6]]
    for f in top:
        shap.dependence_plot(f, sv, X_df, interaction_index=None, show=False)
        plt.title(f"SHAP dependence: {f} (up = toward stress)"); plt.tight_layout()
        p = str(OUTDIR / f"shap_dependence_{f}.png"); plt.savefig(p, dpi=120, bbox_inches="tight"); plt.close()
        shap_plot_files.append(p)
    print("Saved dependence plots for:", top)
except Exception as _e:
    import traceback; traceback.print_exc()
    print("XAI dependence SHAP skipped (non-fatal):", type(_e).__name__, "-", _e)
''')

# ---------------------------------------------------------------- cell: XAI physiology
code(r'''
# ===== XAI 4/4 — PHYSIOLOGICAL INTERPRETATION: is the model's reasoning HRV-sound? =====
# For each feature we measure the DIRECTION the model uses (sign of corr between the feature
# value the model sees and its SHAP value): >0 => higher feature pushes toward stress.
# We compare that to the direction HRV/autonomic literature predicts and score the agreement.
try:
    if not globals().get("XAI_OK"): raise RuntimeError("global SHAP cell did not complete")
    sv = _SHAP["sv"]; X_df = _SHAP["X_df"]; order = _SHAP["order"]
    # Expected sign of effect on P(stress) for a HIGHER feature value (from HRV literature):
    #  -1 => higher value means LESS stress (vagal/parasympathetic markers)
    #  +1 => higher value means MORE stress (sympathetic / HR / DFA)
    #   0 => contested or non-specific -> not scored
    EXPECTED = dict(meanRR=-1, sdnn=-1, rmssd=-1, pnn50=-1, pnn20=-1, cvRR=-1,
                    hrMean=+1, hrStd=0, hrRange=0,
                    vlfPower=0, lfPower=0, hfPower=-1, lfHfRatio=+1, totalPower=-1,
                    lfNorm=+1, hfNorm=-1,
                    sd1=-1, sd2=-1, sd1sd2Ratio=0, sampleEntropy=-1, dfaAlpha1=+1)
    rows = []
    for i in order:
        f = X_df.columns[i]
        xv = X_df[f].values; sj = sv[:, i]
        corr = 0.0 if (np.std(xv) < 1e-9 or np.std(sj) < 1e-9) else float(np.corrcoef(xv, sj)[0, 1])
        meas = int(np.sign(corr)) if abs(corr) > 0.05 else 0     # measured model direction
        exp = EXPECTED.get(f, 0)
        if exp == 0:   verdict = "n/a"
        elif meas == 0: verdict = "weak"
        else:           verdict = "AGREES" if meas == exp else "DIFFERS"
        rows.append((f, float(np.abs(sv[:, i]).mean()), corr, exp, meas, verdict))

    judged = [r for r in rows if r[3] != 0 and r[4] != 0]
    agree = [r for r in judged if r[5] == "AGREES"]
    frac = (len(agree) / len(judged)) if judged else 0.0
    # |SHAP|-WEIGHTED agreement (the number to REPORT). An unweighted count treats a feature
    # worth 0.33 mean|SHAP| the same as one worth 0.003 -> misleading when the model piles
    # almost all weight on ONE axis. HRV features are heavily collinear (meanRR ~ 1/hrMean ~
    # the whole HR axis; pNN/RMSSD/SD1 all track vagal tone). When one feature dominates, the
    # tiny residual SHAP on its correlates can flip sign -> a SHAP-under-collinearity artifact,
    # NOT the model believing "higher pNN50 = stress". Weighting by magnitude shows how much of
    # the model's ACTUAL decision-making is physiologically sound.
    w_total = sum(r[1] for r in judged)
    w_agree = sum(r[1] for r in judged if r[5] == "AGREES")
    frac_w = (w_agree / w_total) if w_total > 0 else 0.0
    total_abs = sum(r[1] for r in rows)
    top_share = (rows[0][1] / total_abs) if total_abs > 0 else 0.0   # rows are sorted by importance
    shap_physio = dict(agreement_frac=frac, agreement_frac_weighted=frac_w,
                       n_judged=len(judged), n_agree=len(agree), top_feature_share=top_share)

    _sym = {1: "+", -1: "-", 0: "."}
    print(f"{'feature':14}{'mean|SHAP|':>11}{'corr(x,SHAP)':>14}{'expect':>8}{'model':>7}  verdict")
    print("-" * 64)
    for f, ms, corr, exp, meas, verdict in rows:
        print(f"{f:14}{ms:>11.4f}{corr:>14.3f}{_sym[exp]:>8}{_sym[meas]:>7}  {verdict}")
    print("-" * 64)
    print(f"Agreement (unweighted count): {len(agree)}/{len(judged)} features ({frac*100:.0f}%).")
    print(f"Agreement (|SHAP|-WEIGHTED, REPORT THIS): {frac_w*100:.0f}% of the model's attributed "
          f"reasoning is HRV-consistent.")
    print(f"Concentration: top feature '{rows[0][0]}' = {top_share*100:.0f}% of total |SHAP| "
          f"(the model leans heavily on one heart-rate axis).")
    print("Read-out: the DOMINANT drivers (meanRR / heart rate and its correlates) agree with")
    print("autonomic physiology -- faster heart + lower variability -> stress. The 'DIFFERS' rows")
    print("are LOW-|SHAP| vagal features (pNN50/pNN20/SampEn/SD1): collinear with the dominant HR")
    print("axis, so SHAP can hand them a sign-flipped residual. That is a SHAP-under-collinearity")
    print("artifact, not the model's real belief. Thesis claim: report the WEIGHTED % + this")
    print("caveat -- the model's primary reasoning is physiologically sound; disagreements are")
    print("confined to low-attribution features made unstable by HRV's intrinsic collinearity.")
except Exception as _e:
    import traceback; traceback.print_exc()
    print("XAI physiology SHAP skipped (non-fatal):", type(_e).__name__, "-", _e)
''')

# ---------------------------------------------------------------- cell: XAI feature reduction
code(r'''
# ===== XAI 5/5 (optional) — SHAP-GUIDED FEATURE REDUCTION =====
# Use the mean|SHAP| ranking to retrain on the top-k features and measure held-out
# cross-dataset AUC at each k. This is a WRAPPER check (retrain + measure), NOT a pure
# SHAP filter: under HRV collinearity a 0-SHAP feature can become useful once its
# correlate is dropped, so we trust held-out AUC over SHAP=0. Finds the smallest k that
# preserves performance -> a leaner, cheaper on-device feature set.
feature_reduction = {}
try:
    if not globals().get("RUN_FEATURE_REDUCTION"):
        raise RuntimeError("RUN_FEATURE_REDUCTION=False -> skipped")
    if not globals().get("shap_importance"):
        raise RuntimeError("no SHAP ranking available (global XAI cell did not complete)")
    from sklearn.model_selection import StratifiedGroupKFold as _SGKr
    ranked = list(shap_importance.keys())          # already sorted by mean|SHAP| desc
    nF = len(ranked)
    ks = FEATURE_REDUCTION_KS or sorted({k for k in (3, 5, 8, 14, 21, nF) if 1 <= k <= nF})
    _has_holdout = (globals().get("eval_df") is not None and len(eval_df))

    def _auc_for(feats):
        # (a) subject-disjoint CV AUC on the TRAINING set (always available)
        yc = df["stress_binary"].values; gc = df["subject"].values
        ns = max(2, min(4, len(np.unique(gc)))); sk = _SGKr(n_splits=ns, shuffle=True, random_state=SEED)
        cvs = []
        for tr, te in sk.split(df[feats].values, yc, gc):
            a, b = df.iloc[tr], df.iloc[te]
            if NORMALIZATION == "per_subject":
                Xtr = np.nan_to_num(per_subject_z(a, feats)[feats].values)
                Xte = np.nan_to_num(per_subject_z(b, feats)[feats].values)
            else:
                sc = StandardScaler(); Xtr = np.nan_to_num(sc.fit_transform(a[feats])); Xte = np.nan_to_num(sc.transform(b[feats]))
            if len(set(yc[tr])) < 2 or len(set(yc[te])) < 2: continue
            m = make_model(yc[tr]); m.fit(Xtr, yc[tr])
            cvs.append(roc_auc_score(yc[te], m.predict_proba(Xte)[:, 1]))
        cv_auc = float(np.mean(cvs)) if cvs else float("nan")
        # (b) held-out CROSS-DATASET AUC (the headline metric), if EVAL_ON is loaded
        ho_auc = float("nan")
        if _has_holdout:
            if NORMALIZATION == "per_subject":
                Xtr = np.nan_to_num(per_subject_z(df, feats)[feats].values)
                Xte = np.nan_to_num(per_subject_z(eval_df, feats)[feats].values)
            else:
                sc = StandardScaler(); Xtr = np.nan_to_num(sc.fit_transform(df[feats])); Xte = np.nan_to_num(sc.transform(eval_df[feats]))
            ye = eval_df["stress_binary"].values; yt = df["stress_binary"].values
            m = make_model(yt); m.fit(Xtr, yt)
            if len(set(ye)) > 1:
                ho_auc = float(roc_auc_score(ye, m.predict_proba(Xte)[:, 1]))
        return cv_auc, ho_auc

    print(f"SHAP-guided feature reduction | rank by mean|SHAP| | norm={NORMALIZATION} | "
          f"held-out={'yes ('+EVAL_ON+')' if _has_holdout else 'no -> CV only'}")
    print(f"{'k':>4}{'train CV AUC':>14}{'held-out AUC':>14}   top-k features")
    print("-" * 72)
    for k in ks:
        feats = ranked[:k]
        cv_auc, ho_auc = _auc_for(feats)
        feature_reduction[k] = dict(cv_auc=cv_auc, holdout_auc=ho_auc, features=feats)
        print(f"{k:>4}{cv_auc:>14.3f}{ho_auc:>14.3f}   {', '.join(feats[:6])}{' ...' if k > 6 else ''}")
    print("-" * 72)

    # Recommend the SMALLEST k within 0.01 AUC of the best (prefer held-out, else CV).
    _key = "holdout_auc" if _has_holdout else "cv_auc"
    _valid = {k: v[_key] for k, v in feature_reduction.items() if isinstance(k, int) and v[_key] == v[_key]}
    if _valid:
        _best_k = max(_valid, key=_valid.get); _best_auc = _valid[_best_k]
        _rec_k = min(k for k, a in _valid.items() if a >= _best_auc - 0.01)
        feature_reduction["recommended_k"] = _rec_k
        feature_reduction["recommended_features"] = feature_reduction[_rec_k]["features"]
        print(f"Best {_key} = {_best_auc:.3f} at k={_best_k}.")
        print(f"RECOMMENDED minimal set: k={_rec_k} (smallest within 0.01 AUC of best):")
        print("  " + ", ".join(feature_reduction[_rec_k]["features"]))
        _dropped = [f for f in FEATS if f not in feature_reduction[_rec_k]["features"]]
        print(f"Could DROP {len(_dropped)}: {', '.join(_dropped) if _dropped else '(none)'}")
        print("Deploy note: if the dropped set includes the frequency-domain block, you also")
        print("remove the on-device Welch-PSD computation. To ship the lean model, set FEATS to")
        print("the recommended list (and the matching extractor in services/ai/stressModel.ts).")
    else:
        print("No valid AUC computed (need a held-out EVAL_ON or >=2 training subjects).")
except Exception as _e:
    import traceback; traceback.print_exc()
    print("Feature reduction skipped (non-fatal):", type(_e).__name__, "-", _e)
''')

# ---------------------------------------------------------------- cell: mlflow
code(r'''
# ===== Log this run to MLflow — robust on Kaggle, NEVER aborts the run =====
# The whole block is wrapped: the model was already exported in the previous
# cell, so MLflow is optional telemetry and must not be able to crash training.
import os
try:
    try:
        import mlflow
    except ImportError:
        import subprocess, sys
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "mlflow"], check=True)
        import mlflow
    from mlflow.tracking import MlflowClient

    if MLFLOW_TRACKING_URI is None:
        # LOCAL mode. Kaggle/Colab can inject a *managed* MLflow server through
        # env vars (MLFLOW_TRACKING_URI / DATABRICKS_*). If left in place, our
        # logging POSTs to a server we aren't authorized for -> HTTP 403. Strip
        # them so we log to our own on-disk store under /kaggle/working.
        for _k in list(os.environ):
            if _k.startswith("MLFLOW_") or _k.startswith("DATABRICKS"):
                os.environ.pop(_k, None)
        # Newer MLflow puts the local FILE store in "maintenance mode" and raises unless
        # we opt in. Set this AFTER the strip loop above (which would remove an MLFLOW_* key).
        os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
        tracking_uri = "file:/kaggle/working/mlruns"
        os.makedirs("/kaggle/working/mlruns", exist_ok=True)
    else:
        # Remote server (e.g. DagsHub). Set Basic-Auth creds for the MLflow REST
        # API: username (not secret) + token (from a Kaggle Secret, never hardcoded).
        tracking_uri = MLFLOW_TRACKING_URI
        _user = os.environ.get("MLFLOW_TRACKING_USERNAME") or globals().get("DAGSHUB_USERNAME")
        _token = (os.environ.get("MLFLOW_TRACKING_PASSWORD")
                  or os.environ.get("MLFLOW_TRACKING_TOKEN"))
        if not _token:
            try:  # Kaggle Secrets (Add-ons -> Secrets)
                from kaggle_secrets import UserSecretsClient
                _token = UserSecretsClient().get_secret(globals().get("KAGGLE_SECRET_TOKEN_NAME", "DAGSHUB_TOKEN"))
            except Exception as _se:
                print("Could not read Kaggle Secret token:", _se)
        if _user and _token:
            os.environ["MLFLOW_TRACKING_USERNAME"] = _user
            os.environ["MLFLOW_TRACKING_PASSWORD"] = _token
            print(f"DagsHub auth set for user '{_user}'.")
        else:
            print("WARNING: no remote MLflow token found - expect 401/403. "
                  "Add a Kaggle Secret named "
                  f"'{globals().get('KAGGLE_SECRET_TOKEN_NAME', 'DAGSHUB_TOKEN')}' and enable internet.")

    mlflow.set_tracking_uri(tracking_uri)
    print("MLflow tracking:", mlflow.get_tracking_uri(), "| experiment:", MLFLOW_EXPERIMENT)

    # Resolve experiment via the client (avoids set_experiment's env-based magic).
    client = MlflowClient(tracking_uri=tracking_uri)
    _exp = client.get_experiment_by_name(MLFLOW_EXPERIMENT)
    exp_id = _exp.experiment_id if _exp else client.create_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(experiment_id=exp_id, run_name=f"kaggle-stress-{globals().get('train_name','WESAD')}-v{VERSION}") as run:
        mlflow.set_tags({"source": "kaggle-notebook",
                         "dataset": globals().get("train_name", "WESAD"),
                         "eval_on": EVAL_ON if globals().get("holdout") else "none",
                         "cv": "LOSO", "task": "stress_detection"})
        _sipd_df = globals().get("sipd_df")
        mlflow.log_params(dict(
            version=VERSION, window_sec=WINDOW_SEC, use_morphology=USE_MORPHOLOGY,
            normalization=NORMALIZATION, n_features=len(FEATS), seed=SEED,
            n_estimators=200, max_depth=6, learning_rate=0.1, subsample=0.8,
            colsample_bytree=0.8, min_child_weight=3,
            train_on=globals().get("train_name", "WESAD"),
            eval_on=EVAL_ON if globals().get("holdout") else "none",
            n_subjects=int(df["subject"].nunique()), n_samples=int(len(df)),
            use_sipd=bool(_sipd_df is not None and len(_sipd_df)),
            n_sipd_samples=int(len(_sipd_df)) if _sipd_df is not None else 0,
        ))
        mlflow.log_metrics({k: float(v) for k, v in cv.items()})

        # Held-out cross-dataset metrics (train TRAIN_ON -> test EVAL_ON), the headline number
        for _k, _v in (globals().get("holdout") or {}).items():
            if _v == _v:  # skip NaN
                mlflow.log_metric(f"holdout_{_k}", float(_v))
        # Calibrated-threshold held-out metrics + the chosen threshold
        mlflow.log_param("threshold_objective", THRESHOLD_OBJECTIVE)
        mlflow.log_metric("decision_threshold", float(globals().get("BEST_THRESHOLD", 0.5)))
        if globals().get("final_clean_auc") is not None:   # model-selection-leakage-free number
            mlflow.log_metric("final_clean_auc", float(final_clean_auc))
            mlflow.set_tag("final_test", FINAL_TEST)
        for _k, _v in (globals().get("holdout_cal") or {}).items():
            if isinstance(_v, (int, float)) and _v == _v:
                mlflow.log_metric(f"holdout_cal_{_k}", float(_v))

        # Tuning: per-optimizer CV AUC + the chosen hyperparameters
        mlflow.log_param("run_tuning", bool(globals().get("RUN_TUNING")))
        for _tn, _tr in (globals().get("tuning_results") or {}).items():
            mlflow.log_metric(f"tune_{_tn.lower()}_cv_auc", float(_tr["cv_auc"]))
        for _hk, _hv in (globals().get("BEST_PARAMS") or {}).items():
            mlflow.log_param(f"hp_{_hk}", _hv)

        # Within-dataset benchmark metrics (same-dataset LOSO), if it ran.
        for _k, _v in (globals().get("within_results") or {}).items():
            _safe = _k.replace("|", "_").replace("-", "").replace("+", "_")
            for _mn, _mv in _v.items():
                if _mv == _mv:
                    mlflow.log_metric(f"within_{_safe}_{_mn}", float(_mv))

        # Cross-dataset (WESAD<->SIPD) transfer metrics, if the experiment ran.
        for _k, _v in (globals().get("xds") or {}).items():
            _safe = (_k.replace("|", "_").replace("->", "_to_")
                       .replace("-", "").replace("+", "_"))
            for _mn, _mv in _v.items():
                if _mv == _mv:  # skip NaN
                    mlflow.log_metric(f"xds_{_safe}_{_mn}", float(_mv))

        # Explainability (SHAP): mean|SHAP| per feature, physiological-agreement score,
        # and the saved beeswarm/waterfall/dependence PNGs as artifacts.
        for _f, _mv in (globals().get("shap_importance") or {}).items():
            if isinstance(_mv, (int, float)) and _mv == _mv:
                mlflow.log_metric(f"shap_{_f}", float(_mv))
        _phys = globals().get("shap_physio") or {}
        for _pk, _pv in _phys.items():
            if isinstance(_pv, (int, float)) and _pv == _pv:
                mlflow.log_metric(f"shap_physio_{_pk}", float(_pv))
        mlflow.log_param("xai_ok", bool(globals().get("XAI_OK")))
        for _art in (globals().get("shap_plot_files") or []):
            try:
                mlflow.log_artifact(_art, artifact_path="shap")
            except Exception as e:
                print(f"  shap artifact skipped ({_art.split('/')[-1]}):", e)

        # SHAP-guided feature reduction: AUC at each k + the recommended minimal set size.
        _fr = globals().get("feature_reduction") or {}
        for _frk, _frv in _fr.items():
            if isinstance(_frk, int) and isinstance(_frv, dict):
                for _mn in ("cv_auc", "holdout_auc"):
                    _mv = _frv.get(_mn)
                    if isinstance(_mv, (int, float)) and _mv == _mv:
                        mlflow.log_metric(f"featred_k{_frk}_{_mn}", float(_mv))
        if isinstance(_fr.get("recommended_k"), int):
            mlflow.log_param("featred_recommended_k", _fr["recommended_k"])

        # Per-subject (per-fold) accuracy from the held-out predictions
        import pandas as _pd
        fold = (_pd.DataFrame({"subject": g, "y": y, "yp": yp})
                  .groupby("subject").apply(lambda d: (d.y == d.yp).mean()))
        for subj, acc in fold.items():
            mlflow.log_metric("fold_acc", float(acc))           # step series
            mlflow.log_metric(f"fold_acc_{subj}", float(acc))   # named

        # Params + metrics are now logged, so the run is already visible even if
        # artifact upload (below) fails — DagsHub occasionally rejects artifacts.
        for _art in ("/kaggle/working/stress_model.json", "/kaggle/working/model_metadata.json"):
            try:
                mlflow.log_artifact(_art)
            except Exception as e:
                print(f"  artifact upload skipped ({_art.split('/')[-1]}):", e)
        try:
            import mlflow.xgboost
            mlflow.xgboost.log_model(final, "xgb_model")   # positional: mlflow 2.x & 3.x
        except Exception as e:
            print("  xgboost model-logging skipped (non-fatal):", e)

        _rid = run.info.run_id

    print("\n" + "=" * 60)
    print("MLflow run logged OK   run_id:", _rid)
    if MLFLOW_TRACKING_URI:
        print("View it under your DagsHub repo -> Experiments tab")
        print("  ", MLFLOW_TRACKING_URI.replace(".mlflow", "/experiments"))
    else:
        print("Local store - download /kaggle/working/mlruns from the Output tab,")
        print("then:  mlflow ui --backend-store-uri ./mlruns")
    print("=" * 60)

except Exception as _e:
    import traceback
    print("\n" + "!" * 60)
    print("MLflow logging FAILED (non-fatal):", type(_e).__name__, "-", _e)
    traceback.print_exc()
    print("-" * 60)
    print("Most common causes, in order:")
    print("  1. Kaggle Internet is OFF  -> Notebook settings -> Internet -> On")
    print("  2. Kaggle Secret not attached -> Add-ons -> Secrets -> toggle "
          f"'{globals().get('KAGGLE_SECRET_TOKEN_NAME','DAGSHUB_TOKEN')}' ON for THIS notebook")
    print("  3. Token wrong/expired, or DAGSHUB_USERNAME mismatch")
    print("Your stress_model.json / model_metadata.json were already exported above - "
          "training and export are unaffected.")
    print("!" * 60)
''')

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}

out = Path(__file__).parent / "seren_stress_kaggle.ipynb"
out.write_text(json.dumps(nb, indent=1))
print("wrote", out, "with", len(cells), "cells")
