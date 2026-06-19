#!/usr/bin/env python3
"""
Orchestrate model training on Kaggle from a CI runner.

Pushes a Kaggle kernel that clones this (public) repo, acquires the dataset,
trains, and returns the artifacts (model + metrics + figures + feature CSVs).
Sleep runs on a GPU kernel; focus/bio-age run on CPU kernels that download the
PhysioNet datasets, extract features, and train.

  python ml/ci/kaggle_run.py --model sleep|focus|bioage --user <kaggle_user> \
      --repo owner/name --ref main [--mlflow]

Requires KAGGLE_USERNAME / KAGGLE_KEY in the environment.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# ── Shared kernel preamble: clone the public repo ────────────────────────────
PREAMBLE = '''\
import os, subprocess, sys, shutil, glob
def sh(*a): print("+"," ".join(a), flush=True); subprocess.run(list(a), check=True)
def find_dir(root, marker):
    for d, subs, files in os.walk(root):
        if marker in subs or marker in files:
            return d
    raise SystemExit(f"marker {{marker!r}} not found under {{root}}")
REPO_URL = {repo_url!r}
sh("git","clone","--depth","1","--branch",{ref!r},REPO_URL,"/kaggle/working/repo")
os.chdir("/kaggle/working/repo")
OUT="/kaggle/working"
'''

SLEEP_BODY = '''\
sh(sys.executable,"-m","pip","install","-q","-r","ml/sleep/requirements.txt")
sh(sys.executable,"-m","pip","install","-q","-U","onnx2tf","onnx-graphsurgeon","onnxsim","tf_keras","dvc[s3]")
# torch 2.3.1's torch.onnx import trips on Kaggle's newer onnxscript via beartype
# type-hints; removing beartype makes those decorators no-ops so the import works.
subprocess.run([sys.executable,"-m","pip","uninstall","-y","beartype"])
# Pull the sleep feature cache from DVC (DagsHub) -- single source of truth,
# same as every other model. No Kaggle dataset attached.
sh("dvc","remote","modify","--local","origin","access_key_id","{dagshub_token}")
sh("dvc","remote","modify","--local","origin","secret_access_key","{dagshub_token}")
sh("dvc","pull","ml/data/features/sleep.dvc")
sh(sys.executable,"ml/sleep/train.py","--params","params.yaml","--data","ml/data/features/sleep",
   "--out","assets/ml/sleep","--metrics","ml/sleep/metrics.json")
for f in ("assets/ml/sleep/sleep_stage_model.onnx","assets/ml/sleep/sleep_stage_model.onnx.data",
          "assets/ml/sleep/sleep_stage_model.tflite","assets/ml/sleep/sleep_model_metadata.json",
          "ml/sleep/metrics.json","ml/sleep/output/training_history.json"):
    if os.path.exists(f): shutil.copy(f, os.path.join(OUT, os.path.basename(f)))
for p in glob.glob("ml/sleep/figures/*.png"):
    shutil.copy(p, os.path.join(OUT, "fig__sleep__"+os.path.basename(p)))
'''

FOCUS_BODY = '''\
sh(sys.executable,"-m","pip","install","-q","xgboost","scikit-learn","scipy","pandas","matplotlib","shap","pyyaml")
COG="https://physionet.org/static/published-projects/consumer-grade-wearables/consumer-grade-wearables-1.0.0.zip"
sh("wget","-q","-O","/tmp/cog.zip",COG)                 # /tmp -> not in kernel output
os.makedirs("/tmp/cogx", exist_ok=True)
sh("unzip","-q","-o","/tmp/cog.zip","-d","/tmp/cogx")
src=find_dir("/tmp/cogx","pilot")                       # dir containing pilot/
exp="ml/focus/data/cogwear/cogwear-can-we-detect-cognitive-effort-with-consumer-grade-wearables-1.0.0"
os.makedirs(os.path.dirname(exp), exist_ok=True)
if not os.path.exists(exp): os.symlink(os.path.abspath(src), exp)
sh(sys.executable,"ml/focus/src/extract_cogwear_features.py")
sh(sys.executable,"ml/focus/src/train.py","--params","params.yaml","--data","ml/focus/data",
   "--device","samsung","--out","assets/ml/focus","--metrics","ml/focus/metrics.json","--figures","ml/focus/figures")
for f in ("ml/focus/data/cogwear_features_samsung.csv","ml/focus/data/cogwear_features_empatica.csv",
          "assets/ml/focus/focus_model.json","ml/focus/metrics.json"):
    if os.path.exists(f): shutil.copy(f, os.path.join(OUT, os.path.basename(f)))
for p in glob.glob("ml/focus/figures/*.png"):
    shutil.copy(p, os.path.join(OUT, "fig__focus__"+os.path.basename(p)))
'''

BIOAGE_BODY = '''\
sh(sys.executable,"-m","pip","install","-q","wfdb","xgboost","scikit-learn","scipy","pandas","matplotlib","shap","pyyaml")
os.makedirs("ml/bioage/data", exist_ok=True)
FAN="https://physionet.org/static/published-projects/fantasia/fantasia-database-1.0.0.zip"
AA="https://physionet.org/static/published-projects/autonomic-aging-cardiovascular/autonomic-aging-a-dataset-to-quantify-changes-of-cardiovascular-autonomic-function-during-healthy-aging-1.0.0.zip"
sh("wget","-q","-O","/tmp/fan.zip",FAN)                 # /tmp -> not in kernel output
sh("unzip","-q","-o","/tmp/fan.zip","-d","/tmp/fanx")
fsrc=os.path.dirname(glob.glob("/tmp/fanx/**/*.hea",recursive=True)[0])
if not os.path.exists("ml/bioage/data/fantasia"): os.symlink(os.path.abspath(fsrc),"ml/bioage/data/fantasia")
sh("wget","-q","-O","/tmp/aa.zip",AA)
sh("unzip","-q","-o","/tmp/aa.zip","-d","/tmp/aax")
asrc=find_dir("/tmp/aax","subject-info.csv")
if not os.path.exists("ml/bioage/data/autonomic_aging"): os.symlink(os.path.abspath(asrc),"ml/bioage/data/autonomic_aging")
sh(sys.executable,"ml/bioage/src/extract_bioage.py","fantasia")
sh(sys.executable,"ml/bioage/src/extract_bioage.py","aa")
sh(sys.executable,"ml/bioage/src/train.py","--params","params.yaml","--data","ml/bioage/data",
   "--out","assets/ml/bioage","--metrics","ml/bioage/metrics.json","--figures","ml/bioage/figures")
for f in ("ml/bioage/data/bioage_features_aa.csv","ml/bioage/data/bioage_features_fantasia.csv",
          "assets/ml/bioage/bioage_model.json","ml/bioage/metrics.json"):
    if os.path.exists(f): shutil.copy(f, os.path.join(OUT, os.path.basename(f)))
for p in glob.glob("ml/bioage/figures/*.png"):
    shutil.copy(p, os.path.join(OUT, "fig__bioage__"+os.path.basename(p)))
'''

# model -> (gpu, kaggle dataset_sources, body, {downloaded_name: repo_dest})
MODELS = {
    "sleep": dict(gpu=True, datasets_default=None, body=SLEEP_BODY, artifacts={
        "sleep_stage_model.onnx": "assets/ml/sleep/sleep_stage_model.onnx",
        "sleep_stage_model.onnx.data": "assets/ml/sleep/sleep_stage_model.onnx.data",
        "sleep_stage_model.tflite": "assets/ml/sleep/sleep_stage_model.tflite",
        "sleep_model_metadata.json": "assets/ml/sleep/sleep_model_metadata.json",
        "metrics.json": "ml/sleep/metrics.json",
        "training_history.json": "ml/sleep/output/training_history.json",
    }),
    "focus": dict(gpu=False, datasets_default=None, body=FOCUS_BODY, artifacts={
        "cogwear_features_samsung.csv": "ml/focus/data/cogwear_features_samsung.csv",
        "cogwear_features_empatica.csv": "ml/focus/data/cogwear_features_empatica.csv",
        "focus_model.json": "assets/ml/focus/focus_model.json",
        "metrics.json": "ml/focus/metrics.json",
    }),
    "bioage": dict(gpu=False, datasets_default=None, body=BIOAGE_BODY, artifacts={
        "bioage_features_aa.csv": "ml/bioage/data/bioage_features_aa.csv",
        "bioage_features_fantasia.csv": "ml/bioage/data/bioage_features_fantasia.csv",
        "bioage_model.json": "assets/ml/bioage/bioage_model.json",
        "metrics.json": "ml/bioage/metrics.json",
    }),
}


def ensure_kaggle_auth():
    """Set up Kaggle credentials for the CLI, supporting both auth styles:
      - NEW API token  -> KAGGLE_API_TOKEN  (CLI >= 1.8; written to ~/.kaggle/access_token)
      - LEGACY key      -> KAGGLE_USERNAME + KAGGLE_KEY (written to ~/.kaggle/kaggle.json)
    """
    kdir = Path.home() / ".kaggle"
    kdir.mkdir(parents=True, exist_ok=True)
    tok = (os.environ.get("KAGGLE_API_TOKEN") or "").strip()
    u = (os.environ.get("KAGGLE_USERNAME") or "").strip()
    k = (os.environ.get("KAGGLE_KEY") or "").strip()

    if tok:
        os.environ["KAGGLE_API_TOKEN"] = tok
        af = kdir / "access_token"
        af.write_text(tok)
        try:
            os.chmod(af, 0o600)
        except OSError:
            pass
        print(f"Kaggle auth: API token (length={len(tok)}), user={u!r}")
        return
    if u and k:
        kf = kdir / "kaggle.json"
        kf.write_text(json.dumps({"username": u, "key": k}))
        try:
            os.chmod(kf, 0o600)
        except OSError:
            pass
        print(f"Kaggle auth: legacy user={u!r}, key length={len(k)}")
        return
    print("WARN: no Kaggle credentials (set KAGGLE_API_TOKEN, or KAGGLE_USERNAME+KAGGLE_KEY)")


def sh(cmd, **kw):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run(cmd, check=True, **kw)


def kaggle(*args, **kw):
    try:
        return sh(["kaggle", *args], **kw)
    except FileNotFoundError:
        return sh([sys.executable, "-m", "kaggle", *args], **kw)


def dump_kernel_log(slug):
    """Fetch the kernel's execution log and print it (so failures are visible
    in the GitHub Actions console, not just on Kaggle)."""
    d = Path("kernel_log"); d.mkdir(exist_ok=True)
    try:
        kaggle("kernels", "output", slug, "-p", str(d))
    except Exception as e:
        print(f"(could not fetch kernel log: {e})")
        return
    logs = list(d.glob("*.log"))
    if not logs:
        print("(no .log file in kernel output)")
        return
    txt = logs[0].read_text(errors="replace")
    print(f"\n===== KAGGLE KERNEL LOG ({logs[0].name}) =====")
    try:
        for e in json.loads(txt):            # Kaggle log = JSON list of {stream,data}
            print(e.get("data", ""), end="")
    except Exception:
        print(txt[-12000:])
    print("\n===== END KERNEL LOG =====")


def push_and_wait(workdir, slug, timeout):
    kaggle("kernels", "push", "-p", str(workdir))
    t0 = time.time()
    while True:
        time.sleep(30)
        r = subprocess.run(["kaggle", "kernels", "status", slug], capture_output=True, text=True)
        out = (r.stdout + r.stderr).strip()
        print(out, flush=True)
        low = out.lower()
        if "complete" in low:
            return
        if "error" in low or "cancel" in low:
            dump_kernel_log(slug)
            sys.exit(f"Kaggle kernel failed: {out}")
        if time.time() - t0 > timeout:
            dump_kernel_log(slug)
            sys.exit("Kaggle kernel timed out")


def place_artifacts(model, out_dir):
    cfg = MODELS[model]
    placed = []
    for src, dst in cfg["artifacts"].items():
        p = out_dir / src
        if p.exists():
            Path(dst).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(p, dst); placed.append(dst); print("placed", dst)
        else:
            print("WARN missing artifact:", src)
    # figures: fig__<model>__<name>.png -> ml/<model>/figures/<name>.png
    for p in out_dir.glob(f"fig__{model}__*.png"):
        name = p.name.replace(f"fig__{model}__", "")
        dst = Path(f"ml/{model}/figures/{name}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(p, dst); placed.append(str(dst)); print("placed", dst)
    return placed


def log_to_mlflow(model, run_ref):
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not uri:
        print("MLFLOW_TRACKING_URI not set — skipping MLflow log.")
        return
    try:
        import mlflow
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(f"seren-{model}-train")
        mpath = Path(f"ml/{model}/metrics.json")
        with mlflow.start_run(run_name=f"kaggle-{model}-{run_ref}"):
            if mpath.exists():
                m = json.loads(mpath.read_text())
                for k, v in m.items():
                    if isinstance(v, (int, float)):
                        mlflow.log_metric(k, float(v))
                if m.get("eval_set_id"):
                    mlflow.log_param("eval_set_id", m["eval_set_id"])
                mlflow.log_artifact(str(mpath))   # metrics.json carries the *_ci95 lists
            for fig in Path(f"ml/{model}/figures").glob("*.png"):
                mlflow.log_artifact(str(fig), artifact_path="figures")
        print("Logged to MLflow.")
    except Exception as e:
        print(f"MLflow logging skipped ({e}).")


def main():
    ap = argparse.ArgumentParser(description="Run model training on Kaggle")
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--user", required=True, help="Kaggle username")
    ap.add_argument("--repo", required=True, help="owner/name of the GitHub repo")
    ap.add_argument("--ref", default="main")
    ap.add_argument("--cache", default=None, help="Kaggle dataset slug (sleep cache override)")
    ap.add_argument("--dagshub-token", default=os.environ.get("DAGSHUB_TOKEN", ""),
                    help="DagsHub token so the kernel can dvc pull features from DagsHub")
    ap.add_argument("--out", default="kaggle_out")
    ap.add_argument("--timeout", type=int, default=43200)
    ap.add_argument("--mlflow", action="store_true")
    args = ap.parse_args()

    ensure_kaggle_auth()
    cfg = MODELS[args.model]
    repo_url = f"https://github.com/{args.repo}.git"     # public repo: anonymous clone
    body = cfg["body"]
    datasets = []
    if cfg["datasets_default"]:                          # (legacy Kaggle-dataset path)
        slug = args.cache or f"{args.user}/{cfg['datasets_default']}"
        datasets = [slug]
        body = body.format(cache_name=slug.split("/")[-1])
    if "{dagshub_token}" in body:                        # sleep pulls features from DVC
        body = body.format(dagshub_token=args.dagshub_token)

    run_src = PREAMBLE.format(repo_url=repo_url, ref=args.ref) + body

    workdir = Path("kaggle_kernel")
    workdir.mkdir(exist_ok=True)
    (workdir / "run.py").write_text(run_src)
    slug_name = f"seren-{args.model}-train"
    meta = {
        "id": f"{args.user}/{slug_name}", "title": slug_name, "code_file": "run.py",
        "language": "python", "kernel_type": "script", "is_private": True,
        "enable_gpu": bool(cfg["gpu"]), "enable_internet": True,
        "dataset_sources": datasets, "competition_sources": [], "kernel_sources": [],
    }
    (workdir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))

    slug = f"{args.user}/{slug_name}"
    print(f"Launching {slug} (gpu={cfg['gpu']}) for model={args.model}")
    push_and_wait(workdir, slug, args.timeout)

    out_dir = Path(args.out); out_dir.mkdir(exist_ok=True)
    kaggle("kernels", "output", slug, "-p", str(out_dir))
    if not place_artifacts(args.model, out_dir):
        sys.exit("No artifacts downloaded — check the Kaggle kernel log.")
    if args.mlflow:
        log_to_mlflow(args.model, args.ref)
    print(f"{args.model} training complete; artifacts placed in repo.")


if __name__ == "__main__":
    main()
