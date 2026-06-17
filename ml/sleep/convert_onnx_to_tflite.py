"""
Convert the sleep ONNX model -> TFLite for on-device inference via
react-native-fast-tflite (New-Architecture compatible; onnxruntime-react-native
is a legacy module that Expo SDK 54 autolinking won't link).

Env:  pip install -U onnx2tf tensorflow onnxruntime onnx onnx-graphsurgeon onnxsim
Run:  python ml/sleep/convert_onnx_to_tflite.py
Out:  assets/ml/sleep/sleep_stage_model.tflite  (+ parity report vs ONNX)

Input is pinned to a static [1, 41, 11] (batch 1, SEQ_LEN 41, 11 features) which is
exactly how the app windows the night -> robust TFLite conversion of the BiLSTM.

NOTE: feature count changed 12 -> 11 after XAI confirmed `immobility_frac` is dead
weight. If you see a shape-mismatch error, you're converting an old (pre-prune) ONNX;
re-run the training notebook to produce a fresh 11-feature ONNX.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ONNX = ROOT / "assets" / "ml" / "sleep" / "sleep_stage_model.onnx"
OUTDIR = ROOT / "assets" / "ml" / "sleep"
WORK = OUTDIR / "_tflite_work"
SEQ, FEAT, CLS = 41, 11, 4


def convert():
    WORK.mkdir(parents=True, exist_ok=True)
    # Static input shape -> avoids dynamic-LSTM conversion issues in TFLite.
    subprocess.run(
        # -rtpo Erf GeLU: replace Erf/GELU with native TFLite ops (no Flex / Select-TF-ops),
        # so react-native-fast-tflite (standard runtime) can run the model on-device.
        [sys.executable, "-m", "onnx2tf", "-i", str(ONNX), "-o", str(WORK),
         "-ois", f"features:1,{SEQ},{FEAT}", "-rtpo", "Erf", "GeLU", "-osd", "-n"],
        check=True,
    )
    cands = sorted(WORK.glob("*_float32.tflite")) or sorted(WORK.glob("*.tflite"))
    if not cands:
        raise SystemExit("onnx2tf produced no .tflite — check the log above")
    target = OUTDIR / "sleep_stage_model.tflite"
    target.write_bytes(cands[0].read_bytes())
    print(f"WROTE {target}  ({target.stat().st_size // 1024} KB)")
    return target


def verify(target):
    import onnxruntime as ort
    import tensorflow as tf

    x = np.random.randn(1, SEQ, FEAT).astype("float32")
    onnx_out = ort.InferenceSession(str(ONNX)).run(None, {"features": x})[0]

    it = tf.lite.Interpreter(model_path=str(target))
    it.allocate_tensors()
    ind, outd = it.get_input_details(), it.get_output_details()
    it.set_tensor(ind[0]["index"], x)
    it.invoke()
    tfl_out = it.get_tensor(outd[0]["index"])

    print("ONNX out shape  :", onnx_out.shape)
    print("TFLite out shape:", tfl_out.shape, "| input detail:", list(ind[0]["shape"]))
    print("argmax match    :",
          bool(np.array_equal(onnx_out.argmax(-1), tfl_out.reshape(onnx_out.shape).argmax(-1))))
    print("MAX ABS DIFF    :", float(np.max(np.abs(onnx_out - tfl_out.reshape(onnx_out.shape)))))


if __name__ == "__main__":
    verify(convert())
