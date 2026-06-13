"""
Seren — Sleep Stage Classification: ONNX Export
=================================================
Exports the trained PyTorch model to ONNX format for
on-device inference in the React Native app via onnxruntime-react-native.

Also generates the metadata JSON with normalization parameters
and class mapping needed by the TypeScript inference code.
"""

import json
import torch
import numpy as np
from pathlib import Path

import config as cfg
from model import create_model


def export_to_onnx(checkpoint_path: str = None, output_dir: str = None):
    """Export trained model to ONNX + metadata JSON."""
    if checkpoint_path is None:
        checkpoint_path = cfg.CHECKPOINT_DIR / "best_model.pt"
    else:
        checkpoint_path = Path(checkpoint_path)

    if output_dir is None:
        output_dir = cfg.EXPORT_DIR
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device("cpu")

    # Load model
    print("Loading model checkpoint...")
    model = create_model(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Create dummy inputs
    dummy_raw = torch.randn(1, cfg.NUM_RAW_CHANNELS, cfg.EPOCH_SAMPLES)
    dummy_aux = torch.randn(1, cfg.NUM_AUX_FEATURES)

    # Export to ONNX
    onnx_path = output_dir / "sleep_stage_model.onnx"
    print(f"Exporting to ONNX: {onnx_path}")

    torch.onnx.export(
        model,
        (dummy_raw, dummy_aux),
        str(onnx_path),
        input_names=["raw_signal", "aux_features"],
        output_names=["logits"],
        dynamic_axes={
            "raw_signal": {0: "batch"},
            "aux_features": {0: "batch"},
            "logits": {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
    )

    # Verify ONNX model
    import onnx
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    print("ONNX model verified successfully")

    # Verify with ONNX Runtime
    import onnxruntime as ort
    session = ort.InferenceSession(str(onnx_path))
    ort_inputs = {
        "raw_signal": dummy_raw.numpy(),
        "aux_features": dummy_aux.numpy(),
    }
    ort_output = session.run(None, ort_inputs)
    print(f"ONNX Runtime test output shape: {ort_output[0].shape}")

    # Compare PyTorch vs ONNX output
    with torch.no_grad():
        pt_output = model(dummy_raw, dummy_aux).numpy()
    max_diff = np.max(np.abs(pt_output - ort_output[0]))
    print(f"Max difference PyTorch vs ONNX: {max_diff:.6f}")

    # Generate metadata JSON
    metadata = {
        "version": "1.0.0",
        "modelType": "cnn_bilstm_sleep_classifier",
        "task": "sleep_stage_classification",
        "numClasses": cfg.NUM_CLASSES,
        "stageNames": cfg.STAGE_NAMES,
        "stageMap": cfg.APP_STAGE_MAP,
        "input": {
            "rawSignal": {
                "channels": cfg.NUM_RAW_CHANNELS,
                "channelNames": ["BVP", "ACC_magnitude"],
                "samplesPerEpoch": cfg.EPOCH_SAMPLES,
                "sampleRateHz": cfg.SAMPLE_RATE,
                "epochSeconds": cfg.EPOCH_SEC,
            },
            "auxFeatures": {
                "count": cfg.NUM_AUX_FEATURES,
                "names": ["HR_mean", "HR_std", "ACC_mean", "ACC_std", "TEMP_mean", "TEMP_slope"],
            },
        },
        "preprocessing": {
            "bvpBandpass": [cfg.BVP_LOW_HZ, cfg.BVP_HIGH_HZ],
            "normalization": "per_subject_z_score",
        },
        "training": {
            "valAccuracy": float(checkpoint.get("val_acc", 0)),
            "valF1": float(checkpoint.get("val_f1", 0)),
            "valKappa": float(checkpoint.get("val_kappa", 0)),
            "epoch": int(checkpoint.get("epoch", 0)),
        },
    }

    metadata_path = output_dir / "sleep_model_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nMetadata saved to: {metadata_path}")

    file_size_mb = onnx_path.stat().st_size / (1024 * 1024)
    print(f"ONNX model size: {file_size_mb:.2f} MB")
    print(f"\nExport complete. Files:")
    print(f"  Model:    {onnx_path}")
    print(f"  Metadata: {metadata_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Export sleep model to ONNX")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()
    export_to_onnx(args.checkpoint, args.output_dir)
