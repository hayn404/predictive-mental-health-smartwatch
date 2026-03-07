"""Tests for training pipeline."""

import json
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import PipelineConfig
from train import load_features, train_and_evaluate, export_model_json


@pytest.fixture
def cfg():
    """Config with small synthetic data for fast tests."""
    c = PipelineConfig()
    c.data.n_subjects = 5
    c.data.windows_per_subject = 20
    c.data.stress_ratio = 0.35
    c.data.seed = 42
    c.data.features_csv = "/nonexistent/features.csv"
    c.data.wesad_dir = "/nonexistent/wesad"
    c.model.n_estimators = 10
    c.model.max_depth = 3
    with tempfile.TemporaryDirectory() as tmpdir:
        c.export.model_dir = tmpdir
        c.export.app_assets_dir = str(Path(tmpdir) / "assets")
        yield c


class TestLoadFeatures:
    def test_synthetic_fallback(self, cfg):
        df = load_features(cfg)
        assert len(df) == 100  # 5 subjects x 20 windows
        assert "stress_binary" in df.columns
        assert "subject" in df.columns


class TestTrainAndEvaluate:
    def test_produces_metrics(self, cfg):
        df = load_features(cfg)
        results = train_and_evaluate(df, cfg)

        assert "cv_accuracy" in results
        assert "cv_auc_roc" in results
        assert "cv_f1_weighted" in results
        assert results["cv_accuracy"] > 0.5  # Better than random
        assert results["model"] is not None
        assert results["scaler"] is not None

    def test_fold_metrics(self, cfg):
        df = load_features(cfg)
        results = train_and_evaluate(df, cfg)

        assert len(results["fold_metrics"]) == 5  # One per subject


class TestExport:
    def test_json_export(self, cfg):
        df = load_features(cfg)
        results = train_and_evaluate(df, cfg)
        model_path = export_model_json(
            model=results["model"],
            scaler=results["scaler"],
            feature_cols=results["feature_cols"],
            metrics=results,
            cfg=cfg,
        )

        assert model_path.exists()
        with open(model_path) as f:
            model_json = json.load(f)

        assert "trees" in model_json
        assert "normalization" in model_json
        assert "features" in model_json
        assert len(model_json["trees"]) == cfg.model.n_estimators
        assert model_json["numFeatures"] == len(results["feature_cols"])

    def test_metadata_export(self, cfg):
        df = load_features(cfg)
        results = train_and_evaluate(df, cfg)
        export_model_json(
            model=results["model"],
            scaler=results["scaler"],
            feature_cols=results["feature_cols"],
            metrics=results,
            cfg=cfg,
        )

        metadata_path = Path(cfg.export.model_dir) / cfg.export.metadata_json
        assert metadata_path.exists()
        with open(metadata_path) as f:
            meta = json.load(f)
        assert "normalization" in meta
        assert "mean" in meta["normalization"]
        assert "std" in meta["normalization"]
