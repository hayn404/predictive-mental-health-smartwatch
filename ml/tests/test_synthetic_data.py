"""Tests for synthetic data generator."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from synthetic_data import generate_synthetic_dataset
from features import FEATURE_ORDER


class TestSyntheticData:
    def test_shape(self):
        df = generate_synthetic_dataset(n_subjects=3, windows_per_subject=10, seed=42)
        assert len(df) == 30
        assert df["subject"].nunique() == 3

    def test_features_present(self):
        df = generate_synthetic_dataset(n_subjects=2, windows_per_subject=5, seed=42)
        for feat in FEATURE_ORDER:
            assert feat in df.columns, f"Missing feature: {feat}"

    def test_label_distribution(self):
        df = generate_synthetic_dataset(
            n_subjects=5, windows_per_subject=100,
            stress_ratio=0.35, seed=42,
        )
        ratio = (df["stress_binary"] == 1).mean()
        assert 0.25 < ratio < 0.45  # Approximately 35%

    def test_physiological_ranges(self):
        df = generate_synthetic_dataset(n_subjects=5, windows_per_subject=50, seed=42)

        assert df["hrMean"].min() >= 40
        assert df["hrMean"].max() <= 180
        assert df["rmssd"].min() >= 5
        assert df["tempMean"].min() >= 28
        assert df["tempMean"].max() <= 38

    def test_stress_vs_baseline_separation(self):
        """Stressed samples should have lower HRV and higher HR on average."""
        df = generate_synthetic_dataset(n_subjects=10, windows_per_subject=50, seed=42)

        baseline = df[df["stress_binary"] == 0]
        stressed = df[df["stress_binary"] == 1]

        assert stressed["rmssd"].mean() < baseline["rmssd"].mean()
        assert stressed["hrMean"].mean() > baseline["hrMean"].mean()
        assert stressed["lfHfRatio"].mean() > baseline["lfHfRatio"].mean()

    def test_reproducibility(self):
        df1 = generate_synthetic_dataset(n_subjects=3, windows_per_subject=10, seed=42)
        df2 = generate_synthetic_dataset(n_subjects=3, windows_per_subject=10, seed=42)
        assert df1.equals(df2)
