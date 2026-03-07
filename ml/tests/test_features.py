"""Tests for feature engineering pipeline."""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from features import (
    compute_time_domain,
    compute_frequency_domain,
    compute_nonlinear,
    compute_temperature_features,
    compute_accelerometer_features,
    FEATURE_ORDER,
)


class TestTimeDomain:
    def test_normal_rr_intervals(self):
        """Normal sinus rhythm ~70 BPM -> expected HRV ranges."""
        rng = np.random.RandomState(42)
        rr = rng.normal(857, 50, size=100)  # ~70 BPM
        result = compute_time_domain(rr)

        assert result["meanRR"] == pytest.approx(857, rel=0.1)
        assert result["hrMean"] == pytest.approx(70, rel=0.15)
        assert result["sdnn"] > 0
        assert result["rmssd"] > 0
        assert 0 <= result["pnn50"] <= 100
        assert 0 <= result["pnn20"] <= 100

    def test_insufficient_data(self):
        """Less than 5 RR intervals returns zeros."""
        rr = np.array([800, 810, 820])
        result = compute_time_domain(rr)
        assert result["meanRR"] == 0
        assert result["rmssd"] == 0

    def test_empty_input(self):
        result = compute_time_domain(np.array([]))
        assert all(v == 0 for v in result.values())


class TestFrequencyDomain:
    def test_sufficient_data(self):
        """300+ RR intervals (5-min window) should produce frequency features."""
        rng = np.random.RandomState(42)
        rr = rng.normal(857, 50, size=350)  # ~5 min of beats at ~70 BPM
        result = compute_frequency_domain(rr)

        assert result["totalPower"] > 0
        assert result["lfPower"] >= 0
        assert result["hfPower"] >= 0
        assert result["lfHfRatio"] >= 0

    def test_insufficient_data(self):
        rr = np.array([800, 810, 820, 830, 840])
        result = compute_frequency_domain(rr)
        assert result["totalPower"] == 0


class TestNonLinear:
    def test_poincare_descriptors(self):
        rng = np.random.RandomState(42)
        rr = rng.normal(857, 50, size=100)
        result = compute_nonlinear(rr)

        assert result["sd1"] > 0
        assert result["sd2"] > 0
        assert result["sd1sd2Ratio"] > 0

    def test_insufficient_data(self):
        rr = np.array([800, 810])
        result = compute_nonlinear(rr)
        assert result["sd1"] == 0


class TestTemperature:
    def test_normal_temperature(self):
        temp = np.linspace(33.0, 33.5, 1200)  # 5 min @ 4 Hz
        result = compute_temperature_features(temp, fs=4)

        assert result["tempMean"] == pytest.approx(33.25, rel=0.05)
        assert result["tempSlope"] > 0  # Rising
        assert result["tempRange"] == pytest.approx(0.5, rel=0.05)


class TestAccelerometer:
    def test_sedentary(self):
        rng = np.random.RandomState(42)
        acc = rng.normal(0, 0.01, size=(9600, 3))
        acc[:, 2] += 1.0  # Gravity on z-axis
        result = compute_accelerometer_features(acc, fs=32)

        assert result["accelMagnitudeMean"] == pytest.approx(1.0, rel=0.1)
        assert result["activityType"] == 0  # sedentary


class TestFeatureOrder:
    def test_count(self):
        """Feature order should have 29 features (28 + activityType handled separately)."""
        assert len(FEATURE_ORDER) == 29

    def test_no_duplicates(self):
        assert len(FEATURE_ORDER) == len(set(FEATURE_ORDER))
