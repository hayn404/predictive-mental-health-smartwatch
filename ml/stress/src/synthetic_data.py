"""
Seren ML Pipeline — Synthetic Data Generator
================================================
Generates physiologically realistic synthetic biometric data
matching the WESAD feature structure. Used when real WESAD
data is not available (e.g., in CI/CD or development).

The generator models known physiological relationships:
  - Stress: lower HRV (RMSSD, SDNN), higher HR, higher LF/HF
  - Baseline: normal HRV ranges, lower HR, balanced autonomic tone
  - Inter-subject variability in baseline physiology
"""

import numpy as np
import pandas as pd
import logging
from typing import Tuple

from features import FEATURE_ORDER

logger = logging.getLogger(__name__)

# Physiological parameter distributions (mean, std)
# Based on published WESAD and HRV literature ranges

BASELINE_PARAMS = {
    # Time-domain HRV
    "meanRR": (850, 100),       # ~70 BPM
    "sdnn": (55, 20),           # Normal SDNN
    "rmssd": (45, 18),          # Normal RMSSD
    "pnn50": (25, 15),
    "pnn20": (55, 15),
    "hrMean": (72, 10),
    "hrStd": (4, 2),
    "hrRange": (15, 7),
    "cvRR": (0.065, 0.02),

    # Frequency-domain HRV
    "vlfPower": (1200, 600),
    "lfPower": (800, 400),
    "hfPower": (600, 350),
    "lfHfRatio": (1.5, 0.7),
    "totalPower": (2600, 1000),
    "lfNorm": (55, 10),
    "hfNorm": (45, 10),

    # Non-linear HRV
    "sd1": (32, 13),
    "sd2": (55, 18),
    "sd1sd2Ratio": (0.6, 0.15),
    "sampleEntropy": (1.6, 0.4),
    "dfaAlpha1": (1.0, 0.2),

    # Temperature
    "tempMean": (33.5, 0.8),
    "tempSlope": (0.001, 0.01),
    "tempStd": (0.15, 0.08),
    "tempRange": (0.5, 0.2),

    # Activity
    "accelMagnitudeMean": (1.01, 0.02),
    "accelMagnitudeStd": (0.03, 0.015),
    "stepCount": (5, 8),
    "activityType": (0, 0),     # Mostly sedentary
}

# Stress shifts: how features change under stress
STRESS_SHIFTS = {
    "meanRR": -120,     # HR increases -> RR decreases
    "sdnn": -20,        # HRV drops
    "rmssd": -18,       # HRV drops significantly
    "pnn50": -12,
    "pnn20": -15,
    "hrMean": +15,      # Heart rate increases
    "hrStd": +2,
    "hrRange": +8,
    "cvRR": -0.015,

    "vlfPower": +200,
    "lfPower": +300,    # Sympathetic activation
    "hfPower": -200,    # Vagal withdrawal
    "lfHfRatio": +1.5,  # Sympathovagal imbalance
    "totalPower": -100,
    "lfNorm": +12,
    "hfNorm": -12,

    "sd1": -12,
    "sd2": -10,
    "sd1sd2Ratio": -0.1,
    "sampleEntropy": -0.3,
    "dfaAlpha1": +0.15,

    "tempMean": -0.4,   # Vasoconstriction
    "tempSlope": -0.015,
    "tempStd": +0.05,
    "tempRange": +0.2,

    "accelMagnitudeMean": +0.01,
    "accelMagnitudeStd": +0.01,
    "stepCount": +2,
    "activityType": 0,
}


def generate_synthetic_dataset(
    n_subjects: int = 15,
    windows_per_subject: int = 40,
    stress_ratio: float = 0.35,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a synthetic dataset matching WESAD feature structure.

    Each subject gets a unique physiological "personality" (baseline shift),
    then stress/baseline windows are generated with realistic noise.

    Args:
        n_subjects: Number of synthetic subjects
        windows_per_subject: Windows per subject
        stress_ratio: Fraction of windows that are stressed
        seed: Random seed for reproducibility

    Returns:
        DataFrame with FEATURE_ORDER columns + subject, label, stress_binary
    """
    rng = np.random.RandomState(seed)
    rows = []

    logger.info(f"Generating synthetic data: {n_subjects} subjects x {windows_per_subject} windows")

    for subj_idx in range(n_subjects):
        subject_id = f"S{subj_idx + 2}"  # Match WESAD naming (S2, S3, ...)

        # Subject-level baseline shift (inter-individual variability)
        subject_shift = {
            feat: rng.normal(0, params[1] * 0.3)
            for feat, params in BASELINE_PARAMS.items()
        }

        n_stress = int(windows_per_subject * stress_ratio)
        n_baseline = windows_per_subject - n_stress
        labels = [0] * n_baseline + [1] * n_stress
        rng.shuffle(labels)

        for label in labels:
            row = {}
            for feat in FEATURE_ORDER:
                base_mean, base_std = BASELINE_PARAMS[feat]

                # Apply subject-level shift
                mean = base_mean + subject_shift[feat]

                # Apply stress shift if stressed
                if label == 1:
                    mean += STRESS_SHIFTS.get(feat, 0)

                # Sample with noise
                value = rng.normal(mean, base_std * 0.4)

                # Clamp physiologically impossible values
                value = _clamp_feature(feat, value)
                row[feat] = value

            row["subject"] = subject_id
            row["stress_binary"] = label
            row["label"] = 2 if label == 1 else 1  # WESAD: 2=stress, 1=baseline
            row["label_name"] = "stress" if label == 1 else "baseline"
            rows.append(row)

    df = pd.DataFrame(rows)

    logger.info(f"Generated {len(df)} samples: {(df['stress_binary']==0).sum()} baseline, {(df['stress_binary']==1).sum()} stress")
    logger.info(f"Subjects: {df['subject'].nunique()}")

    return df


def _clamp_feature(feat: str, value: float) -> float:
    """Clamp feature values to physiologically valid ranges."""
    clamps = {
        "meanRR": (300, 1500),
        "sdnn": (5, 200),
        "rmssd": (5, 200),
        "pnn50": (0, 100),
        "pnn20": (0, 100),
        "hrMean": (40, 180),
        "hrStd": (0.5, 30),
        "hrRange": (2, 60),
        "cvRR": (0.01, 0.3),
        "vlfPower": (0, 10000),
        "lfPower": (0, 10000),
        "hfPower": (0, 10000),
        "lfHfRatio": (0.1, 20),
        "totalPower": (0, 30000),
        "lfNorm": (10, 90),
        "hfNorm": (10, 90),
        "sd1": (2, 150),
        "sd2": (5, 200),
        "sd1sd2Ratio": (0.1, 3),
        "sampleEntropy": (0, 3),
        "dfaAlpha1": (0.3, 2),
        "tempMean": (28, 38),
        "tempSlope": (-0.1, 0.1),
        "tempStd": (0.01, 1),
        "tempRange": (0.05, 3),
        "accelMagnitudeMean": (0.9, 3),
        "accelMagnitudeStd": (0.005, 1),
        "stepCount": (0, 200),
        "activityType": (0, 3),
    }
    if feat in clamps:
        lo, hi = clamps[feat]
        return max(lo, min(hi, value))
    return value


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = generate_synthetic_dataset()
    print(df.describe())
    print(f"\nShape: {df.shape}")
    print(f"\nStress distribution:\n{df['stress_binary'].value_counts()}")
